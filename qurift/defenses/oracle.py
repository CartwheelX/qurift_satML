"""Raw model adapter with exact output and measurement parity."""
from __future__ import annotations

import threading
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from .base import PredictionBatch, PredictionOracle


def _resolve_head(model: torch.nn.Module) -> torch.nn.Module:
    linear = getattr(model, "linear", None)
    if isinstance(linear, torch.nn.Module):
        return linear
    head = getattr(model, "head", None)
    if isinstance(head, torch.nn.Module):
        return head
    raise NotImplementedError(
        "RawOracle requires a final `linear` or `head` module so it can capture "
        "the post-measurement representation and logits from the same forward pass."
    )


class RawOracle(PredictionOracle):
    """Adapt an existing model without changing its forward implementation.

    Hooks capture the input and output of the final classifier.  For QFCModel,
    these are respectively the post-PQC Pauli-Z expectation vector and logits.
    The model's returned log-probabilities are retained byte-for-byte as
    ``model_output`` (apart from ``detach``), enabling strict parity tests.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        output_kind: str = "log_probabilities",
        parity_atol: float = 2e-5,
        decision_threshold: Optional[float] = None,
    ) -> None:
        if output_kind not in {"log_probabilities", "logits", "probabilities"}:
            raise ValueError(f"Unsupported output_kind={output_kind!r}")
        self.model = model
        self.head = _resolve_head(model)
        self.output_kind = output_kind
        self.parity_atol = float(parity_atol)
        if decision_threshold is not None and not 0.0 < float(decision_threshold) < 1.0:
            raise ValueError("binary decision threshold must be strictly between zero and one")
        self.decision_threshold = (
            None if decision_threshold is None else float(decision_threshold)
        )
        self._lock = threading.RLock()

    @property
    def config(self) -> Mapping[str, Any]:
        config = {
            "defense": "none",
            "oracle": "RawOracle",
            "output_kind": self.output_kind,
            "head": self.head.__class__.__name__,
            "parity_atol": self.parity_atol,
            "label_rule": (
                "argmax"
                if self.decision_threshold is None
                else "binary_probability_threshold"
            ),
        }
        if self.decision_threshold is not None:
            config["decision_threshold"] = self.decision_threshold
        return config

    def predict(
        self,
        inputs: torch.Tensor,
        *,
        query_ids: Optional[Sequence[str]] = None,
    ) -> PredictionBatch:
        if query_ids is not None and len(query_ids) != int(inputs.shape[0]):
            raise ValueError("query_ids length does not match input batch")
        captured: Dict[str, torch.Tensor] = {}

        def capture_input(_module: torch.nn.Module, args: Any) -> None:
            if not args or not torch.is_tensor(args[0]):
                raise RuntimeError("final classifier did not receive a tensor input")
            captured["measurement"] = args[0]

        def capture_output(_module: torch.nn.Module, _args: Any, output: Any) -> None:
            if not torch.is_tensor(output):
                raise RuntimeError("final classifier did not return a tensor")
            captured["logits"] = output

        with self._lock:
            pre_handle = self.head.register_forward_pre_hook(capture_input)
            post_handle = self.head.register_forward_hook(capture_output)
            try:
                self.model.eval()
                with torch.no_grad():
                    model_output = self.model(inputs)
            finally:
                pre_handle.remove()
                post_handle.remove()

        if "measurement" not in captured or "logits" not in captured:
            raise RuntimeError("final classifier hooks were not called during model inference")
        model_output = model_output.detach()
        logits = captured["logits"].detach()
        measurement = captured["measurement"].detach()

        if self.output_kind == "log_probabilities":
            log_probabilities = model_output
            probabilities = model_output.exp()
        elif self.output_kind == "logits":
            logits = model_output
            log_probabilities = torch.log_softmax(model_output, dim=1)
            probabilities = log_probabilities.exp()
        else:
            probabilities = model_output
            log_probabilities = probabilities.clamp_min(
                torch.finfo(probabilities.dtype).tiny
            ).log()

        head_probabilities = torch.softmax(logits, dim=1)
        if not torch.allclose(
            probabilities, head_probabilities, atol=self.parity_atol, rtol=self.parity_atol
        ):
            maximum = float((probabilities - head_probabilities).abs().max().item())
            raise RuntimeError(
                "model output is inconsistent with captured classifier logits; "
                f"maximum probability difference={maximum:.3e}"
            )
        if self.decision_threshold is None:
            labels = probabilities.argmax(dim=1)
        else:
            if probabilities.shape[1] != 2:
                raise ValueError("configured decision threshold requires a binary model")
            labels = (probabilities[:, 1] >= self.decision_threshold).long()
        result = PredictionBatch(
            model_output=model_output,
            logits=logits,
            log_probabilities=log_probabilities,
            probabilities=probabilities,
            labels=labels,
            measurement=measurement,
            metadata=dict(self.config),
        )
        return result.validate(atol=max(self.parity_atol, 1e-5))
