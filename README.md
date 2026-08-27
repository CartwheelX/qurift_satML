
<p align="center">
  <img src="qurift_logo_1.png" alt="QuRiFT logo" width="420"/>
</p>

> **SaTML development branch.** This repository now contains the frozen
> cross-domain, low-FPR, geometry, frozen-noise, and fresh privacy-selector
> study. Start with [the SaTML protocol](docs/SATML_PROTOCOL.md) and
> [the executable runbook](docs/SATML_RUNBOOK.md). The completed NeurIPS
> rebuttal outputs remain as legacy evidence and are not silently mixed with
> new SaTML results; new outputs use the `satml_*` directories.

> **PETS defense extension.** A separate, fresh-seed defense study is available
> under `qurift/defenses`, `pets_tools`, and `commands/pets_*`. It does not
> overwrite the frozen SaTML evidence. Read the
> [defense protocol](docs/PETS_DEFENSE_PROTOCOL.md),
> [provenance record](docs/PETS_DEFENSE_PROVENANCE.md), and
> [MemGQ specification](docs/PETS_MEMGQ.md) before launching the one-block
> pilot. The full launcher is deliberately gated until pilot settings are
> recorded and frozen.

For the current completed pilot, run the corrected evaluation and utility-only
tuning stages before confirmation:

```bash
export QURIFT_PETS_STICKY_SECRET='the-same-private-pilot-secret'
nohup bash commands/pets_run_credit_corrections.sh \
  > pets_logs/corrections_launcher.log 2>&1 &
echo $!
```

The correction stage recoverably archives the old pilot result directories and
reuses unaffected target/reference checkpoints while recomputing evaluations
with task-label-matched MIA pools and full-test utility. It also adds matched
output-defense LiRA and nearby-query controls. The tuning stage trains new,
versioned Watkins-style DP checkpoints (RMSprop, batch 32, 30 epochs, fixed
clipping norm 1), selects binary operating thresholds on validation utility,
and selects L2/DP settings without reading membership-attack outcomes. It then
freezes `pets_b02`--`pets_b05`. Exact definitions and the subsequent full-run
command are in the defense protocol.

**QuRiFT** (**Quantum Risk and Inference Fault-line Tracer**) is a controlled audit framework for studying **structural privacy leakage in quantum machine learning (QML)**.

QuRiFT is designed to answer a specific question: **how much of membership-inference risk in QML is induced by circuit structure, especially the non-trainable classical-to-quantum encoder?**

Rather than treating privacy leakage only as a consequence of trainable model capacity, QuRiFT performs controlled interventions over QML design choices, keeps the training protocol fixed within each experimental family, and records utility, overfitting, and membership-inference signals.

---

## What QuRiFT Provides

QuRiFT provides an end-to-end experimental pipeline for:

- running controlled QML architecture sweeps,
- varying encoder and ansatz design factors,
- training QNN, HQNN, and QCNN-style models,
- exporting prediction vectors for member and non-member samples,
- selecting stress, baseline, and hard target configurations,
- training black-box membership-inference attacks,
- generating CSV summaries for paper tables and analysis.

The frozen SaTML extension adds paired cross-domain studies on Credit-default,
Fashion-MNIST, and Breast Cancer Wisconsin Diagnostic (WDBC), with
training-only tabular preprocessing, seeded image partitions, direct
post-encoder geometry, dataset-specific low-FPR reporting, and generated
Markdown/LaTeX tables plus publication figures. Exact commands and monitoring
instructions are maintained in [docs/SATML_RUNBOOK.md](docs/SATML_RUNBOOK.md).
The SaTML label-output evaluation separates an always-defined correctness
baseline from a fixed-budget hard-label HopSkipJump-style boundary search; the
earlier validation-anchor chord outputs are retained only as diagnostics. The
corrected protocol is documented in
[docs/SATML_LABEL_ONLY_HSJ.md](docs/SATML_LABEL_ONLY_HSJ.md).
After the 96-target Credit factorial completes, the resumable unattended
wrapper `commands/satml_run_all_remaining.sh` executes every remaining required
stage, final artifact generation, and repository verification in order.

The noise extension is split into three auditable studies under one frozen IBM
backend-calibration snapshot: N1 evaluates all 36 retained MNIST structural
checkpoints; N2 isolates API query count from shots per query; and N3 evaluates
matched LiRA reference models through the same ideal/noisy oracle. Repeated
queries are processed by the trained nonlinear head individually before the
returned probability vectors are averaged. See the frozen definitions in
[docs/SATML_PROTOCOL.md](docs/SATML_PROTOCOL.md).

The framework is intended for controlled privacy auditing, not for claiming hardware-level leakage. The original sweep uses noiseless simulation to isolate representation effects. The NeurIPS rebuttal additionally includes a targeted finite-shot check using local Aer simulation with an IBM-backend-derived noise model; it is not execution on quantum hardware.

---

## Relationship to TorchQuantum

QuRiFT builds around [TorchQuantum](https://github.com/mit-han-lab/torchquantum) as the quantum primitive layer. TorchQuantum provides PyTorch-native quantum devices, gates, differentiable circuit execution, measurements, and GPU-backed simulation.

QuRiFT adds the privacy-audit layer on top of those primitives:

- experiment drivers,
- feature-map and ansatz configuration logic,
- QNN/HQNN/QCNN model wrappers,
- sweep orchestration,
- result logging,
- target-table construction,
- prediction-vector export,
- membership-inference attack training.

TorchQuantum should be credited as the upstream quantum simulation and circuit-execution foundation. QuRiFT is the audit and analysis framework built around it.

---

## Repository Structure

```text
QuRiFT/
├── experiments/
│   ├── qurift_main.py
│   ├── full_sweep_qnn_moons.py
│   ├── full_sweep_qnn_circles.py
│   ├── full_sweep_qnn_blobs.py
│   ├── run_mnist_sweep_qnn.py
│   ├── run_mnist_sweep_hqnn.py
│   ├── run_mnist_sweep_qcnn.py
│   └── gen_results/
│       ├── make_runid_tables_for_mia.py
│       ├── qnn_qcnn_hqnn_models_comp_mnist.py
│       ├── run_selected_configs_for_mia.py
│       ├── train_mia_attack.py
│       └── run_train_mia_attack_cvholdout_multigpu.py
├── data/                           # Downloaded datasets (ignored; checksums are validated)
├── satml_targets/                  # Frozen Credit, Fashion-MNIST, and WDBC manifests
├── satml_tools/                    # Acquisition, validation, inference, and artifact tools
├── commands/                       # SaTML and retained rebuttal workflow wrappers
├── reviewer_targets/               # Prespecified confirmatory target tables
├── reviewer_tools/                 # Training, attack, geometry, noise, and analysis tools
├── reviewer_runs/                  # Exported target runs and attack payloads
├── reviewer_results/               # Rebuttal summaries, tables, figures, and responses
├── requirements.txt
├── setup.py
└── README.md
```

Generated outputs such as checkpoints, sweep folders, CSV summaries, plots, and attack outputs are intentionally kept out of Git unless they are curated paper artifacts.

---

## Main Entry Point

The central experiment driver is:

```bash
python experiments/qurift_main.py
```

After installation, the same driver can also be called through the console command:

```bash
qurift
```

Most sweep scripts in `experiments/` are wrappers around `experiments/qurift_main.py` with pre-defined experimental grids.

---

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/CartwheelX/qurift_satML.git
cd qurift_satML
pip install --editable .
```

This installs the QuRiFT package and exposes the console command:

```bash
qurift
```

If you prefer to install dependencies separately:

```bash
pip install -r requirements.txt
pip install --editable . --no-deps
```

The installable distribution name is:

```text
qurift
```

Current package version:

```text
0.1.0
```

---

## Dependencies

Recommended environment:

- Python `>=3.7, <=3.9`
- PyTorch `>=1.8.0`
- `configargparse >= 0.14`
- CUDA-enabled NVIDIA GPU for larger sweeps and target-model retraining

Python 3.10 may cause compatibility issues with some TorchQuantum/Qiskit dependency combinations, including issues around the `concurrent` package in older stacks. Python 3.8 or 3.9 is recommended for reproducibility.

The experiment code imports TorchQuantum primitives through:

```python
import torchquantum as tq
```

---

## Quick Start

Run a small synthetic Moons experiment:

```bash
python experiments/qurift_main.py \
  --dataset moons \
  --model-type qnn \
  --n-wires 4 \
  --depth 2 \
  --epochs 1 \
  --train_target \
  --extra-feats
```

Equivalent installed command:

```bash
qurift \
  --dataset moons \
  --model-type qnn \
  --n-wires 4 \
  --depth 2 \
  --epochs 1 \
  --train_target \
  --extra-feats
```

On Windows PowerShell, replace Linux/macOS line continuations `\` with `^`.

---

## Example: Export Prediction Vectors for MIA

The following example trains a QNN target model on Moons and exports prediction-vector data for membership-inference analysis:

```bash
python experiments/qurift_main.py \
  --dataset moons \
  --model-type qnn \
  --n-wires 4 \
  --depth 6 \
  --vector-train 50 \
  --vector-valid 50 \
  --vector-test 50 \
  --batch-size 8 \
  --epochs 100 \
  --moons-noise 0.3 \
  --fm-kind z \
  --fm-z-pad-mode wrap \
  --fm-z-reps 1 \
  --train_target \
  --extra-feats \
  --export-attack-data \
  --target-model-path checkpoints/moons_qnn.pt \
  --attack-data-out audit_outputs/moons_qnn_attack_data.pt
```

The exported attack data contains prediction vectors and labels needed to train black-box membership-inference attacks.

---

## Sweep Drivers

The main sweep drivers are:

```text
experiments/full_sweep_qnn_moons.py
experiments/full_sweep_qnn_circles.py
experiments/full_sweep_qnn_blobs.py
experiments/run_mnist_sweep_qnn.py
experiments/run_mnist_sweep_hqnn.py
experiments/run_mnist_sweep_qcnn.py
```

These scripts launch controlled sweeps across synthetic datasets and MNIST model families. They call `experiments/qurift_main.py` with different structural configurations.

---

## Structural Factors Swept by QuRiFT

QuRiFT varies QML structure while keeping the data and training protocol fixed within each experimental family. The main factors are:

| Factor | Example values / flags | Purpose |
|---|---|---|
| Feature-map family | `z`, `zz`, `pauli`, `eff_su2` | Tests encoder-induced representation effects |
| Feature-map repetitions | `--fm-*-reps` | Repeatedly injects input-dependent structure |
| Padding mode | e.g., `wrap` | Controls feature-to-qubit mapping when dimensions do not match |
| Feature-map entanglement | e.g., `linear`, `full` | Controls encoder entanglement topology |
| Circuit width | `--n-wires` | Changes number of qubits/wires |
| Variational depth | `--depth` | Changes trainable ansatz capacity |
| Q-layer entanglement | `--qlayer-ent-kind` | Controls trainable-layer connectivity |
| Q-layer two-qubit gate | `--qlayer-twoq-op` | Tests trainable entangling operation choice |
| Model family | `qnn`, `hqnn`, `qcnn`, `mlp_qnn` | Compares QML architecture families |

A key distinction in the paper is between **feature-map repetitions** and **variational depth**. Feature-map repetitions re-inject the input through the fixed encoder, similar in spirit to data re-uploading. Variational depth mainly increases the number of trainable operations after encoding.

---

## Model Families

QuRiFT supports the following model families:

- **`qnn`**: Dense quantum neural network with a quantum encoder, variational circuit, measurement, and classical classifier.
- **`hqnn`**: Hybrid CNN-QNN model with a trainable classical bottleneck before the quantum encoder and an MLP head after measurement.
- **`qcnn`**: Quantum-filter/quanvolutional front end with local quantum processing before the downstream encoder and classifier.
- **`mlp_qnn`**: Optional classical/MLP-style comparison path.

Synthetic benchmarks include:

```text
Moons, Circles, Blobs
```

The MNIST experiments use a four-class subset:

```text
{0, 1, 3, 8}
```

MNIST inputs are represented using compact `1x16` features before the main quantum encoder.

---

## MNIST Data Cache

The repository includes a small MNIST cache under:

```text
data/MNIST/raw
```

MNIST experiments use:

```python
root="./data"
```

Fresh clones can therefore run MNIST smoke tests without downloading MNIST again. If the cache is removed, TorchVision will attempt to download the dataset.

---

## Metrics Recorded

For every configuration, QuRiFT records utility and privacy-relevant signals, including:

- train, validation, and test loss,
- train, validation, and test accuracy,
- train-test accuracy gap,
- prediction vectors for member and non-member samples,
- output-derived attack features such as loss, entropy, confidence, margin, and correctness.

The train-test accuracy gap is used as a structural proxy for memorization pressure. Membership inference is then evaluated directly using exported prediction vectors.

---

## Membership-Inference Threat Model

QuRiFT evaluates membership inference in a strict black-box setting. The attacker observes only the target model's prediction vector for a queried sample.

The attacker does **not** access:

- model parameters,
- gradients,
- optimizer state,
- quantum states,
- circuit internals at inference time,
- target training data.

The attack objective is to distinguish member samples from non-member samples using output-derived signals.

---

## Result and MIA Workflow

The `experiments/gen_results/` directory tracks the scripts needed for paper-table generation and membership-inference attack training. Generated CSVs, plots, checkpoints, and attack outputs are ignored by Git unless intentionally curated.

A typical workflow is:

```text
1. Run QML sweeps.
2. Copy or collect the resulting sweep summary CSVs into experiments/gen_results/.
3. Generate target-configuration tables for MIA.
4. Retrain/export selected target models and prediction vectors.
5. Train membership-inference attacks.
6. Aggregate attack results for paper tables and plots.
```

---

## Step 1: Run Sweeps

Run the desired synthetic and MNIST sweep drivers, for example:

```bash
python experiments/full_sweep_qnn_moons.py
python experiments/full_sweep_qnn_circles.py
python experiments/full_sweep_qnn_blobs.py
python experiments/run_mnist_sweep_qnn.py
python experiments/run_mnist_sweep_hqnn.py
python experiments/run_mnist_sweep_qcnn.py
```

Each sweep creates a timestamped output directory. Examples include:

```text
sweep_full_pipeline_moons_<timestamp>/
sweep_full_pipeline_circles_<timestamp>/
sweep_full_pipeline_blobs_<timestamp>/
mnist_extensive_sweep_qnn_<timestamp>/
hqnn_sweep_<timestamp>/
qcnn_sweep_100_<timestamp>/
```

The corresponding CSV summaries should be copied into `experiments/gen_results/` before target-table generation.

Expected MNIST architecture summary names:

```text
experiments/gen_results/qnn_extensive_results.csv
experiments/gen_results/hqnn_extensive_results.csv
experiments/gen_results/qcnn_extensive_results.csv
```

Synthetic sweep summaries are similarly collected from their generated sweep directories.

---

## Step 2: Generate Run-ID Tables for Selected MIA Targets

For a selected synthetic setup, use:

```bash
python experiments/gen_results/make_runid_tables_for_mia.py \
  --dataset Moons --arch QNN \
  --out-dir experiments/gen_results/paper_arch_compare/retrain_grid \
  --fix "fm_kind=zz,fm_op_eff=rzz,n_wires=3,ql_ent=full,ql_op=crz,pad_mode=wrap,fm_ent=linear" \
  --reps "1,2,3,4,5" \
  --depths "2,3,4,5,6" \
  --train-min 0.99 --gap-lo 0.25 --gap-hi 0.30 \
  --prefer-low-test \
  --fallback-if-empty
```

This script filters sweep results and constructs run-id tables for target-model retraining and MIA export.

---

## Step 3: Generate Matched Target Tables

Generate matched target-configuration CSVs for synthetic QNN and MNIST architecture comparisons:

```bash
python experiments/gen_results/qnn_qcnn_hqnn_models_comp_mnist.py
```

This step expects the sweep summary CSVs to be present under `experiments/gen_results/` with consistent architecture names.

Typical output tables include:

```text
experiments/gen_results/paper_arch_compare/synthetic_qnn_targets_table.csv
experiments/gen_results/paper_arch_compare/mnist_matched_runids_table.csv
```

These tables are later consumed by the target retraining and prediction-vector export script.

---

## Step 4: Retrain Selected Targets and Export Attack Data

Train/export selected target models and prediction-vector attack data for synthetic QNN targets:

```bash
python experiments/gen_results/run_selected_configs_for_mia.py \
  --targets experiments/gen_results/paper_arch_compare/synthetic_qnn_targets_table.csv \
  --out experiments/gen_results/paper_arch_compare/saved_models_for_mia \
  --save-model
```

Train/export selected target models and prediction-vector attack data for matched MNIST targets:

```bash
python experiments/gen_results/run_selected_configs_for_mia.py \
  --targets experiments/gen_results/paper_arch_compare/mnist_matched_runids_table.csv \
  --out experiments/gen_results/paper_arch_compare/saved_models_for_mia \
  --save-model
```

The output directory stores trained target checkpoints and exported attack-data files.

---

## Step 5: Train Membership-Inference Attacks

Train MLP membership-inference attacks on a single GPU:

```bash
python experiments/gen_results/train_mia_attack.py \
  --attack-data-dir experiments/gen_results/paper_arch_compare/saved_models_for_mia \
  --out experiments/gen_results/paper_arch_compare/mia_results \
  --test-ratio 0.2 --cv-folds 5 \
  --tune --n-trials 30 --max-epochs 200 --patience 15 \
  --device cuda --seed 42
```

Train attacks with the multi-GPU launcher:

```bash
python experiments/gen_results/run_train_mia_attack_cvholdout_multigpu.py \
  --attack-data-dir experiments/gen_results/paper_arch_compare/saved_models_for_mia \
  --out experiments/gen_results/paper_arch_compare/mia_results_multiGPU \
  --launcher \
  --device cuda \
  --tune --n-trials 120 --max-epochs 300 --patience 25 \
  --test-ratio 0.2 --cv-folds 5 \
  --jobs-per-gpu 4 \
  --cpu-threads 1 \
  --resume \
  --gpus 2,3,4,5,6 \
  --summary-only
```

---

## Important Generated Files

The following files are commonly used in the paper analysis pipeline:

```text
experiments/gen_results/paper_arch_compare/synthetic_qnn_targets_table.csv
experiments/gen_results/paper_arch_compare/mnist_matched_runids_table.csv
```

They are generated from sweep summary CSVs and are used as target lists for `run_selected_configs_for_mia.py`.

The target retraining/export step then produces saved models and attack-data files under:

```text
experiments/gen_results/paper_arch_compare/saved_models_for_mia/
```

The MIA training step produces attack results under directories such as:

```text
experiments/gen_results/paper_arch_compare/mia_results/
experiments/gen_results/paper_arch_compare/mia_results_multiGPU/
```

---

## NeurIPS 2026 Rebuttal Reproduction

The rebuttal supplements the submission's broad exploratory sweep with a focused confirmatory workflow. Its principal components are:

- a `3 × 2 × 2` MNIST-QNN factorial over feature-map family, encoder repetitions, and variational depth;
- three independently initialized target models per structural configuration (`36` targets total);
- separate scalar threshold attacks, a learned prediction-vector attack, calibrated LiRA, and a class-label-only boundary attack;
- direct post-encoder fidelity-kernel geometry;
- a five-configuration finite-shot and IBM-backend-derived noise check; and
- complete-wrapper QNN, HQNN, QCNN, and small classical MLP controls.

All commands below are run from the repository root. The target tables under `reviewer_targets/` are the committed, prespecified tables used for the rebuttal. Rebuilding those tables is unnecessary and requires the original broad-sweep summaries, which are not part of the clean-clone reproduction path.

The historical `reviewer_tools/apply_qurift_main_reviewer_patch.py` records how the experiment driver was updated during rebuttal development. Do not apply it again: the checked-in `experiments/qurift_main.py` already contains those changes.

### Rebuttal environment and live logs

Activate the Python environment, install the project, and create the output directories:

```bash
cd /path/to/quarift_neurips_rebutal_2
pip install --editable .
mkdir -p reviewer_logs reviewer_runs reviewer_results
```

The experiment driver enables deterministic PyTorch behavior. Set the cuBLAS workspace configuration before starting Python so CUDA matrix multiplication also follows the deterministic configuration:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Commands below use unbuffered Python and `tee`, so progress remains visible while a complete log is written. In an additional terminal, a log can be followed with:

```bash
tail -f reviewer_logs/multiseed_factorial.log
```

All long-running launchers support `--resume`; rerunning the same command skips completed targets.

### 1. Train the 36-target confirmatory factorial

This launches 12 structural configurations across model seeds 43, 44, and 45:

```bash
set -o pipefail
python -u reviewer_tools/run_multiseed_factorial.py \
  --targets reviewer_targets/multiseed_factorial_targets.csv \
  --repo-root . \
  --out reviewer_runs \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu 2 \
  --cpu-threads 2 \
  --resume \
  2>&1 | tee reviewer_logs/multiseed_factorial.log
```

The target checkpoints and exported member/non-member attack payloads are written under `reviewer_runs/multiseed_factorial/`. Membership labels in the raw target payload are normalized by the reviewer analysis tools to `1=member, 0=nonmember`.

### 2. Train the architecture-control targets

The architecture experiment compares complete QNN, HQNN, QCNN, and small classical MLP wrappers across three structural roles and three target-model seeds:

```bash
python -u reviewer_tools/run_architecture_controls.py \
  --targets reviewer_targets/architecture_control_targets.csv \
  --repo-root . \
  --out reviewer_runs \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu 2 \
  --cpu-threads 2 \
  --resume \
  2>&1 | tee reviewer_logs/architecture_controls.log
```

These are complete-wrapper controls, not isolated causal comparisons of only the quantum circuit. Outputs are written under `reviewer_runs/architecture_control/`.

### 3. Compute direct post-encoder geometry

This computes the pure-state fidelity kernel immediately after the fixed encoder and reports class-similarity separation, centered kernel-label alignment, effective rank, and train/test MMD²:

```bash
python -u reviewer_tools/run_multiseed_geometry.py \
  --targets reviewer_targets/geometry_targets.csv \
  --repo-root . \
  --out-dir reviewer_results/geometry_multiseed \
  --seeds 43,44,45 \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu 1 \
  --cpu-threads 2 \
  --n-train 100 \
  --n-test 100 \
  --batch-size 32 \
  --bootstrap 5000 \
  --bootstrap-seed 2026 \
  --resume \
  2>&1 | tee reviewer_logs/geometry_multiseed.log
```

Principal outputs are under `reviewer_results/geometry_multiseed/`, including `geometry_raw.csv`, `geometry_summary.csv`, `geometry_repetition_effects.csv`, and `repetition_integrity.csv`.

### 4. Extract factorial metrics, resources, and threshold attacks

Extract target utility/generalization metrics:

```bash
python -u reviewer_tools/extract_retrained_metrics.py \
  --attack-data-dir reviewer_runs/multiseed_factorial \
  --targets reviewer_targets/multiseed_factorial_targets.csv \
  --out-dir reviewer_results/factorial_metrics
```

Record trainable-parameter and main-circuit gate counts:

```bash
python -u reviewer_tools/count_model_resources.py \
  --run-root reviewer_runs/multiseed_factorial \
  --targets reviewer_targets/multiseed_factorial_targets.csv \
  --out-dir reviewer_results/factorial_resources \
  --fail-on-missing-exact
```

Run the six scalar attacks separately—loss, confidence, maximum probability, entropy, margin, and correctness—with five-fold cross-fitted thresholds and record-bootstrap AUC intervals:

```bash
python -u reviewer_tools/threshold_mia_bootstrap.py \
  --attack-data-dir reviewer_runs/multiseed_factorial \
  --targets reviewer_targets/multiseed_factorial_targets.csv \
  --out-dir reviewer_results/factorial_threshold_mia \
  --bootstrap 10000 \
  --bootstrap-chunk-size 2048 \
  --bootstrap-seed 2026 \
  --threshold-folds 5 \
  --threshold-seed 2026 \
  --fprs 0.05,0.10 \
  2>&1 | tee reviewer_logs/factorial_threshold_mia.log
```

### 5. Run the learned prediction-vector MIA

The wrapper trains the prediction-vector-plus-statistics attacker with attacker seeds 41, 42, and 43. It consumes completed attack payloads under `reviewer_runs/` and can therefore cover both factorial and architecture targets:

```bash
bash run_learned_mia.sh
```

Outputs are written to `reviewer_results/learned_mia_seed41/`, `learned_mia_seed42/`, and `learned_mia_seed43/`. The corresponding live logs are stored under `reviewer_logs/`.

### 6. Run calibrated LiRA and label-only baselines

Run both additional attack families:

```bash
bash commands/run_missing_mia_baselines.sh
```

The LiRA experiment trains 16 same-architecture reference models per structural configuration, with each candidate included in exactly eight references. The label-only attack estimates decision-boundary distance from changed-label validation anchors and consumes predicted class labels only. Outputs are written under:

```text
reviewer_results/lira_reference_mia/
reviewer_results/label_only_boundary/
```

The complete baseline protocol and provenance are documented in `reviewer_tools/README_MISSING_MIA_BASELINES.md`.

### 7. Run finite-shot and backend-derived noisy evaluation

This experiment queries IBM backend metadata only to construct a local Aer noise model; it does not submit jobs to quantum hardware. The completed rebuttal environment used:

```bash
pip install qiskit==1.4.3 qiskit-aer==0.17.1 qiskit-ibm-runtime==0.43.1
```

The five-configuration target selection can be reconstructed deterministically from the factorial target table:

```bash
python -u reviewer_tools/build_noisy_sanity_targets.py \
  --factorial-targets reviewer_targets/multiseed_factorial_targets.csv \
  --out-dir reviewer_targets \
  --model-seeds 43,44,45
```

Export IBM credentials without placing them in a command-line argument or source file:

```bash
export QISKIT_IBM_TOKEN="<your-IBM-Quantum-token>"
export QISKIT_IBM_INSTANCE="<your-IBM-instance-CRN>"
export QURIFT_NOISE_BACKEND="ibm_kingston"
unset QURIFT_IBM_ACCOUNT_NAME
```

Quotes are recommended. Never commit actual tokens or CRNs. If using a genuinely saved Qiskit account instead, omit the token/instance exports and set `QURIFT_IBM_ACCOUNT_NAME` to that saved account's name.

Verify that the backend-derived noise model loads:

```bash
python -u reviewer_tools/probe_ibm_backend_noise.py \
  --backend-name "$QURIFT_NOISE_BACKEND" \
  --require-noise \
  --out reviewer_results/noisy_sanity/backend_probe.json
```

Run all five selected structural configurations, all three target-model seeds, shot counts 128/512/1024, and simulator seeds 0–9 on the DGX GPUs:

```bash
bash commands/run_all_seeds_dgx.sh
```

For a CPU-only run, use:

```bash
bash commands/run_all_seeds_cpu.sh
```

Validate sample consistency and combine the noisy results with bootstrap summaries:

```bash
bash commands/combine_and_validate.sh
```

Raw and combined outputs are written under `reviewer_results/noisy_sanity/`. The reported condition is backend-derived local Aer simulation, not hardware execution or a claim about all devices and calibrations.

### 8. Generate consolidated tables, figures, and reviewer responses

The artifact wrapper extracts architecture metrics, audits factorial and architecture resources, runs the architecture loss-threshold analysis, performs paired architecture comparisons, fits the descriptive gap–AUC models, and generates the consolidated tables and figures:

```bash
bash commands/generate_reviewer_artifacts.sh
```

Generate the completed reviewer-response documents and the LiRA repetition/reference-bank audit:

```bash
python -u reviewer_tools/generate_final_reviewer_responses.py \
  --results-root reviewer_results \
  --out-dir reviewer_results/reviewer_artifacts/final_responses \
  --bootstrap 5000 \
  --bootstrap-seed 2026

python -u reviewer_tools/generate_lira_repetition_audit.py \
  --metrics reviewer_results/factorial_metrics/retrained_target_metrics_raw.csv \
  --lira reviewer_results/lira_reference_mia/lira_reference_mia_raw.csv \
  --reference-root reviewer_results/lira_reference_mia/reference_models \
  --out-dir reviewer_results/reviewer_artifacts/final_responses \
  --bootstrap 5000

python -u reviewer_tools/generate_reviewer_response_markdown.py \
  --artifact-dir reviewer_results/reviewer_artifacts \
  --out reviewer_results/reviewer_artifacts/REVIEWER_RESPONSE_TABLES.md

python -u reviewer_tools/generate_reviewer_1myw_followup.py
```

The final reproducibility bundle is under `reviewer_results/reviewer_artifacts/`:

```text
reviewer_results/reviewer_artifacts/
├── tables/                 # CSV and LaTeX tables T01–T09
├── figures/                # PNG and PDF figures F01–F07
├── final_responses/        # Consolidated reviewer and Area Chair responses
├── README.md               # Concern-to-evidence index and caveats
├── REVIEWER_RESPONSE_TABLES.md
└── manifest.json
```

The generated artifact index records the replication unit and interpretation limits for each analysis: target-model seeds for factorial/architecture results, data seeds for geometry, simulator seeds for finite-shot conditions, and attacker seeds for learned-MIA robustness.

---


## Attribution

QuRiFT builds on TorchQuantum by the MIT HAN Lab and contributors:

```text
https://github.com/mit-han-lab/torchquantum
```

TorchQuantum is distributed under the MIT License. Preserve upstream license and attribution notices when redistributing code derived from or bundled with TorchQuantum.

---

## Citation

If you use QuRiFT, please cite the accompanying paper:

```bibtex
@misc{qurift2026,
  title        = {Structural Privacy Vulnerabilities in Quantum Neural Networks},
  author       = {Anonymous Authors},
  year         = {2026},
  note         = {QuRiFT: Quantum Risk and Inference Fault-line Tracer}
}
```

We will update the BibTeX entry with the final author list, venue, and DOI once the paper is public.
