# Defense and attack provenance

No unlicensed third-party source has been copied into this repository.

- **DynaNoise.** The native PyTorch adapter follows the algorithm and defaults
  in `Javad-Forough/DynaNoise-PoPETs2026-Artifact`, pinned to revision
  `27c6ba5664eb3ba28973d44e2ea4830d15fd3ee5` (MIT): normalized confidence
  (R=1-H(p)/\log K), variance
  (\sigma^2=\sigma_0^2(1+\lambda R)), Gaussian logit noise, temperature
  scaling, and optional Monte Carlo averaging. The upstream MIT notice is in
  `third_party/DYNANOISE_NOTICE.txt`.

- **HAMP.** The training/output split is a native adaptation informed by the
  HAMP paper and the MIT-licensed DynaNoise artifact: high-entropy soft targets
  plus an entropy reward during training, and rank-preserving reassignment from
  predictions on random support-constrained inputs at inference. Credit/WDBC
  random inputs are generated only from the defense-calibration partition and
  are checked against `[-1,1]^6`.

- **MemGuard.** `jinyuan-jia/MemGuard` was inspected at revision
  `34e1859e37c133b6517fd01b834e1c091012b197`. That legacy Python-2/Keras
  repository declares no license. The implementation here is therefore a
  clean-room PyTorch implementation of the paper objective; no repository code
  or trained defense network is copied. Results must be called
  “clean-room MemGuard,” not a bitwise reproduction.

- **LiRA.** The QuRiFT reference-model implementation is independent and records
  the inspected repository revisions already listed in
  `reviewer_tools/qurift_lira_attack.py`. Defended LiRA requires saved reference
  checkpoints and applies the same output defense to target and references.
  The paper's Eq. (4) defines offline LiRA as a one-sided OUT test, implemented
  here as a stable Gaussian log-CDF under `lira_offline`. The explicit
  `lira_offline_one_sided_z` names remain rank-equivalent compatibility aliases
  and are not counted as additional inferential tests. The authors' released
  TensorFlow Privacy code uses negative OUT log-density instead; that behavior
  is retained under the explicit auxiliary name
  `lira_offline_density_surprise`, rather than conflating the two definitions.
  Its disjoint calibration/evaluation candidate subsets are exactly
  task-label matched across target members and nonmembers.

- **Label-only MIA.** The HSJ implementation is the repository's independent
  PyTorch/TorchQuantum port of the decision-boundary protocol from
  Choquette-Choo et al. It assigns zero distance to initially misclassified
  inputs and records initialization failure, censoring, convergence, and actual
  queries.

- **Evaluation populations.** Credit target models retain the dataset's natural
  78/22 task-class distribution. MIA pools are balanced by membership and
  exactly task-label matched across membership. Output-defense utility is
  evaluated on all held-out test records and contains no target-training
  members. Pre-correction pilot outputs are retained under
  `pets_results/archive/pre_label_matched/` and excluded from analysis.

- **DP accounting.** Formal epsilon reporting uses Opacus 1.5.4 (Apache-2.0),
  selected because Opacus 1.6 requires PyTorch 2.6 while the reproducibility
  environment contains PyTorch 2.5.1. The QNN per-example gradients are computed
  directly through autograd so unsupported custom quantum modules cannot be
  silently skipped by gradient hooks. The corrected defense uses the
  Watkins-style RMSprop/0.05/batch-32/30-epoch optimization core, fixed
  per-example clipping norm 1, independent sampling and noise streams, and a
  Gaussian update on accounted empty Poisson steps. Binary deployment
  thresholds are selected from validation balanced accuracy and stored with
  each checkpoint; no test or MIA outcome enters threshold selection.
