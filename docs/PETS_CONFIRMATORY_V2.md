# PETS Credit confirmatory protocol v2

This is the publication-facing defense evaluation. It is an isolated,
prospective extension of the inspected four-block pilot: it writes only to
`pets_v2_targets/`, `pets_v2_runs/`, `pets_v2_results/`, and `pets_v2_logs/`.
The earlier `pets_*` artifacts remain development evidence and are never pooled
with v2.

## Contribution and conditions

The study is an evaluation/systematization contribution; it does not depend on
claiming a novel defense. The five literature-derived defense conditions are:

1. strong L2 regularization;
2. DP-QML using the Watkins-style RMSprop/0.05/batch-32/30-epoch schedule;
3. HAMP training plus HAMP output transformation (`hamp_full`);
4. clean-room MemGuard; and
5. DynaNoise.

The undefended condition is the common baseline. LogitGuard (continuous and
quantized), MeasurementGuard, LatticeRound, and MemGQ (lattice and sticky)
remain mechanistic controls. MemGQ is an instrument for testing whether the
QNN measurement space and finite-shot lattice are useful defense domains, not
a claimed new PET.

Three structural roles are evaluated in each paired target-model block:

- `low`: EfficientSU2, repetition 1, depth 6;
- `repetition`: EfficientSU2, repetition 5, depth 6; and
- `stress`: ZZ, repetition 5, depth 6.

The repetition-minus-low contrast changes repetition while holding feature-map
family and depth fixed. The stress role was selected from the completed Credit
discovery study, before these fresh v2 runs: over eight paired blocks,
fixed-variance online LiRA had mean AUC 0.6044 for ZZ-r5-d6, 0.5649 for
EfficientSU2-r5-d6, and 0.5251 for EfficientSU2-r1-d6. The paired ZZ-minus-
EfficientSU2-r5-d6 difference was +0.0395 in all eight blocks (exact two-sided
sign-flip p=0.0078), while ZZ-minus-low was +0.0793. These are role-selection
statistics, not confirmatory results. Because the ZZ comparisons change the
feature-map family and may also differ in utility, they are defense stress tests,
not clean one-factor causal estimates; repetition-minus-low remains the controlled
structural contrast.
Eight fresh blocks use data seeds 90261--90268 and model seeds 100261--100268;
none appears in the discovery, pilot, or earlier confirmatory manifests.

## Frozen primary and secondary analyses

The primary endpoint is the paired block-level difference

`AUC(defense, stress) - AUC(none, stress)`

for fixed-variance online LiRA. Holm correction is applied over the five
literature defenses. Secondary results include all scalar, learned,
hard-label, nearby-query, and LiRA attacks; the repetition-minus-low,
stress-minus-low, and stress-minus-repetition structural contrasts; and the
corresponding difference-in-differences. Uncertainty is computed over the eight
paired target-model blocks, never by treating attack records as independent
model replications. The analysis exports exact two-sided paired sign-flip
p-values, paired-block bootstrap intervals, and multiplicity-adjusted values.

The paper-defined offline LiRA is the one-sided OUT test and is implemented as
the numerically stable Gaussian log-CDF under `lira_offline` (and its fixed-
variance form). The explicitly named `lira_offline_one_sided_z` scores are kept
as rank-equivalent compatibility aliases: they have identical ROC curves and
are excluded from multiplicity counting. The authors' released TensorFlow
Privacy artifact instead uses negative OUT log-density; that implementation is
also retained, under the unambiguous auxiliary names
`lira_offline_density_surprise` and its fixed-variance form. Thus no strong
one-sided result is removed and the paper/code discrepancy is visible.

Every binary checkpoint freezes its class-1 threshold from validation balanced
accuracy before test or MIA outcomes are read. The same rule is now used by
ordinary, L2, HAMP, and DP targets; every output sanitizer; correctness and
learned attacks; HSJ; nearby-query stress; and target/reference LiRA oracles.
Label-preserving defenses preserve this deployed decision, not merely argmax.

Scalar/output-defense evaluation keeps the 100 final members and widens only
the held-out-test nonmember side to 1,000. It can therefore resolve a 0.1% FPR
increment. Defended LiRA retains its task-label-matched 100/100 final pool, so
1% is its finest nonzero empirical FPR step; 0.1% is reported only as an
unresolved zero-FP operating point. Every output includes the attained FPR,
pool size, and resolution flag.

LiRA uses 16 clean references for every structural-role, block, and training
arm combination. HAMP references use the HAMP objective; DP references use the
same DP mechanism and epsilon and recalibrate the noise multiplier for their
own sampling rate and exact step count. Within a structural-role/block tuple,
all training arms share the same reference inclusion matrix and initialization
seeds as common random numbers. Bank identities still include the training
mechanism, preventing ordinary/HAMP/DP path collisions.
HSJ likewise shares its randomized probe sequence across the four training
arms within each structural-role/block tuple and across output defenses.

## Run order

From the repository root:

```bash
export PYTHON_BIN=/home/najeeb/.conda/envs/tq39_vv2/bin/python
export QURIFT_GPUS=0,1,2,3,4,5,6,7
export QURIFT_JOBS_PER_GPU=auto
export QURIFT_PETS_STICKY_SECRET='<the same private value used for the pilot>'

bash commands/pets_run_credit_confirmatory_v2.sh 1  # build + validate manifests
bash commands/pets_run_credit_confirmatory_v2.sh 2  # train 96 target conditions
bash commands/pets_run_credit_confirmatory_v2.sh 3  # checkpoint/manifest gate
bash commands/pets_run_credit_confirmatory_v2.sh 4  # evaluation, HSJ, query stress
bash commands/pets_run_credit_confirmatory_v2.sh 5  # 16-reference matched LiRA banks
bash commands/pets_run_credit_confirmatory_v2.sh 6  # defended LiRA scoring
bash commands/pets_run_credit_confirmatory_v2.sh 7  # final gate, tables, figures
```

`commands/pets_finalize.sh N` is a compatibility alias for the same numbered
stage. Each expensive stage is resumable. Stage 5 creates 1,536 reference
models and will dominate runtime; do not reduce the frozen reference count for
headline results.

The launcher takes a nonblocking `flock` under `pets_v2_logs/` for every stage
it can run. A second invocation of the same numbered stage exits immediately
with status 75. An `all` invocation holds all seven stage locks, so it cannot
overlap a numbered invocation. The kernel releases locks when the launcher
exits, including after a forced termination, and the stage can then be resumed
normally. The adjacent `.owner` file records the holder PID and start time for
diagnosis. Literal documentation examples such as
`YOUR_SAME_FROZEN_PILOT_SECRET` are rejected as sticky-secret placeholders.

For unattended execution:

```bash
nohup bash commands/pets_run_credit_confirmatory_v2.sh all \
  >> pets_v2_logs/confirmatory_launcher.log 2>&1 &
echo $!
```

Use append redirection (`>>`) for unattended launcher logs. Redirection is
performed by the calling shell before the launcher can test its lock, so `>`
could truncate an existing log even though duplicate computation is rejected.

## Pre-analysis evaluation refresh

If a correctness-affecting implementation issue is found after target and
reference training but before confirmatory analysis, do not delete or overwrite
the affected results in place. The refresh tool preserves `pets_v2_runs/`, all
1,536 LiRA score/checkpoint references, and
`pets_v2_results/defenses/training_status.csv`. It archives all 96 target result
directories and every non-training status CSV using same-filesystem atomic
renames. A write-ahead manifest records the reason, source/destination paths,
content hashes, progress, and protected-tree hashes; an ordinary failure is
rolled back, while the journal identifies the exact last operation after a
machine-level interruption. Stage-4/6 worker logs can be included explicitly;
training and reference logs are never selected.

First inspect the plan. This is read-only apart from taking the same temporary
stage locks used by the launcher:

```bash
bash commands/pets_refresh_confirmatory_v2_evaluation.sh plan \
  --archive-worker-logs
```

Only after reviewing the complete inventory, execute the recoverable archive.
This also creates a fresh 256-bit sticky secret in an untracked, mode-600 file;
the value is never printed:

```bash
bash commands/pets_refresh_confirmatory_v2_evaluation.sh archive \
  --archive-worker-logs
```

Then rerun corrected Stage 4. The wrapper reads the secret without placing it
in shell history and the ordinary stage lock prevents duplicate execution:

```bash
nohup bash commands/pets_refresh_confirmatory_v2_evaluation.sh stage4 \
  >> pets_v2_logs/stage4_evaluation_refresh.log 2>&1 &
echo $!
```

After Stage 4 completes, use the same secret for corrected Stage 6:

```bash
nohup bash commands/pets_refresh_confirmatory_v2_evaluation.sh stage6 \
  >> pets_v2_logs/stage6_lira_scoring_refresh.log 2>&1 &
echo $!
```

The archiver refuses to proceed if a v2 launcher/worker is active, if any of
the 96 result directories or required Stage-4 status files is missing, if the
96 target checkpoints or 1,536 reference pairs are incomplete, or if an atomic
same-filesystem rename cannot be guaranteed. The archive action is intentionally
not invoked by either rerun action.

Monitor without changing the run:

```bash
tail -f pets_v2_logs/confirmatory_launcher.log
watch -n 5 nvidia-smi
find pets_v2_results/defenses -name '*_metrics.json' | wc -l
find pets_v2_results/lira_references/reference_models -name 'reference_*.npz' | wc -l
```

The final integrity gate requires all 96 target evaluations, the complete
predeclared HSJ/query/LiRA condition matrix, protocol-v4 decision-rule
provenance, and exactly 16 score/checkpoint references in every bank. Analysis
does not run when any item is missing or stale.

## Outputs

Publication-facing outputs are written to `pets_v2_results/analysis/`:

- `primary_defense_efficacy.csv` and its PDF/PNG forest plot;
- paired secondary defense, structural, and difference-in-differences tables;
- privacy and full-test utility summaries;
- `lira_score_family_summary.csv`, including the paper score, z aliases, and
  released-artifact density comparators;
- `lira_alias_equivalence.csv`, which verifies that the z aliases reproduce the
  paper score's reported ranking metrics exactly; and
- `low_fpr_resolution.csv`, which states which operating points are empirically
  resolvable for every attack family.
