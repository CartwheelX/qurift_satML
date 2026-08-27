# PETS defense-extension protocol

This extension is isolated from the frozen SaTML study. Generated checkpoints,
results, and logs live only in `pets_runs/`, `pets_results/`, and `pets_logs/`.
The SaTML manifests and result trees are read-only discovery inputs.

## Confirmatory question

The primary question is whether a defense reduces membership leakage while
preserving useful prediction quality, and whether it reduces the structural
leakage contrast rather than merely shifting every configuration uniformly.
The predeclared Credit pair holds feature-map family and variational depth fixed
and changes EfficientSU2 repetitions from 1 (low) to 5 (high), at depth 6.
Five fresh paired blocks use data/model seeds absent from the discovery sweep.
`pets_b01` is a development pilot and is excluded from confirmatory estimates;
the independent confirmatory units are `pets_b02` through `pets_b05`.
The discovery-only values motivating this utility-matched pair were loss-MIA
AUC 0.512 versus 0.551, generalization gap 0.013 versus 0.068, and test
accuracy 0.780 versus 0.771 (repetitions 1 versus 5).

For defense (d), structural role (s\in\{L,H\}), and attack (a), the first
estimand is

\[
\Delta_{d,a}=\operatorname{AUC}_{d,a,H}-\operatorname{AUC}_{d,a,L}.
\]

The defense-by-structure interaction is

\[
\operatorname{DiD}_{d,a}=\Delta_{d,a}-\Delta_{\mathrm{None},a}.
\]

Negative DiD means the defense attenuates the high-minus-low structural leakage
contrast. The independent unit for uncertainty is the fresh paired target-model
block, never individual attack records.

Credit utility is not summarized by accuracy alone. Every training and output
condition reports accuracy, balanced accuracy, minimum and minority-class
recall, prediction-collapse status, predicted minority fraction, task ROC-AUC,
average precision, NLL, and mean entropy. Training and output-defense utility
are evaluated on all 2,000 held-out test records, never on a pool containing
members. ROC-AUC and average precision retain ranking information even when the
default argmax threshold predicts one class.

## Disjoint data roles

Every target receives three deterministic, disjoint partitions. With the
default 200 target-training and validation records:

- defense calibration: 50 target-training members and 50 validation
  non-members;
- adaptive-attack calibration: a different 50 target-training members and 50
  validation non-members;
- final evaluation: the remaining 100 target-training members and 100 test
  non-members.

Membership is balanced, and task-label counts are matched exactly within each
member/non-member pair. Under Credit's natural 78/22 class distribution, each
50-versus-50 calibration pool contains 39 class-0 and 11 class-1 records on
each side; the 100-versus-100 final pool contains 78 class-0 and 22 class-1
records on each side. The same paired ordering is retained in HSJ prefixes.
The target itself is still trained on the natural class distribution; task
balancing is applied only to membership comparisons.

The target test records are never used to fit a defense. For threshold and
learned attacks they are also never used to fit the attacker. Defended LiRA has
one explicit exception: its reference bank is defined on the standard balanced
target-train/target-test candidate pool, so disjoint subsets of that pool are
used for LiRA threshold calibration and final evaluation. No final-evaluation
record appears in its calibration subset. The exact sources, counts, record IDs,
and fingerprints are recorded per target. PETS analyses always use `1=member,
0=nonmember`; the older raw attack payload convention is not reused.

## Conditions

Training baselines are None, strong L2, HAMP-train, and DP-QML. HAMP-full is the
HAMP-trained checkpoint plus HAMP output transformation. Prediction-time
conditions on the ordinary target are DynaNoise, clean-room MemGuard,
LogitGuard-continuous, LogitGuard-quantized, MeasurementGuard-continuous,
LatticeRound, MemGQ-lattice, and MemGQ-lattice-sticky. Sticky behavior is a
query-hardening ablation, not the core novelty claim.

DP-QML follows the Watkins-style optimization core: RMSprop at learning rate
0.05, batch size 32, 30 epochs, no learning-rate scheduler, unweighted NLL,
and fixed clipping norm \(C=1\). It computes full per-example gradient vectors,
globally clips each vector, adds Gaussian noise to the sum, and divides by the
expected Poisson batch size. Sampling and Gaussian noise use independent random
streams. An empty Poisson draw still receives the accounted Gaussian update to
a zero clipped-gradient sum. The Opacus RDP accountant receives the exact
sampling probability and noise multiplier at every step. Missing Opacus, a
non-Poisson sampler, a missing accountant, or a non-finite epsilon prevents a
formal accounting claim.

Credit's natural 78/22 prevalence is retained. For every binary checkpoint, a
class-1 operating threshold is selected using validation balanced accuracy;
exact ties use validation accuracy and then proximity to 0.5. The rule is
frozen before the held-out test split or any membership-attack outcome is read.
Both default-0.5 and calibrated-threshold utility are recorded. The earlier
Adam/100-epoch DP checkpoints and clipping-norm probes are diagnostics and are
not eligible for confirmatory inference.

The final L2 penalty and DP epsilon are selected in a separate utility-only
tuning stage on `pets_b01`. The L2 grid is weight decay
\(\{10^{-3},10^{-4}\}\). The initial corrected DP grid was
\(\epsilon\in\{1,4,8\}\) at \(\delta=10^{-5}\), with \(C=1\) fixed. None passed
the frozen utility gate: the closest, \(\epsilon=4\), had worst-role ROC-AUC
0.595 and average precision 0.270. Following the predeclared fail-closed rule,
the utility-only grid was expanded with \(\epsilon\in\{16,32,64\}\); neither
attack outcomes nor confirmatory blocks were inspected. A setting is eligible
only when both structural roles have test ROC-AUC at least 0.65, average
precision at least 0.30, calibrated balanced accuracy at least 0.55,
calibrated minority recall at least 0.02, and non-collapsed calibrated
predictions. Among eligible settings, the selector chooses the largest L2
penalty and smallest epsilon over the combined grid. It fails closed if either
family has no eligible setting; attack outcomes are never loaded by the
selector. Opacus derives the noise multiplier from the frozen sample rate and
exact integer step count, and the final ledger must reproduce the selected
budget.
HAMP uses the artifact settings \(\gamma=0.95\) and \(\alpha=0.001\); its
soft-label true-class probability is solved from
\(H(\tilde y)=\gamma\log K\) for the target's class count.

## Attacks

The main attacks adapt to each defense using only the disjoint attack-calibration
partition: loss, confidence, maximum probability, entropy, margin, correctness,
a learned prediction-vector/statistics attacker, hard-label HSJ, and defended
reference-model LiRA. LiRA transforms both target and reference outputs and uses
checkpointed reference models. HAMP/DP-trained LiRA references are not silently
approximated by standard references; the current tool refuses those mismatches.
HSJ is also run directly on the L2-, HAMP-, and DP-trained checkpoints so that
training-induced boundary changes are measured. Because HAMP output replacement
preserves the predicted label, HAMP-full shares the HAMP-train HSJ result.
Paired HSJ comparisons use common random numbers per target, partition, and
record; the defense name is deliberately excluded from the attack seed.
The ordinary MemGQ-lattice HSJ value was empirically identical to the undefended
value in the development equivalence audit, with no relevant hard-label change
on the audited records. It is therefore not redundantly recomputed in the
confirmatory run. None, DynaNoise, the matched LatticeRound control, and sticky
MemGQ remain as the relevant output-boundary conditions.

For DynaNoise only, confidence threshold 0.9 and loss threshold 0.5 are also
reported as an artifact-faithful appendix. They are not the main adaptive
evidence. Nearby-query stress retrains an attacker on repeated bounded queries.
It uses identical perturbations across defenses and compares MemGQ/sticky
MemGQ with undefended, continuous/quantized LogitGuard, continuous
MeasurementGuard, and LatticeRound controls. Defended LiRA applies the same
output defense to the target and every matched reference model; standard and
L2 targets use separately trained references. HAMP/DP LiRA is not reported
until genuinely matched HAMP/DP reference training exists.

## Staging rule

Run `commands/pets_run_credit_pilot.sh` on `pets_b01` first. Record optimization
iterations, L2/HAMP/DP parameters, lattice shots, query radius, attack budgets,
and any justified change in a dated protocol note. Then freeze them. The full
launcher deliberately requires `QURIFT_PETS_PILOT_FROZEN=1`.
Use `pets_targets/PILOT_DECISION_TEMPLATE.md` for that record; decisions should
be based on feasibility, convergence, and constraint validity rather than
choosing the most favorable privacy outcome.
For the already completed pilot, `commands/pets_run_credit_corrections.sh`
moves all pre-label-matching result directories into a recoverable timestamped
archive, then reruns only partition-dependent defense and attack evaluation.
Trained targets and LiRA reference checkpoints are reused. The corrected pilot
adds matched LiRA/query controls and full-test imbalance-aware utility tables,
then performs utility-only L2/DP selection and writes the frozen confirmatory
manifest.
The pilot remains visible in `pets_results/pilot_analysis`, while
`commands/pets_analyze.sh` excludes it from final tables and figures.

## Reproduction order

```bash
python -m pip install -r requirements-pets.txt
bash commands/pets_prepare.sh
export QURIFT_PETS_STICKY_SECRET='replace-with-a-private-random-value'
bash commands/pets_run_credit_pilot.sh

# If the original pilot already completed, tune L2/DP from utility only and
# add missing controls without rerunning it:
bash commands/pets_run_credit_corrections.sh

# Only after accepting and recording the pilot settings:
export QURIFT_PETS_PILOT_FROZEN=1
nohup bash commands/pets_run_credit_full.sh \
  > pets_logs/full_launcher.log 2>&1 &

bash commands/pets_monitor.sh
```

Keep the same sticky secret for pilot and full blocks, do not commit it, and do
not print it in logs. Result metadata records only its SHA-256 digest.
