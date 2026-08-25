# SaTML hard-label boundary MIA

## Why the earlier Credit result is diagnostic only

The first Credit label-only implementation searched along chords from each
candidate to held-out validation records receiving a different predicted
class. Seven of 96 targets predicted class 0 for every validation record, so
their scores were undefined. The remaining targets received between one and
16 useful directions depending on validation prediction diversity. Because
directional coverage was itself associated with encoder repetition, the 89
successful outputs cannot be used for structural inference.

Those files remain under `satml_results/credit_factorial/label_only/` for
auditability. They are not read by the corrected launchers or paired analyses.

## Corrected threat model

The corrected attack follows the boundary-distance construction of
Choquette-Choo et al., *Label-Only Membership Inference Attacks* (ICML 2021),
whose released implementation invokes HopSkipJump for initially correct
records and assigns distance zero to initially misclassified records. The
reference repository revision studied is
`cchoquette/membership-inference@ce12e12139b61b8d042ec38bba2eeac56b55b357`.

QuRiFT implements the procedure independently for PyTorch/TorchQuantum. The
attacker consumes only the returned class label and knows the candidate's true
label. It never consumes a probability, logit, loss, gradient, quantum state,
or model parameter. For each initially correct candidate it:

1. searches for an untargeted adversarial initialization using bounded random
   probes;
2. projects that point toward the candidate using hard-label binary search;
3. estimates a boundary normal from symmetric hard-label probes;
4. takes a geometric decision-based step and reprojects; and
5. repeats until the iteration limit or maximum query budget is reached.

The implementation is HSJ-style rather than a claim of bitwise equivalence to
the historical TensorFlow/CleverHans code, and the returned distance is not a
certified global minimum.

## Uniform opportunity and failure handling

Every eligible record receives the same nominal settings: maximum queries,
initialization cap, initialization distribution, gradient sample count,
binary-search depth, iteration count, stopping rules, and input bounds. Early
termination and different realized query counts are permitted outcomes of the
same algorithm. The initial label query is included in the per-record maximum.

If no changed-label initialization is observed within the prescribed search,
the record is retained with one predeclared, record-independent capped
operational score equal to the L2 diameter of the declared input box and is
marked `search_censored=True`. This means only that the declared search did not
find a boundary; it does not assert that the true global boundary distance
equals the cap or is infinite.

Credit and WDBC use their train-fitted PCA/MinMax domain `[-1,1]^6`.
Fashion-MNIST and MNIST use the normalized image bounds corresponding to raw
pixels in `[0,1]`. Common random numbers are used within a data seed and sample
identifier so matched structural configurations receive the same random probe
stream.

For the tabular studies, this threat model audits the deployed six-component
continuous model-input interface. It is not presented as a certificate that
every perturbed PCA vector corresponds to a semantically valid raw Credit or
WDBC record.

## Prespecified outputs and interpretation

The primary performance summaries are ROC-AUC and TPR at declared FPRs. The
cross-fitted threshold fields are descriptive evaluation summaries and are not
described as shadow-calibrated operational thresholds. Every target also
records:

- train/test candidate and validation prediction histograms;
- initialization success, stopping-reason, and search-completion diagnostics;
- search-censored and query-budget-exhausted fractions;
- member and nonmember censoring fractions;
- mean, median, and maximum realized label queries;
- declared input bounds and their provenance; and
- the exact protocol version and seed rule.

Correctness-only label MIA is reported separately. It is always defined but is
not presented as a boundary-distance attack.

## Query-budget pilot

Before the 96-target run, execute the prespecified three-target pilot. It uses
the same 20 members and 20 nonmembers at all three budgets and covers one
validation-prediction-collapsed EfficientSU2 target, one nearly collapsed Z
target, and one healthy repetition-5 Z target:

```bash
export QURIFT_GPUS=0,1,2,3,4,5,6,7
export QURIFT_LABEL_JOBS_PER_GPU=1

bash commands/satml_pilot_credit_label_only_hsj.sh
```

The 128-, 512-, and 2,500-query regimes write to separate `q*` directories.
The analysis checks that candidate identities are identical and compares
initialization/censoring, realized queries, and per-record score convergence.
It must not be used to choose the final budget by whichever target AUC is
largest. The declared primary full-run budget remains 512 queries; 2,500 is a
high-budget sensitivity check motivated by the original paper's approximate
query regime.

Pilot summaries are written under:

```text
satml_results/credit_factorial/label_only_hsj_pilot/analysis/
```

To validate all pilot commands without starting any target process:

```bash
QURIFT_LABEL_PILOT_DRY_RUN=1 \
  bash commands/satml_pilot_credit_label_only_hsj.sh
```

## Full Credit run

```bash
export QURIFT_GPUS=0,1,2,3,4,5,6,7
export QURIFT_LABEL_JOBS_PER_GPU=1
export QURIFT_LABEL_MAX_QUERIES=512
export QURIFT_LABEL_INIT_QUERIES=128

bash commands/satml_run_credit_label_only_hsj.sh
```

Monitor the corrected stage with:

```bash
watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/credit_factorial/label_only_hsj/target_scoring_status.csv \
  --expected 96
```

The launcher is resumable because it writes to a fresh directory. It cannot
mistake the earlier chord scores for completed HSJ-style scores. Once this
stage finishes, run `bash commands/satml_analyze_credit_all_attacks.sh` to
regenerate the final paired table directly from the completed threshold,
learned, LiRA, correctness-only, and corrected HSJ files.
