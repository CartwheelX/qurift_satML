# Frozen SaTML Experimental Protocol

## Research claim

The confirmatory claim is:

> Quantum data encoding is not privacy-neutral: feature-map family and
> repetition systematically alter post-encoding geometry and downstream
> membership leakage, and this structural information can support
> privacy-aware circuit selection.

The study tests an empirically supported, overfitting-mediated pathway. It does
not claim that encoder geometry directly reveals membership, that geometry is
independent of generalization, or that the reported associations are causal.

## Evidence layers

1. The completed 36-target MNIST QNN factorial is retained without rerunning or
   retrospectively changing its protocol.
2. A new Credit-default factorial is the large tabular cross-domain replication.
3. A Fashion-MNIST factorial tests transfer to a harder image domain, while a
   targeted WDBC study tests a sensitive biomedical tabular domain.
4. Post-encoder fidelity-kernel measurements test the proposed representation
   mechanism before the trainable circuit.
5. Threshold, learned/posterior, LiRA reference-model, and label-only attacks
   cover distinct adversary access assumptions.
6. Three finite-shot studies separate structural robustness, API query policy,
   and attack breadth under one frozen backend-derived calibration.
7. A structural privacy selector is developed on the factorial targets and
   evaluated once on entirely fresh split/initialization blocks.

## Dataset and leakage-safe preprocessing

The second domain is the UCI **Default of Credit Card Clients** dataset: 30,000
records, 23 attributes, binary default outcome, UCI dataset 350, DOI
`10.24432/C55S3H`. The repository fetcher pins OpenML data ID 42477, falls
back to the official UCI archive, and finally to a commit-pinned mirror if the
official providers are unavailable. It writes deterministic gzip and requires
canonical uncompressed-CSV SHA-256
`dfb1570f223efb65c0084027570369bdff6cc291b8238b9adce17ab60da4ca83`.

Each experimental block contains:

- 200 target-training records, which are MIA members;
- 200 validation records, used for training monitoring and selector utility;
- 2,000 target-test records, which are MIA nonmembers.

The partition is stratified and determined only by `split_seed`. Numeric
standardization, categorical one-hot encoding, PCA to six components, and the
final mapping to `[-1,1]` are fitted on the 200 target-training records only.
The fitted preprocessor, split hashes, PCA variance, source checksum, and range
diagnostics are saved beside every target. Evaluation values outside the
training-derived range are explicitly clipped by the fitted range transform.

Fashion-MNIST uses the original train and test partitions and classes 0, 1, 3,
and 8 (T-shirt/top, Trouser, Dress, and Bag), remapped to four labels. Each of
five independent blocks uses 200 balanced training members, 200 balanced
validation records, and 2,000 balanced test nonmembers. The train/validation
split and balanced test subset are determined by the block's `split_seed`.
Normalization uses fixed Fashion-MNIST constants `(0.2860, 0.3530)` and is not
estimated from evaluation data. The four canonical IDX source checksums, split
hashes, and class counts are validated and saved.

WDBC is UCI dataset 17 (DOI `10.24432/C5DW2B`), with 569 records, 30 numeric
features, and a binary diagnosis. Its deterministic snapshot has canonical
CSV SHA-256 `ec5134d1f4db4e0accdbb8705285cc335eabf53785c06d4f0e75126a84c7cefc`.
Each of five blocks partitions all records into 160 training members, 80
validation records, and 329 test nonmembers. Standardization, PCA to six
components, and mapping to `[-1,1]` are fitted on training members only.

## Paired factorial

The confirmatory Credit experiment uses eight independent blocks. A block is
one `(split_seed, init_seed)` pair. All 12 structural configurations in a block
share both seeds:

| Factor | Values |
| --- | --- |
| Feature map | Z, ZZ, EfficientSU2 |
| Encoder repetitions | 1, 5 |
| Variational depth | 2, 6 |

Width, optimizer, learning rate, epochs, batch size, encoder padding,
feature-map entanglement, variational entanglement, and measurement design are
fixed. This yields `8 × 12 = 96` target models.

The Fashion-MNIST replication uses the same 12 configurations over five paired
blocks (`60` targets). WDBC is deliberately targeted because of its smaller
sample size: it fixes variational depth at 2 and tests three feature maps by two
repetition levels over five paired blocks (`30` targets). Consequently, WDBC
supports repetition and feature-map contrasts, not a depth effect.

The primary endpoint is loss-threshold MIA AUC. The primary structural contrast
is repetitions `5 − 1`. Confirmatory secondary contrasts are depth `6 − 2`,
`Z − EfficientSU2`, and `ZZ − EfficientSU2`. `ZZ − Z` is reported as a
secondary feature-map comparison.

For every outcome, contrasts are calculated inside each block while averaging
over the other factorial dimensions. The eight block effects are the
independent observations. Reported uncertainty is a percentile bootstrap over
those block effects. Exact paired sign-flip tests are Holm-adjusted across the
five prespecified contrasts within each outcome/attack family. A block-fixed
additive regression with CR1 standard errors clustered by block is secondary
and descriptive. Both accuracy gap and `test_loss - train_loss` are reported.

For feature-map and repetition contrasts at a fixed variational depth, the
trainable parameter count is held constant. A fail-closed resource check
verifies that repetition changes fixed-encoder gate count while leaving
trainable capacity unchanged. Depth is analyzed as a separate, intentionally
capacity-changing factor.

## Attack endpoints and access models

Separate scalar threshold attacks use loss, confidence/maximum probability,
entropy, margin, and correctness. Threshold calibration is cross-fitted. The
analysis reports AUC, balanced accuracy, membership advantage, TPR at requested
FPRs of 1%, 5%, and 10%, and the actually attained empirical FPR.

TPR@1% FPR is confirmatory for the loss attack. Every target has at least 2,000
nonmembers, giving an empirical FPR resolution of at most `1/2000 = 0.0005`.
Record-level bootstrap confidence intervals accompany both AUC and fixed-FPR
TPR.

Fashion-MNIST retains 1%, 5%, and 10% FPR endpoints. For WDBC, 5% and 10% are
primary; 1% is displayed only as exploratory because 329 nonmembers leave very
few false-positive observations at that operating point.

Online/offline LiRA uses a balanced candidate subset: all target-training
members and an equal-size deterministic subset of nonmembers (400 records for
Credit/Fashion-MNIST and 320 for WDBC). Sixteen references
are trained per structural-configuration × split-block candidate population,
with each candidate included in exactly half. Reference banks are never shared
across different data splits. Credit uses full LiRA coverage. For the added
datasets, learned, correctness-only label-output, and hard-label boundary
attacks cover every target, while LiRA is a prespecified representative
analysis of all six depth-2 structural configurations in the first three
paired blocks (18 targets per dataset). This is a reference-model robustness
attack, not the source of the 1% FPR claim.

The corrected label-only boundary attack is an independent hard-label
HopSkipJump-style implementation. It uses a balanced deterministic subset of
members and nonmembers, assigns distance zero to initially misclassified
records, and gives every initially correct record the same nominal search
procedure and maximum query budget. Credit and WDBC probes are clipped to the
predeclared `[-1,1]` PCA domain; image probes are clipped to the valid
normalized-pixel domain. If no changed-label initialization is found, the
operational score is capped at the record-independent L2 diameter of the
declared input box and marked search-censored rather than removed or stored as
NaN. Query counts, prediction histograms, initialization success, and censored
fractions are reported. The earlier validation-anchor chord
results are diagnostic only and are excluded from SaTML inference.

## Direct geometry

The pure-state fidelity kernel is measured immediately after the fixed encoder
and before the variational circuit. For Z, ZZ, and EfficientSU2 at repetitions
1 and 5, the study reports:

- within-class and between-class similarity;
- their difference;
- centered kernel-label alignment;
- effective rank and kernel spectrum summaries;
- train-test MMD²;
- encoder-operation and state-signature integrity checks.

Geometry is evaluated over the same eight Credit split seeds. Repetition
operation counts and state signatures must differ in the expected direction
before repetition results are accepted.

The pathway analysis connects configuration-level geometry, accuracy/loss
gaps, and loss-MIA AUC using independent block/geometry-seed resampling. Its
block-clustered regressions are explanatory associations only; they are not
presented as causal mediation estimates.

## Encoding-scale robustness

Angle scale is targeted robustness evidence rather than another full
factorial. At depth 2, all three feature maps and both repetition levels are
evaluated at `alpha ∈ {0.5, 1, 2}` over five paired blocks. The `alpha=1`
targets are reused from the main factorial; only `0.5` and `2` are additionally
trained. The preprocessing is unchanged, isolating the intervention
`theta = alpha × f(x)`.

## Fresh privacy-selector evaluation

Development uses only the 96 factorial targets. Three policies are frozen:

1. `utility_only`: highest mean development validation accuracy;
2. `privacy_aware`: lowest mean loss-MIA AUC among configurations whose mean
   validation accuracy is within 0.02 of the utility-only maximum;
3. `utility_regularized`: the utility-only configuration trained with
   prespecified Adam weight decay `0.001`.

Ties are resolved deterministically by privacy, utility, and structural ID as
recorded by the selector script. After the decision JSON is written, all three
policies are trained on five new split/initialization blocks: 15 targets. Their
utility, gap, and leakage differences are evaluated as fresh paired contrasts.
The fresh seeds do not overlap development seeds.

## Frozen noise, query policy, and noisy attacks

All three noise studies use local Aer simulation. IBM access is used once to
capture a credential-free reconstruction snapshot containing the complete Aer
noise-model dictionary, backend configuration/properties, calibration
timestamp, and SHA-256 manifest. Subsequent target evaluations load that same
snapshot from disk and make no IBM request. The results are therefore described
as backend-calibration-derived simulation, not hardware execution. A result
directory is fail-closed to one snapshot hash.

The externally visible repeated-query semantics are fixed before evaluation.
For each API query, shot counts are converted to expectation values and passed
through the trained classical head independently. The returned attack feature
is the arithmetic mean of the resulting class-probability vectors. Pooling all
counts before the nonlinear head is retained only as a named diagnostic and is
never substituted for the primary API aggregation.

### N1: structural robustness

N1 evaluates the full retained MNIST QNN `3 × 2 × 2 × 3` design: three
feature-map families, repetitions 1 and 5, depths 2 and 6, and three trained
model seeds (`36` checkpoints). The same 200 members and 200 nonmembers are
evaluated by exact inference, ideal Aer at one query of 512 shots, and frozen
backend-derived noisy Aer at one query of 512 shots. Finite-shot conditions use
ten simulator seeds. Scalar threshold attacks and a fixed, five-fold
cross-fitted learned prediction-vector-plus-statistics attack are reported.

Repetition effects are calculated separately at depth 2 and depth 6; depth
effects are calculated separately at repetition 1 and repetition 5. The
interaction is

`(AUC[r5,d6] - AUC[r1,d6]) - (AUC[r5,d2] - AUC[r1,d2])`.

The analysis also reports repetition and depth main effects and the paired
feature-map contrasts `Z - EfficientSU2`, `ZZ - EfficientSU2`, and `ZZ - Z`,
averaging over repetition and depth inside each trained model-seed block.

Simulator seeds are averaged inside a trained checkpoint before paired
structural inference. The three trained model seeds, not simulator seeds, are
the inferential units. Noise moderation is the noisy structural effect minus
the matching exact structural effect. Rank correlation with exact structural
means is explicitly descriptive.

### N2: API query policy

N2 is a targeted policy experiment, not another factorial. It fixes depth 6
and model seed 43 and evaluates all three feature maps at repetitions 1 and 5
(`6` checkpoints). Six policies distinguish added shots from repeated API
calls: `1×128`, `1×512`, `1×2560`, `5×128`, `5×512`, and `20×128`.
Ideal and frozen-noisy Aer each use ten simulator seeds. Prespecified contrasts
include single-query shot increases, repeated-query increases at fixed shots
per query, and the equal-total-shot comparisons `5×512` versus `1×2560` and
`20×128` versus `1×2560`. In addition to the mean returned probability vector,
the learned attacker may use across-query probability standard deviations.
Because N2 contains one trained seed per structural cell, its uncertainty is a
bootstrap over the six targeted checkpoints and is labeled policy robustness,
not confirmatory structural inference.

### N3: attack breadth under noise

N3 selects two preregistered structural endpoints, EfficientSU2 repetition 1
depth 6 and ZZ repetition 5 depth 6, across the three model seeds. Each target
uses 16 matched reference models. The target and every reference are evaluated
under the same exact, ideal-shot, or frozen-noisy oracle, and LiRA reports the
paired high-minus-low endpoint contrast after averaging five simulator seeds
inside each checkpoint. Reference checkpoints and their candidate fingerprints
are saved and checked; exact scores cannot be silently reused as noisy
reference outputs.

Noisy label-only boundary scoring is an optional, explicitly query-accounted
pilot on two endpoints. Each label is one independent shot-based API query;
there is no hidden probability access or majority vote in the attack output.
Its small selected design is not used as factorial evidence.

## Exclusions and interpretation

- A failed IBM noise-model load is never relabeled as ideal noise.
- Incomplete paired blocks are not used in confirmatory contrasts.
- The Credit snapshot checksum, preprocessing provenance, member convention,
  target count, and low-FPR resolution must pass the fail-closed validator.
- Fashion-MNIST and WDBC split provenance, class balance, target counts, and
  dataset-specific FPR claims must pass their fail-closed validator.
- Results from the old NeurIPS folders and new `satml_*` folders are never
  automatically pooled.
- Scaling, N2 query-policy, and N3 selected-attack analyses remain targeted
  robustness checks; only N1 contains the complete retained noise factorial.
- Real-device execution and broader architectures remain future extensions;
  the cross-domain evidence now spans MNIST, Fashion-MNIST, Credit-default,
  WDBC, and the retained synthetic tasks.
