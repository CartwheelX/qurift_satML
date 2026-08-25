# SaTML Runbook

Run every command from the repository root in the `tq39_vv2` environment.

## 1. Install and prepare

```bash
pip install -r requirements.txt -r requirements-satml.txt
bash commands/satml_prepare.sh
```

If the dataset provider temporarily resets the connection, rerun the command.
The fetcher reuses `data/openml_cache/uci_credit_default_350.zip` if present.
The preparation stage pins both tabular dataset checksums; creates the Credit,
Fashion-MNIST, and WDBC manifests; and validates every paired design.

## 2. Confirmatory Credit factorial

```bash
export QURIFT_GPUS=0,1,2,3,4,5,6,7
export QURIFT_JOBS_PER_GPU=1
bash commands/satml_run_credit_factorial.sh
```

Monitor it from another terminal:

```bash
watch -n 15 python satml_tools/progress.py \
  --targets satml_targets/credit_factorial_targets.csv \
  --run-root satml_runs
```

Inspect an individual live/partial target with:

```bash
tail -f satml_runs/satml_credit_factorial/CREDIT_QNN_z_r1_d2_b01/train.log
```

The launcher is resumable. Re-running the same command with `--resume` skips
complete model-plus-attack exports.

### Unattended continuation after Credit

After the Credit progress command reports `96/96`, the remaining required
stages can be run sequentially by one fail-closed wrapper. It checks the Credit
artifacts, imports all 36 retained MNIST checkpoints, captures or validates the
frozen IBM calibration before starting long work, preserves that snapshot
across resumed invocations, and then executes Sections 3--12 in dependency
order. A stage receives a completion marker only after a zero exit status.

First confirm the planned stages without launching work:

```bash
QURIFT_GPUS=0,1,2,3,4,5,6,7 \
PYTHON_BIN="$(command -v python)" \
bash commands/satml_run_all_remaining.sh --dry-run
```

If no frozen snapshot already exists, provide a current IBM backend and either
a working saved account or environment credentials. Launch under `nohup` so a
terminal disconnect does not terminate the pipeline:

```bash
export PYTHON_BIN="$(command -v python)"
export QURIFT_GPUS=0,1,2,3,4,5,6,7
export QURIFT_JOBS_PER_GPU=1
export QURIFT_NOISE_JOBS_PER_GPU=1
export QURIFT_LABEL_JOBS_PER_GPU=1
export QURIFT_LABEL_MAX_QUERIES=512
export QURIFT_LABEL_INIT_QUERIES=128
export QURIFT_LEGACY_REPO='/home/najeeb/quarift_neurips_rebutal_2'
export QURIFT_NOISE_BACKEND='ibm_kingston'
export QISKIT_IBM_TOKEN='YOUR_VALID_TOKEN'
export QISKIT_IBM_INSTANCE='YOUR_INSTANCE_CRN'

nohup bash commands/satml_run_all_remaining.sh \
  > satml_logs/satml_all_launcher.log 2>&1 &
echo $!
```

Alternatively, set `QURIFT_NOISE_SNAPSHOT` to an already verified snapshot and
omit IBM credentials. The wrapper unsets credentials immediately after the
snapshot preflight. The optional noisy label-only pilot is excluded by default;
set `QURIFT_INCLUDE_OPTIONAL_NOISY_LABEL=1` before launch to include it.

Monitor the detached run with:

```bash
cat satml_results/unattended_pipeline/current_stage.txt
tail -f satml_logs/satml_all_remaining_latest.log
cat "$(cat satml_results/unattended_pipeline/latest_status_path.txt)" | column -ts $'\t'
```

If a stage fails, correct the reported cause and launch the same wrapper again.
Completed stages are skipped using durable markers, incomplete training
launchers retain their normal `--resume` behavior, and the originally frozen
snapshot is reused. Do not set `QURIFT_MASTER_FORCE=1` unless intentionally
rerunning completed stages; in particular, do not regenerate a selector after
fresh outcomes have been inspected.

## 3. Primary metrics, threshold attacks, and paired inference

```bash
bash commands/satml_analyze_credit_factorial.sh
```

This stage extracts utility/generalization metrics, runs all scalar threshold
signals with 10,000 record bootstraps, calculates TPR at 1%, 5%, and 10% FPR,
verifies that repetition changes encoder gate count without changing trainable
parameter count, runs paired block inference, and performs fail-closed protocol
validation.

## 4. Direct Credit geometry

```bash
bash commands/satml_run_credit_geometry.sh
```

Do not interpret repetition geometry unless
`satml_results/credit_geometry/repetition_integrity.csv` passes.

After the geometry and threshold results both exist, quantify the proposed
pathway without treating it as proof of causal mediation:

```bash
bash commands/satml_analyze_mechanism.sh
```

The script resamples independent target blocks and geometry seeds, reports
configuration-level associations, and fits secondary block-clustered
explanatory regressions with accuracy and loss gaps.

## 5. Fashion-MNIST and WDBC replications

Run the 60-target Fashion-MNIST factorial and 30-target fixed-depth WDBC study:

```bash
bash commands/satml_run_fashion_factorial.sh
bash commands/satml_run_wdbc_targeted.sh
```

Monitor either target table:

```bash
watch -n 15 python satml_tools/progress.py \
  --targets satml_targets/fashion_factorial_targets.csv --run-root satml_runs

watch -n 15 python satml_tools/progress.py \
  --targets satml_targets/wdbc_targeted_targets.csv --run-root satml_runs
```

Then extract metrics, run threshold attacks, calculate paired contrasts, audit
resources, and validate provenance:

```bash
bash commands/satml_analyze_fashion.sh
bash commands/satml_analyze_wdbc.sh
```

Fashion-MNIST treats TPR at 1%, 5%, and 10% FPR as planned endpoints. WDBC
treats 5% and 10% as primary; its 1% output is exploratory because only 329
nonmembers are available.

## 6. Added-domain geometry and pathway analysis

```bash
bash commands/satml_run_added_geometry.sh
bash commands/satml_analyze_added_mechanisms.sh
```

Check each `repetition_integrity.csv` before interpreting geometry. These
analyses are dataset-specific and are not automatically pooled.

## 7. LiRA and label-only robustness attacks

First run the prespecified hard-label query-budget pilot:

```bash
bash commands/satml_pilot_credit_label_only_hsj.sh
```

It compares 128, 512, and 2,500 queries on fixed candidates from three
predeclared Credit targets. Use it to audit runtime, censoring, initialization,
and score convergence—not to select the final budget by attack AUC. See
`docs/SATML_LABEL_ONLY_HSJ.md` for the target roles and outputs.

```bash
bash commands/satml_run_credit_attacks.sh
```

This is intentionally separate because it is substantially more expensive. It
trains the cross-validated learned prediction-vector/statistics attacker and
LiRA, computes the correctness-only label baseline, and runs the hard-label
HopSkipJump-style boundary attack. LiRA trains 16 references for each of the
96 structural-configuration × split-block candidate populations (1,536
references total), then scores its matching target. Banks cannot be reused
across blocks because their candidate records differ. The final command
regenerates paired structural contrasts across all attack families.

The HSJ-style attack uses 200 members and a deterministic 200-nonmember subset
per Credit/Fashion target; WDBC uses 160+160. Its default maximum is 512 label
queries per record, including the initial prediction. The same initialization,
gradient-estimation, projection, clipping, and censoring rules are applied to
every structural configuration. The old outputs under
`satml_results/credit_factorial/label_only/` are the invalidated validation-
anchor diagnostic and are never resumed or combined. Corrected outputs are
written under `label_only_hsj/`; see `docs/SATML_LABEL_ONLY_HSJ.md`.

If the learned and LiRA stages are already complete, resume specifically from
the Credit label-only stage with:

```bash
bash commands/satml_run_credit_label_only_hsj.sh
```

Then run `bash commands/satml_analyze_credit_all_attacks.sh` to regenerate the
all-attack paired analysis without relaunching the completed learned/LiRA jobs.

After the Fashion-MNIST and WDBC threshold analyses finish, run their full
learned and label-only attacks plus the prespecified representative LiRA subset:

```bash
bash commands/satml_run_added_attacks.sh
```

The added-domain LiRA subset covers all six depth-2 configurations in the first
three independent blocks (18 targets and 288 reference models per dataset).
Its paired uncertainty therefore uses three blocks and is robustness evidence,
not the primary endpoint.

```bash
watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/fashion_factorial/lira_representative/reference_training_status.csv \
  --expected 288

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/wdbc_targeted/lira_representative/reference_training_status.csv \
  --expected 288

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/fashion_factorial/label_only_hsj/target_scoring_status.csv \
  --expected 60

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/wdbc_targeted/label_only_hsj/target_scoring_status.csv \
  --expected 30
```

Monitor each long launcher from another terminal without hiding its output:

```bash
watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/credit_factorial/lira/reference_training_status.csv \
  --expected 1536

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/credit_factorial/lira/target_scoring_status.csv \
  --expected 96

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/credit_factorial/label_only_hsj/target_scoring_status.csv \
  --expected 96
```

## 8. Targeted encoding-scale experiment

```bash
bash commands/satml_run_encoding_scale.sh
```

The `alpha=1` baseline is reused from the confirmatory factorial. The script
trains only `alpha=0.5` and `alpha=2` targets and calculates within-block scale
contrasts.

## 9. Freeze and evaluate the privacy selector

Only after the development factorial and loss-threshold outputs are complete:

```bash
bash commands/satml_build_selector.sh
git diff -- satml_targets/selector
```

The command writes the policy decision and five fresh blocks. Commit or archive
the decision files before training the fresh targets. Then run:

```bash
bash commands/satml_run_fresh_selector.sh
```

Do not regenerate the selector decision after inspecting fresh results.

## 10. Frozen backend-derived noise studies (N1/N2/N3)

The retained MNIST checkpoints are ignored by Git. Import all 36 factorial
checkpoints from the completed NeurIPS workspace, without retraining them or
pooling old result tables with the new SaTML analysis:

```bash
export QURIFT_LEGACY_REPO='/absolute/path/to/quarift_neurips_rebutal_2'
bash commands/satml_import_legacy_mnist.sh
```

The importer copies files byte-for-byte and records SHA-256 values under
`satml_results/imported_mnist_manifest.json`. It refuses to overwrite a
different destination artifact. Confirm the summary says `targets=36` and
`artifacts=108`.

Credentials remain environment-only. Use IBM access once to capture a frozen,
credential-free snapshot:

```bash
export QISKIT_IBM_TOKEN='YOUR_NEW_TOKEN'
export QISKIT_IBM_INSTANCE='YOUR_INSTANCE_CRN'
export QURIFT_NOISE_BACKEND='ibm_kingston'
bash commands/satml_capture_noise_snapshot.sh
```

Never commit or print the token. Rotate any token that has appeared in chat,
terminal history, logs, or screenshots. The command prints the snapshot path.
Export that exact directory for every subsequent study, then credentials may be
unset because no target evaluation contacts IBM:

```bash
export QURIFT_NOISE_SNAPSHOT='satml_results/backend_snapshots/<printed-directory>'
unset QISKIT_IBM_TOKEN QISKIT_IBM_INSTANCE
```

Do not overwrite or recapture the snapshot between N1, N2, and N3. The saved
manifest hash is part of every condition and resume check.

### N1: full structural robustness

Run exact, ideal 512-shot, and frozen-noisy 512-shot inference over all 36
MNIST factorial checkpoints, ten simulator seeds, scalar attacks, and the
cross-fitted learned prediction-vector-plus-statistics attacker:

```bash
bash commands/satml_noise_n1_structural.sh
```

Monitor target completion and one target's condition-level progress:

```bash
watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/noise/n1_structural/conditions/target_status.csv \
  --expected 36

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/noise/n1_structural/conditions/MNIST_QNN_z_r1_d2_s43/condition_status.csv \
  --expected 21
```

The 21 conditions are one exact result plus ideal/noisy results for ten
simulator seeds. The analysis averages simulator seeds within each checkpoint
before using the three trained seeds for paired structural inference.

### N2: query and shot-allocation policy

Run the six fixed-depth targeted checkpoints under `1×128`, `1×512`,
`1×2560`, `5×128`, `5×512`, and `20×128` query/shot policies:

```bash
bash commands/satml_noise_n2_queries.sh
```

```bash
watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/noise/n2_query_policy/conditions/target_status.csv \
  --expected 6

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/noise/n2_query_policy/conditions/MNIST_QNN_z_r1_d6_s43/condition_status.csv \
  --expected 121
```

The compatibility command `commands/satml_noise_budget.sh` invokes this
corrected N2 study. Repeated queries are passed through the nonlinear head
separately and the returned probabilities are averaged; pooled-count results
are diagnostics only.

### N3: noisy LiRA attack breadth

Train two structure-matched reference banks of 16 models, save those
checkpoints, and use them to score six target checkpoints (two endpoints by
three model seeds) through the same ideal/noisy oracle:

```bash
bash commands/satml_noise_n3_attacks.sh
```

```bash
watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/noise/n3_attack_breadth/lira_references/reference_training_status.csv \
  --expected 32

watch -n 15 python satml_tools/status_progress.py \
  --csv satml_results/noise/n3_attack_breadth/noisy_lira/target_status.csv \
  --expected 6
```

N3 is a paired endpoint confirmation, not a full factorial. The optional noisy
label-only pilot has a much larger per-record query cost and is intentionally
separate:

```bash
bash commands/satml_noise_n3_label_only_optional.sh
```

All three launchers are resumable and reject a result created with a different
snapshot hash, sample selection, or aggregation protocol.

## 11. Generate submission artifacts

After all available result families finish, generate submission tables and
figures directly from the analysis CSVs:

```bash
bash commands/satml_generate_artifacts.sh
```

The artifact manifest lists every loaded and missing result family. Missing
analyses are omitted rather than rendered as zero-valued results. Markdown and
LaTeX tables are written below `satml_results/paper_artifacts/tables`, with PNG
and PDF figures below `satml_results/paper_artifacts/figures`.

## 12. Full verification

```bash
PYTHONPATH=.:reviewer_tools python -m unittest \
  test.test_satml_data \
  test.test_satml_capacity \
  test.test_satml_targets \
  test.test_satml_paired_analysis \
  test.test_satml_selector \
  test.test_satml_noise_budget \
  test.test_satml_noise_protocol \
  test.test_satml_lira_candidates \
  test.test_satml_import \
  test.test_satml_mechanism \
  test.test_satml_progress \
  test.test_satml_artifacts \
  test.test_satml_added_datasets \
  test.test_satml_end_to_end -v

python -m py_compile satml_tools/*.py reviewer_tools/*.py experiments/qurift_main.py
git diff --check
```

The end-to-end test constructs a temporary Credit-like snapshot, fits the
training-only preprocessing, trains a QNN for one epoch, exports attack data,
and verifies the model, preprocessing, provenance, membership counts, and
feature-angle scale.
