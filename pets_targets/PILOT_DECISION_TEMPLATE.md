# PETS pilot decision record

- Date/time:
- Commit:
- Pilot block: `pets_b01` (development only; excluded from confirmation)
- Hardware/software environment:
- Completed-target count:
- Protocol validator output:
- Partition protocol: `pets_label_matched_defense_attack_final_v2`
- Utility scope: full 2,000-record held-out test split
- Pre-correction result archive:

## Frozen settings

- Strong-L2 weight decay selected by utility-only tuning:
- HAMP gamma / alpha:
- DP epsilon selected by utility-only tuning / delta / derived noise multiplier:
- Output-optimizer iterations / learning rate:
- Lattice shots:
- Logit quantization step:
- Sticky input resolution and secret digest (never write the secret):
- HSJ records / query budget:
- LiRA references / Monte Carlo samples:
- Nearby-query count / radius:

## Utility-only selection

- Tuning manifest: `pets_targets/credit_defense_tuning_targets.csv`
- Eligibility thresholds: ROC-AUC >= 0.65, AP >= 0.30, minority recall >= 0.02,
  and no prediction collapse in either structural role
- Selection file: `pets_results/tuning/selection.json`
- Confirmatory manifest:
  `pets_targets/credit_defense_training_targets_confirmatory.csv`
- Confirm that no attack result was consulted: yes / no

## Decision

- Settings accepted or changed:
- Reason based on convergence/runtime/task-utility constraints (not desired privacy outcome):
- Files changed before confirmation:
- Confirmation blocks authorized: `pets_b02,pets_b03,pets_b04,pets_b05`
