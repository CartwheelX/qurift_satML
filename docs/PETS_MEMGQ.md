# MemGQ: measurement-aware output sanitization

Let (z\in[-1,1]^m) be the post-PQC Pauli expectation vector, (Wz+b) the
frozen classical head, (p(z)=\operatorname{softmax}(Wz+b)), and (A) a
defense-side membership discriminator trained only on defense-calibration
records. MemGQ minimizes

\[
 |A(\operatorname{sort}(p(z')))|
 + \lambda\lVert p(z')-p(z)\rVert_1
 + \mu\,[\max_{j\ne y^*}p_j(z')-p_{y^*}(z')+\kappa]_+,
\]

where (y^*=\arg\max p(z)). After each update, the candidate is clipped to
physical expectation bounds. In the lattice condition it is projected onto

\[
 \mathcal L_S=\{-1+2k/S:k=0,\ldots,S\}^m,
\]

the exact expectation-value lattice induced by (S) binary Pauli shots.

The matched controls separate possible explanations:

- LogitGuard-continuous tests generic score sanitization;
- LogitGuard-quantized tests whether discretizing logits suffices;
- MeasurementGuard-continuous tests the effect of optimizing in measurement
  space without a shot lattice;
- LatticeRound tests the shot lattice without membership-aware optimization;
- MemGQ-lattice combines measurement-aware optimization and the lattice;
- MemGQ-lattice-sticky adds secret-shifted input buckets only as a nearby-query
  hardening ablation.

Every condition reports probability/measurement distortion, label preservation,
optimization validity, censored fraction, runtime, and adaptive attack metrics.
Task utility—including ROC-AUC, average precision, balanced accuracy,
minority-class recall, accuracy, and NLL—is computed separately on the full
held-out test split. A failed constrained search is recorded and never
relabeled as a successful sanitization.
