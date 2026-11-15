# Separate dense-key instrument calibration

The single fixed dense-key linear control detects substantial interference after adequate K8 acquisition. It does **not** establish a periodic-replay advantage: the prespecified primary interval includes zero. This supplemental configuration was specified after the main orthogonal ceiling and before its own outcomes; it does not change neural checkpoint selection or qualification.

All three short gates achieve 128/128 all-correct episodes. The K8 D0 four-target gate achieves 31/32 (96.875%), exceeding its declared 80% threshold. The initial load results are:

| Initial facts | D0 four-target accuracy | D0 four-target all-correct |
|---|---:|---:|
| K8 | 127/128 (99.22%) | 31/32 (96.88%) |
| K16 | 122/128 (95.31%) | 26/32 (81.25%) |
| K32 | 97/128 (75.78%) | 8/32 (25.00%) |

K8 is eligible for the prespecified retention comparison. K32 starts with weak acquisition, so its later performance mixes initial load failure with subsequent interference and cannot be called a clean retention result.

At K8 D64, four-target online accuracy falls to 35/128 (27.34%). For the **prespecified stable-query periodic-minus-continuous contrast**, periodic answers 20/64 correctly and continuous answers 17/64: **+4.6875 percentage points**, with a **95% paired-episode bootstrap interval [0.0000, 10.9375] points**. The bootstrap uses 32 whole episodes, 5,000 draws, and seed 90404. The point difference is three answers; the interval includes no advantage. This uncertainty concerns sampled episodes for one fixed codebook, not variation over codebook seeds or training runs.

All 64 stable-query instances in that primary cell are absent from the final FIFO. Of them, 26 receive at least one earlier replay in the shared frozen trace. Targets never recur as delay observations. Continuous and periodic consume the same causal frozen source pairs in the same order, with matched write opportunities. Repeat and online have equal observed accuracies throughout this calibration. For a unit-norm linear key and step one, an immediate second write is idempotent up to floating-point error; matching write opportunities does not imply equal nonzero gradient magnitudes.

Important scope limits remain:

- The key geometry and fast-matrix dimension both differ from the orthogonal Delta control (64 normalized Gaussian vectors in dimension 32; a 32×32 state). Their between-control difference is not a single-factor ablation. Comparison with the trained nonlinear control changes additional model/training factors.
- Stable and revised targets differ in age and FIFO residency. Compare schedules within each stratum; their cross-stratum gap is not a pure revision effect.
- This is one untrained linear reference with no learned slow parameters, no parameter search, and no new accelerator allocation. It is not a Titans, sleep, or Hebbian method result.

The independent audit reconstructs all labels and summaries, validates 94,656 causal plan writes and 162 serialization batches, verifies code/source/checkpoint fingerprints, and reproduces all 9,792 saved score vectors and argmaxes exactly with a separate direct linear implementation. Adaptive tensor storage is 4,096 bytes per stream; logical FIFO/snapshot limits remain eight/four pairs. The 38,395 planned training histories are used only for exclusion; this control receives no offline training.

Evidence: `dense_calibration_audit.json`, `dense_pressure_fixed_targets.csv`, `dense_pressure_paired_contrasts.csv`, and `audit_dense_calibration.py`. The separately generated and visually inspected figure is `figures/dense_interference_calibration.png` with vector PDF, reproducible through `plot_dense_calibration.py`.
