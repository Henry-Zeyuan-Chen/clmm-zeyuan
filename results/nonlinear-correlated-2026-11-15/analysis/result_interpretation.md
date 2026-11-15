# Nonlinear correlated-key control: independent results

The rank-32 setting is a qualified, non-ceiling component baseline at K8 and K16. The prespecified periodic-minus-continuous timing contrast remains inconclusive: the fixed-two-initializer effect is **−0.59 percentage points, 95% paired-history interval [−2.15, 0.98]**. Outer training improves late stable-query accuracy relative to the exact initial checkpoints at K8, while late coordinate MSE worsens for online, continuous and periodic. This provides a useful Member C memory foundation; it establishes no sleep/Hebbian novelty or full-language Titans result.

## Selection and qualification

The prospective design and frozen source `b5f27ff4738ded4832538573d454d2461786d38b` were recorded before outcomes. Rank 32, the first candidate, passed every calibration criterion in both original initializers; ranks 16 and 8 were **not run**. Selection saw only online pressure outcomes. Fresh confirmation passed, the comparison was frozen before replay evaluation, and no fallback/reselection was used. Initial checkpoints already scored 128/128 on the three short cells, so training was not necessary for their observed short-task accuracy.

The actual fresh comparison's four-target D0 all-correct rates are:

| Initial facts | Trained 90511 | Trained 90515 | Declared ≥80% gate in both |
|---|---:|---:|---|
| K8 | 128/128 | 128/128 | Pass |
| K16 | 125/128 | 124/128 | Pass |
| K32 | 75/128 | 77/128 | Fail; diagnostic only |

K32 delay results mix inadequate initial acquisition with subsequent interference. They remain in the tables and figures but do not support clean retention-efficacy claims. Untrained checkpoints remain diagnostic anchors under the declared protocol.

## Prespecified primary

K8, D64, two stable queries in each of 128 paired histories:

| Trained initializer | Continuous | Periodic | Periodic minus continuous, percentage points [95% interval] |
|---|---:|---:|---:|
| 90511 | 92/256, 35.94% | 88/256, 34.38% | −1.56 [−3.91, 0.39] |
| 90515 | 89/256, 34.77% | 90/256, 35.16% | 0.39 [−1.56, 2.34] |
| Mean of these two fixed initializers | 35.35% | 34.77% | −0.59 [−2.15, 0.98] |

All trained intervals include zero. This does not establish a periodic advantage, equivalence, or zero population effect. Bootstrap draws resample the same 128 history IDs across both fixed models, averaging each history's seed-specific differences before summarizing. There are 5,000 draws with seed 90504. The interval does not describe new training datasets, codebooks, or initialization populations; the 512 seed×query records are not independent replications. Exact untrained results, including the differing per-initializer signs, are retained in `primary.csv` and the figure rather than selected as favorable evidence.

## What outer training changes

For those same fresh K8 D64 stable histories, fixed-two-initializer accuracy means and paired intervals are:

| Arm | Exact initial checkpoint | Trained checkpoint | Trained minus initial, percentage points [95% interval] |
|---|---:|---:|---:|
| Online | 29.10% | 39.06% | 9.96 [5.08, 14.84] |
| Repeat | 14.45% | 40.43% | 25.98 [20.90, 31.05] |
| Continuous | 21.29% | 35.35% | 14.06 [9.37, 18.95] |
| Periodic | 21.88% | 34.77% | 12.89 [8.01, 17.58] |

These are secondary anchored comparisons of the same initializers, representation, histories and write schedules. Training improves accuracy at this condition; this is not a claim of uniform improvement across loads or metrics. Online coordinate MSE at D64 increases from 0.05603 to 0.07031; continuous increases from 0.06592 to 0.07497 and periodic from 0.06588 to 0.07590. At D0/D16, online MSE decreases. Accuracy and coordinate reconstruction error therefore tell different parts of the result; neither should replace the other post hoc.

The trained online anchor reaches 39.06% late stable accuracy using fewer writes. Continuous and periodic score 35.35% and 34.77%. Their secondary fixed-model contrasts against online are −3.71 [−7.03, −0.39] and −4.30 [−7.81, −0.98] percentage points. Repeat reaches 40.43%, or +1.37 [0.20, 2.73] points against online. These unadjusted secondary intervals describe this condition and combine differing write content/budgets; they are not a general causal statement against replay or evidence for a sleep mechanism. Online has K+2+D writes, while repeat/continuous/periodic have K+2+2D.

## Replay, labels and state

Continuous and periodic consume the same four frozen draws per block in the same order, differing only in placement relative to new observations. Literal FIFO capacity is eight pairs, with four frozen selections; draws use replacement. At the primary query, all 256 stable-query instances are evicted from the final FIFO. Of them, 102 received earlier replay, comprising 119 replay writes. Revised targets were replayed more often: 188/256 instances and 303 writes. These traces are shared across both models and schedules. Residency/exposure differences prevent treating stable-versus-revised performance as a pure revision effect. The recent distractor remains separate from the fixed four-target retention denominator.

Per episode, ordinary neural adaptive tensor backing is 128 KiB after a live write and 64 KiB after compact state export/import, unchanged across candidate ranks. Codebooks add 20 KiB, and the model registers 32,898 slow parameter elements, with only 16,384 initial-memory elements trainable offline. Reset initial states batch-share the slow initializer's backing; rolled controls independently copy weights and aux. Their batch-amortized physical values should not be confused with dedicated per-stream adapted state. Python/allocator overhead and training graphs are outside tensor-byte counts; literal replay counts are reported separately.

## Audit and reproduction scope

The independent artifact audit passes **143,616 main saved score/prediction rows, 1,536,000 causal plan writes and 2,232 serialization batches**. It reconstructs all labels, genuine revisions, exclusions, raw-score metrics, summary rows, all expected arms/queries, rolled donors, actual writes, state counters, storage layouts, shared histories, source/design/data/checkpoint hashes, and selection/freeze chronology. Static saved-tensor audit confirms fixed codebooks and that exactly the two permitted memory matrices change offline. There are 20 source files per invocation.

Only two training invocations ran: each has 178,944 supervised answers and 279,968 online observations, totaling **357,888 answers and 559,936 observations**. They share 38,400 episode draws, 38,393 unique whole histories, all 64 IDs and all 1,024 atomic bindings. Fresh evaluation excludes exact ordered training histories, not atomic bindings or local subsets. This is not unseen-key or language generalization.

The separate pre-outcome CPU mechanism proof matches current-state half-SSE gradients exactly and verifies rank-independent initialization/RNG and state continuation. Coordinator-run selected streaming inference rechecks cover 2,432 queries: the frozen GPU loader reproduces scores exactly, while the separate CPU portable loader differs by at most 3.58e−7. These are inference checks, not retraining or complete-grid numerical reproduction. A cross-version normalization-bit loader failure and its nonsemantic compatibility repair are detailed separately in `portability_review.md`; frozen run sources, checkpoint tensors and results remain unchanged.

The compatibility repair also separates CPU-aliased frozen/template and initial-memory parameter storage before strict loading, preserving both distinct saved copies without changing their values or update rule. A further optional portable-loader GPU check has **unknown status** after SSH disconnection/timeouts; neither a pass nor allocation release is established for that attempt. This does not alter the eight successful/downloaded main invocations or the observed strict-GPU and portable-CPU checks.

The numerical source tables are `primary.csv`, `initial_final.csv`, `calibration_initial_final.csv`, `fixed_seed_initial_final.csv`, `fixed_seed_secondary.csv`, `pressure_fixed_targets.csv`, `pressure_paired_contrasts.csv`, `retention_eligibility.csv` and `replay_residency.csv`. Machine-readable audits are `artifact_audit.json`, `checkpoint_audit.json` and `selection_audit.json`. Scripts are saved alongside the tables. `figures/primary_timing_contrast.{png,pdf}` and `figures/pressure_and_initializer.{png,pdf}` are generated from these tables and visually inspected.
