# Independent review of the typed key/value control

Review completed on 2026-11-15 for the main protocol begun on 2026-11-14. Frozen model/data/evaluation source: `ff62df0f85ff7cbf62f3209755ea066959aa7085`. The reviewer edited only this analysis directory, performed no accelerator work, and did not retrain either model.

## Decision

Both fixed neural initializers pass the declared short-acquisition and genuine-revision admission and the separate confirmation. The stateful fixed-codebook baseline is qualified for those sampled closed-vocabulary compositions. The complete orthogonal pressure grid is at an accuracy ceiling, so it supplies no measured periodic-replay accuracy advantage and no sleep or Hebbian result. Initial neural checkpoints already pass the short accuracy gates; outer training improves coordinate reconstruction error, but its necessity for short-task accuracy is not established.

No source or artifact blocker was identified in the checked scope. This is a deliberately simplified memory-component interface, not a full pretrained Titans/byte-language model. The earlier byte-model transfer failures remain unresolved by this component result.

## Independent checks

- Recomputed every main prediction from its saved 16-coordinate score vector and reconstructed all answers from the online key/value history: **47,808 main score/prediction rows**, including initial and final checkpoints, admission, confirmation, and pressure.
- Independently simulated FIFO refresh/eviction and frozen block schedules, checking source events, values, causal boundaries, continuous/periodic source equality, repeat matching, common prefill, target/distractor disjointness, and actual/state write counts: **304,704 main planned writes**.
- Verified **918 main state round-trip batches** report exactly unchanged read scores, with side-effect-free reads and unchanged slow parameters asserted in the frozen runner. A separate CPU mechanism audit independently verifies read immutability, exact save/load/continue behavior, and current-state gradient updates.
- Verified saved source/data/checkpoint/design hashes, shared training arrays, shared evaluation histories across paired models, exact training-history exclusions, genuine revisions, distinct protected target values, rolled-state donor indices, and all summary rows.
- Reloaded final checkpoints independently on CPU and reproduced **540 selected saved answers**. Every argmax matched; largest raw-score difference from the saved GPU output was **3.5762786865234375e-7**. Initial-to-final checkpoint changes are exactly the two allowed `engine.memory_model_parameters` matrices. This is a bounded inference reproduction, not retraining.
- Confirmed no vendored `titans_pytorch` changes between the baseline source and frozen implementation. The nonlinear adapter's explicit deviations are recorded in the design and checkpoints.

The two CPU smokes are audited separately as engineering checks, with 564 score rows and 5,862 plan writes. Their two-step, batch-two, pre-freeze settings do not count as protocol qualification.

## Qualification from raw predictions

Each cell has 128 episodes and requires every query in an episode to be correct. K2 and revised K2 thresholds are 90%; K4 is 80%. The normal-minus-rolled interval resamples eight cyclic-roll blocks of 16 episodes, 5,000 draws with seed 90304.

| Checkpoint | Acquire K2 | Acquire K4 | Revise K2 | Normal minus rolled K2, 95% block interval |
|---|---:|---:|---:|---:|
| Development, initializer 90301 | 128/128 | 128/128 | 128/128 | 1.0000 [1.0000, 1.0000] |
| Development, initializer 90305 | 128/128 | 128/128 | 128/128 | 1.0000 [1.0000, 1.0000] |
| Confirmation, initializer 90301 | 128/128 | 128/128 | 128/128 | 0.9922 [0.9766, 1.0000] |
| Confirmation, initializer 90305 | 128/128 | 128/128 | 128/128 | 0.9844 [0.9609, 1.0000] |

The first candidate, inner step 0.5, passed with both original initializers and was selected. The step-1 alternative was never run. Selection and confirmation were reconstructed from raw prediction rows and match `selection.json`. These intervals describe the sampled development episodes for fixed checkpoints; they are not confidence intervals for a training procedure over new training data or initializers. Degenerate intervals at an observed ceiling do not prove zero population error or method equivalence.

## What outer training changed

Both initial checkpoints already achieve 128/128 all-correct on each of the same three short cells. Final accuracy remains 128/128. Offline updates reduce the mean squared error of the 16 read coordinates:

| Initializer | Cell | Initial MSE | Final MSE |
|---|---|---:|---:|
| 90301 | Acquire K2 | 0.0428294 | 0.000009739 |
| 90301 | Acquire K4 | 0.0429115 | 0.000015460 |
| 90301 | Revise K2 | 0.0434651 | 0.000225679 |
| 90305 | Acquire K2 | 0.0434369 | 0.000010116 |
| 90305 | Acquire K4 | 0.0433296 | 0.000015778 |
| 90305 | Revise K2 | 0.0440140 | 0.000224489 |

The initial/final comparisons are paired on exactly the same episodes. Their full values and paired-episode MSE intervals are in `training_initial_final.csv`. No untrained pressure sweep was run, so the result does not isolate whether outer training was necessary for pressure performance.

Each of the two actual training invocations uses 1,200 updates, batch size 32, **177,728 supervised answers**, and **278,112 online observations**. They share the same generated training arrays. There are 38,400 sampled episode draws, 38,395 unique whole histories, all 64 key IDs, and all 1,024 possible atomic key/value bindings. Evaluation excludes exact ordered training histories, while atomic bindings and local subsets remain familiar. Therefore the supported transfer claim concerns fresh whole histories/compositions, not unseen individual bindings, keys, language, or wording.

## Orthogonal pressure result

At each K in {8,16,32} and D in {0,16,64}, each initializer has 32 pressure episodes. The same four initial targets—two stable and two revised—are the fixed retention denominator. **Both neural models achieve 100% per-query and four-target all-correct accuracy in every ordinary arm and cell:** online, repeat, continuous, and periodic. Delta also gives its construction-expected exact ordinary-arm accuracy.

Consequently every observed periodic-minus-continuous accuracy contrast in this main grid is zero. This is a ceiling-limited null comparison, not evidence of equivalence or a general finding against replay timing. Replay decreases coordinate MSE relative to the online anchor in these measured cells, but all four schedules already produce the correct answers, and online has fewer writes. Small coordinate-error differences cannot be promoted into a sleep-consolidation accuracy benefit.

Boundary-reset accuracy is low while ordinary accuracy is perfect, supporting the role of the saved adaptive state on these histories. Reset is a separate ablation; it discards the prefill state and does not have the same effective state-write count as an ordinary arm.

Stable and revised targets differ in observation age and FIFO residency. Comparisons are valid within each stratum across matched schedules; a cross-stratum gap is not an isolated revision cost. The recent distractor is kept separate because it appears only at D>0. The changing all-query aggregate must not be interpreted as a constant-denominator retention curve.

## State and resource accounting

Per episode, Delta's adaptive tensor backing storage is 16,384 bytes. The neural state is 65,536 bytes after compact export/import and 131,072 bytes after a live write, because its active tensor views retain unused scan backing storage. These are actual unique tensor-storage counts, not counts obtained by summing aliased aux tensors. Each method also has 20,480 bytes of fixed key/value codebook tensors. The neural adapter registers 32,898 slow parameter elements, of which 16,384 initial-memory elements are trainable offline.

Continuous and periodic policies have a logical ceiling of eight working pairs plus four frozen selections. Snapshot draws are with replacement, so four write opportunities need not represent four distinct records. Repeat requires no replay buffer. Logical pair counts do not assert Python object sizes. Evaluator histories/plans remain saved analysis traces; reads receive no history, buffer, labels, or replay plan. Tensor-storage counts exclude Python/container overhead, allocator reservations, and training autograd graphs.

The launcher originally failed while serializing a NumPy boolean after Delta and the first neural invocation had succeeded. Their commands and original timings match the resumed record exactly, and both are marked reused. The failed launcher artifacts are preserved. This was not a failed model or an excluded weak result. The model/data source and protocol remained frozen.

## Reproducible artifacts

- `main_artifact_audit.json` and `audit_artifacts.py`: full trace/hash/summary audit.
- `adapter_cpu_audit.json` and `audit_adapter_cpu.py`: independent analytic gradient/state audit.
- `cpu_checkpoint_reload_audit.json` and `audit_cpu_reload.py`: bounded learned-checkpoint inference reproduction.
- `qualification_audit.json`, `independent_gates.csv`, and `analyze_results.py`: admission and confirmation reconstruction.
- `training_initial_final.csv`, `pressure_fixed_targets.csv`, and `pressure_paired_contrasts.csv`: recomputed numerical source tables.
- `launcher_resume_audit.json`: operational-failure and completed-run reuse check.
- `figures/initial_final_control.{png,pdf}` and `figures/orthogonal_pressure_ceiling.{png,pdf}`, generated by `plot_results.py` and visually inspected.

A separately predeclared, single-configuration dense-key calibration was requested after observing the main ceiling. Its separate audit and interpretation are complete in `dense_calibration_audit.json` and `dense_interpretation.md`. It passes short/K8 acquisition and detects delayed interference, but its primary periodic-minus-continuous interval includes zero. It remains outside neural selection and confirmation. Its representation and state dimension both differ from the orthogonal Delta control, so it is not a single-factor causal ablation.
