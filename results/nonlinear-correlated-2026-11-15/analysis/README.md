# Reproduce the saved-artifact analysis

These scripts read the downloaded nonlinear correlated-key artifacts and write only analysis outputs. They do not allocate an accelerator, train a model, or regenerate experimental predictions. The frozen model/data/evaluator source is `b5f27ff4738ded4832538573d454d2461786d38b`; source copies and exact commands are saved per invocation. Keep the eight main run directories, design, selection/freeze records and raw evaluation files together.

The original analysis used a local Python 3.13 environment with NumPy and PyTorch 2.15.0+cpu for tensor/artifact checks; plotting used system Python with Matplotlib 3.10.3. Edit the paths below for another machine. Each command is independent shell code; run them in this order:

```powershell
$TaskArtifacts = 'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-nonlinear-correlated-2026-11-15'
$TaskAnalysis = Join-Path $TaskArtifacts 'analysis'
$TaskPython = 'C:\Users\henry\Documents\Codex\2026-10-22\i\work\clmm-test-env\Scripts\python.exe'
$TaskPlotPython = 'C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe'

& $TaskPython -B (Join-Path $TaskAnalysis 'audit_artifacts.py') --root $TaskArtifacts --output (Join-Path $TaskAnalysis 'artifact_audit.json')
& $TaskPython -B (Join-Path $TaskAnalysis 'audit_checkpoints.py') --root $TaskArtifacts --out $TaskAnalysis
& $TaskPython -B (Join-Path $TaskAnalysis 'analyze_results.py') --root $TaskArtifacts --out $TaskAnalysis
& $TaskPython -B (Join-Path $TaskAnalysis 'summarize_secondary.py') --root $TaskArtifacts --out $TaskAnalysis
& $TaskPlotPython -B (Join-Path $TaskAnalysis 'plot_results.py') --analysis $TaskAnalysis
```

The artifact audit reconstructs labels and raw-score metrics; checks every expected query/arm, causal FIFO/snapshot plan, exact-history exclusion, rolled donor, write counter, serialization/storage record, source/design/data/checkpoint hash and shared dataset. Static checkpoint inspection checks literal codebooks and permitted offline tensor changes without running inference. Selection analysis reconstructs the ordered final-checkpoint gates, confirmation, comparison freeze and actual per-K D0 eligibility. It keeps initial/final and secondary comparisons separate from the trained primary.

The trained primary is K8 D64 stable periodic-minus-continuous accuracy. Each seed uses 128 whole histories; the fixed-two-seed mean first averages differences within the same history ID. All 5,000 bootstrap draws use seed 90504. The gate's cyclic rolled-state difference instead resamples eight blocks of 16 episodes. Fixed-model/history intervals are not population-training or codebook-seed intervals. Secondary intervals are unadjusted exploratory summaries.

Expected checks and outputs:

- `artifact_audit.json`: 143,616 main score rows, 1,536,000 plan writes, 2,232 serialization batches; two actual training runs, 357,888 supervised answers and 559,936 observations. Smoke rows are counted separately.
- `checkpoint_audit.json`: exact saved codebook identity, only the two allowed memory matrices changed offline, and all six evaluation checkpoint references resolve to the retained initial/final files.
- `selection_audit.json`: selected rank 32, calibration and confirmation pass; ranks 16/8 were not run. Actual trained D0 qualifies K8/K16, not K32.
- `primary.csv`: trained fixed-seed mean −0.005859375 with interval [−0.021484375, 0.009765625] in accuracy units; multiply by 100 for percentage points.
- `initial_final.csv` and `calibration_initial_final.csv`: per-seed accuracy/MSE changes on paired histories. `fixed_seed_initial_final.csv` and `fixed_seed_secondary.csv` preserve fixed-two-seed pairing at the primary condition.
- `pressure_fixed_targets.csv`, `pressure_paired_contrasts.csv`, `retention_eligibility.csv` and `replay_residency.csv`: fixed four-target curves, separate stable/revised/recent strata, qualification and literal-buffer exposure.
- `figures/primary_timing_contrast.{png,pdf}` and `figures/pressure_and_initializer.{png,pdf}`: regenerated numerical figures, visually inspected after final rendering.

`result_interpretation.md` states supported conclusions and limitations. `design_review.md` preserves the prospective review and pre-outcome design amendment. `adapter_cpu_audit.json` records a separate 12.87-second pre-outcome API/analytic-update proof over all three rank constructions; it is not a training or efficacy result. Its optional reproduction requires a complete checkout of frozen commit `b5f27ff` and the model dependencies:

```powershell
$TaskFrozenRepo = 'C:\Users\henry\Documents\Codex\clmm-replay' # Use a clean b5f27ff checkout here.
& $TaskPython -B (Join-Path $TaskAnalysis 'audit_adapter_cpu.py') --repo $TaskFrozenRepo --out (Join-Path $TaskAnalysis 'adapter_cpu_audit.json')
```

Inference verification is distinct from these analysis commands. The coordinator's `gpu-recheck/verification.json` records 2,432 selected queries and real streaming resumes under the frozen strict loader, with exact scores. `cpu-portable-recheck/verification.json` records 2,432 argmax matches under the separate literal-codebook portable loader, maximum score error 3.5762786865234375e−7. Neither is full-grid numerical reproduction or retraining. `portability_review.md` explains the normalization-bit incompatibility and CPU registered-parameter alias fix; both are post-run compatibility work that preserves actual saved codebooks/weights and the inherited update/read rule.

A further optional portable-GPU attempt ended with SSH disconnection/timeouts and has unknown result/closure status. Do not infer a pass or allocation release from the completed strict-GPU/portable-CPU records. The coordinator owns scheduler status and packaging; the eight main model invocations are complete and downloaded.
