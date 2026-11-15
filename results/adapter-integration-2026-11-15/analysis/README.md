# Reproduce the adapter-integration analysis

Frozen model/runner source is `a41bb097af3e7eed356d13e24e4a0d5bc6677b14`. Per-run source copies, configs and exact commands are retained. These scripts read saved artifacts and write only analysis outputs; none allocates a GPU, trains a model or regenerates experimental model predictions.

Use Python with NumPy and CPU PyTorch for artifact/static-tensor checks; plotting additionally needs Matplotlib. The original artifact environment is Python3.13.3, PyTorch2.15.0+cpu and NumPy2.5.4; plotting used system Matplotlib3.10.3. Relocate the paths for another machine:

```powershell
$IntegrationRoot = 'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-adapter-integration-2026-11-15'
$IntegrationAnalysis = Join-Path $IntegrationRoot 'analysis'
$IntegrationPython = 'C:\Users\henry\Documents\Codex\2026-10-22\i\work\clmm-test-env\Scripts\python.exe'

& $IntegrationPython -B (Join-Path $IntegrationAnalysis 'audit_artifacts.py') --root $IntegrationRoot --out (Join-Path $IntegrationAnalysis 'artifact_audit.json')
& $IntegrationPython -B (Join-Path $IntegrationAnalysis 'audit_run_coverage.py') --root $IntegrationRoot --out (Join-Path $IntegrationAnalysis 'run_coverage_audit.json')
& $IntegrationPython -B (Join-Path $IntegrationAnalysis 'analyze_results.py') --root $IntegrationRoot --out $IntegrationAnalysis
& $IntegrationPython -B (Join-Path $IntegrationAnalysis 'summarize_controls.py') --root $IntegrationRoot --out $IntegrationAnalysis
python -B (Join-Path $IntegrationAnalysis 'plot_results.py') --analysis $IntegrationAnalysis
```

The last command uses the system Python with Matplotlib; replace it with the appropriate interpreter if necessary. Keep all19 main/probe folders, the supplied shared-checkpoints, pipeline, design and prospective freeze record together. The compact Git result index omits raw arrays/scores and cannot alone run these auditors; use the complete extracted artifact root.

Expected outputs:

- `artifact_audit.json`:160,768 main raw-score rows,1,107,712 causal writes,2,960 state-serialization batches; eight main training runs,1,431,552 supervised answers and2,239,744 observations. The engineering probe/CPU smokes are counted separately.
- `run_coverage_audit.json`:19 declared invocations with exact order/status/CLI checks,815 required file SHA256s,27 source files per invocation, complete optimizer payloads and saved combined-session bundles. This complements the coordinator's verified compressed transport SHA after the tar directory-mtime warning.
- `admission.csv`: each new system's short/reset/rolled admission and per-K D0 gate. All four trained systems admit K8; only top-one associative systems admit K16. Neural anchors are descriptive. Forks have their own admission labels.
- `primary.csv`: ONLINE K8,D64 stable Hebbian top-one-minus-dense. The fixed-two-seed mean is0.16015625, interval[0.08984375,0.234375] in accuracy units; multiply by100 for percentage points.
- `system_contrasts.csv`: trained-system and fixed-slow-weight transfer contrasts, kept separate. `analysis_audit.json` records all complete pairs and the exact7,296-vector gradient/dense sham-fork match.
- `pressure_fixed_targets.csv`, `timing_secondary.csv`: constant-denominator four-target metrics and stable/revised/recent strata, plus descriptive replay timing comparisons.
- `initial_final.csv`, `fixed_seed_initial_final.csv`: exact initializer comparisons, paired on the same histories. `fixed_seed_timing.csv` preserves pairing across both fixed initializers for secondary timing summaries.
- `learned_dynamics_resources.csv`: eta/alpha, final query-state saturation, logical active-module updates, adaptive tensor bytes and instrumented batch wall times. These timings include controller/validation/serialization overhead.
- `figures/online_retention.{png,pdf}`, `primary_access_contrast.{png,pdf}`, `frozen_transfer.{png,pdf}`: numerical figures generated from these CSVs and visually inspected. Failed per-K admission is shown with diagnostic/dashed curves; the neural architecture anchor is labeled descriptive.

The main bootstrap resamples the same64 whole pressure-history IDs across both seeds and conditions (5,000 draws, seed90604). The short normal-minus-rolled gate instead resamples eight cyclic-roll blocks of16 episodes. Fixed-checkpoint/history intervals are not uncertainty over training procedures or codebooks. This is a development pilot with unadjusted intervals, not a broad confirmatory or novelty claim.

`adapter_cpu_audit.json` is a separate pre-outcome equation/state/session proof, not training or efficacy evidence. Its optional reproduction uses the frozen source and an actual shared neural checkpoint:

```powershell
& $IntegrationPython -B (Join-Path $IntegrationAnalysis 'audit_adapter_cpu.py') --repo <frozen-repo-or-complete-source-root> --checkpoint (Join-Path $IntegrationRoot 'shared-checkpoints\neural-90511.pt') --out (Join-Path $IntegrationAnalysis 'adapter_cpu_audit.json')
```

The coordinator's `gpu-recheck/verification.json` and `cpu-recheck/verification.json` are separate selected inference/session-resume checks:3,616 queries across18 models, GPU exact scores and CPU maximum raw-score error9.5367431640625e−7 with exact argmaxes. They do not retrain models or reproduce every grid score. `result_interpretation.md` states the supported result and limitations; `design_review.md` and `prospective_freeze_review.json` preserve the pre-execution review and runtime-estimator correction.
