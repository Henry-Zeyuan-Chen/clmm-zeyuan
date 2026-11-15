# Frozen analysis artifacts

- `raw_phase_audit.json`: complete independent saved-output audit and exact source/data checks.
- `saved_weights_audit.json`: all48 actual payloads and nine parent tensors bitwise unchanged.
- `final_contrasts.csv`: four candidate inequalities plus nominal factorial mechanism contrasts, with explicit interval levels.
- `final_gates.csv`: every parent/map/rule admission, including the unchanged EMA revision failure.
- `final_aggregate.csv`, `final_permap.csv`, `per_case_metrics.csv`: source tables for means, variability and all fixed controls.
- `interpretation.md`: measured numbers, mechanism evidence, qualification and limits.
- `figures/rule_accuracy.{png,pdf,svg}` and `figures/primary_contrasts.{png,pdf,svg}`: deterministic figures generated from exported audited tables; captions in `figures/captions.txt`. PNGs were visually inspected.

Scripts live one level above this directory. Execute with a fresh output path or preserve earlier outputs deliberately. Commands used were the following, with absolute input/output paths as recorded by the coordinator:

```
python analysis/analyze_phase.py --root OUT/frozen --out OUT/analysis/frozen --repo WORK/source-checkout --prior-root OLD_ROOT/main
CPU_TORCH_PYTHON analysis/audit_saved_weights.py --root OUT/frozen --parents OUT/parents --out OUT/analysis/frozen/saved_weights_audit.json
python analysis/plot_phase.py --analysis OUT/analysis/frozen --title "Frozen rule intervention"
```

No model inference, new training, accelerator work or partial trained-outcome inspection was performed by this analysis task. The root coordinator owns allocation accounting and any selected inference reproduction.
