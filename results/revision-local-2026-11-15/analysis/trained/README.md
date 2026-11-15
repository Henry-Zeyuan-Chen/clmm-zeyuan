# Trained analysis artifacts

- `raw_phase_audit.json`: complete independent initial/final saved-output audit and exact source/data checks. Final `joint_primary_success` isfalse.
- `saved_weights_audit.json`: actual48 initial/final payloads, exact original initial tensor values, retained codebooks/prototypes, and all24 saved optimizer states at1200steps.
- `training_counters.csv`: all24 complete matched training budgets, including every observation and answer count.
- `initial_*` and `final_*` CSVs: complete per-condition gates, primary contrasts, aggregate means and per-map results. All four declared final inequalities remain present, including both revision failures.
- `per_case_metrics.csv`: reconstructed initial/final metrics from raw outputs.
- `initial_to_final_contrasts.csv`, `initial_to_final_permap.csv`, `initial_to_final_audit.json`: secondary paired training effects and query-identity checks; nominal95% intervals.
- `interpretation.md`: primary non-success, qualification, secondary training contribution and limits.
- `figures/rule_accuracy.{png,pdf,svg}`, `figures/primary_contrasts.{png,pdf,svg}`, and `figures/initial_to_final.{png,pdf,svg}`: deterministic source-table plots; captions adjacent. PNGs were visually inspected.

Scripts are one level above. Commands used, with the coordinator's absolute paths substituted for these aliases:

```
python analysis/analyze_phase.py --root OUT/trained --out OUT/analysis/trained --repo WORK/source-checkout --prior-root OLD_ROOT/main --other-phase OUT/frozen
CPU_TORCH_PYTHON analysis/audit_saved_weights.py --root OUT/trained --parents OUT/parents --out OUT/analysis/trained/saved_weights_audit.json
python analysis/analyze_initial_final.py --root OUT/trained --out OUT/analysis/trained
python analysis/plot_phase.py --analysis OUT/analysis/trained --title "Matched trained rule comparison"
python analysis/plot_initial_final.py --analysis OUT/analysis/trained
```

No new model optimization/inference, accelerator use, source changes, rule selection, threshold relaxation or map selection was performed by this analysis task. Allocation accounting and selected GPU/Windows inference reproduction remain owned by the root coordinator.
