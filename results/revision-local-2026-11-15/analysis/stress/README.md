# Stress artifacts and reproduction

- `design.json`: written before stress inference; explicitly exploratory and preserves the original primary failure.
- `raw_audit.json`: complete independent audit, exact reinforced-history reconstruction, source/checkpoint lineage and nominal focus contrasts.
- `aggregate.csv`, `contrasts.csv`: all R1/R4/R16 × D0/D64 curves and differences, with the declared R16D64 exploratory focus flagged.
- `per_case_metrics.csv`, `short_and_stress_d0_gates.csv`: every fixed parent/map result and all short/D0 failures.
- `interpretation.md`: measured results, immediate EMA replacement failure, uncertainty and claim limits.
- `figures/reinforcement_curves.{png,pdf,svg}` and `figures/focus_contrasts.{png,pdf,svg}`: deterministic source-table plots, visually inspected PNGs; captions adjacent.

The executable driver is `WORK/revision_stress.py`; its exact executed copy is `OUT/stress/driver_snapshot.py`. All model code comes from immutable commit `c9f5c153f786f1b82ac383d01c1d18c6c595946b`. The source/data/checkpoint locations, full command and environment are recorded in `OUT/stress/metadata.json`. The original trained checkpoint files are reused by SHA; no duplicate weights were written.

Commands, with recorded absolute paths substituted for these aliases:

```
CPU_TORCH_PYTHON WORK/revision_stress.py --source WORK/source-checkout --trained OUT/trained --frozen OUT/frozen --prior OLD_ROOT/main --out NEW_OUT/stress --analysis NEW_OUT/analysis/stress
python OUT/analysis/stress/analyze.py --root OUT/stress --out OUT/analysis/stress
python OUT/analysis/stress/plot.py --analysis OUT/analysis/stress
```

Use new output directories for inference. The driver enforces a global 600-second deadline and refuses existing output/design directories. No optimizer or GPU work is included. The saved outputs and analysis scripts remain available after this task; no background inference or monitoring remains active.
