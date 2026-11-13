# Memory-pressure pilot evidence

See [the research notes](../../docs/MEMORY_PRESSURE.md) for protocol, results, caveats, and next steps. This directory contains lightweight, independently verified aggregates and the recorded designs. The initial failed training run and full raw data are preserved in the external artifact package, not duplicated here.

- `all-three/`: counts, contrasts, paired episode counts, original two-seed decision, verification, and exact analyzer source. The original criterion failed; seed 48105 is supplementary.
- `longgap-*/`: explicitly posthoc results requiring at least 64 observations since the last selected block.
- `training_audit_*.json`: full-budget and checkpoint checks; heldout scores differ substantially.
- `analysis/`: reconstruction/statistics utilities and focused analysis checks.
- `cluster/` and `pytest-final-local.log`: runtime environment, GPU, allocation accounting, and 91 passing targeted tests on each platform.
- `decay-diagnostic-design.json`: planned, NOT executed. `execution_status.json` records what actually ran.

Source commit for the pressure runs: `d93ac37dbdc7345cf8fc1434d42589cafef544e8`. The later documentation commit does not change experiment code. Baseline tag `baseline-v0.6.0` remains at `35a549f137548d48aecc26e33c6f8a34a6efc97c`.

Full artifact directory: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-pressure-2026-11-13`. Adjacent archive: `CLMM_Memory_Pressure_2026-11-13.zip`. Remote copy of raw runs: `/insomnia001/depts/edu/users/zc2819/clmm/pressure-2026-11-13`. Nothing has been pushed or merged.

The original metadata retains absolute run paths as provenance. For portability, pass new local `--run`, `--out`, and `--design` paths to the supplied analysis scripts. They reconstruct traces and metrics from saved predictions; they do not rerun neural inference. Run the training audits against the downloaded checkpoints before plotting.
