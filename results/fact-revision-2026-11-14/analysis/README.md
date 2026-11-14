# Independent saved-artifact audit

`analyze_fact_revision.py` uses Python, NumPy and CPU PyTorch to read saved tensors. It does not import the experiment runner, data generator or model, and does not rerun inference. Outputs must be new directories beneath this analysis directory. Failed audits retain `verification.json` with a nonzero process exit; existing run artifacts are never changed.

Main API:

```text
python analyze_fact_revision.py --root ROOT --out ANALYSIS/NEW --expected-commit FULL_40_CHARACTER_SHA
```

Optional `--parent-68101 PATH` and `--parent-68105 PATH` override local parent-checkpoint locations. Defaults are `ROOT/../clmm-shared-pretrain-2026-11-14/lm-SEED/checkpoint.pt`. The canonical run names are `before-SEED`, `adapt-SEED`, and `after-SEED`, for both fixed seeds. All six must succeed and obey the prospective design. Main mode requires 1,600 adaptation steps, batch 16, fixed seeds and the complete evaluation suites. A failed acquisition gate still permits artifact verification, but decisions label pressure results as floor diagnostics.

The audit checks source snapshots and hashes, checkpoint lineage and finite tensors, actual training bytes/targets/offsets, exact shared training plans, the prescribed curriculum, namespace isolation, every evaluation prefix, online-only target labels, causal replay selections and budgets, and CSV/logit correspondence by semantic sample identity. It independently computes full-vocabulary and forced-choice metrics, legal-answer mass, obsolete-answer rates, heuristic baselines, counterfactual response checks, paired adaptation effects and replay comparisons. The 5,000-draw bootstrap resamples complete episodes and recomputes ratios; the two K2 questions remain clustered. Checkpoints are never pooled.

Outputs: `verification.json`, `source-manifest.json`, `estimates.csv`, `contrasts.csv`, and `decisions.json`. The manifests certify saved evidence, not execution truth beyond that evidence. Training bytes are parsed and checked against the saved selection plan and latest online labels; this is not a second training run.

Diagnostic APIs, which cannot certify the main experiment:

```text
python analyze_fact_revision.py --selfcheck --selfcheck-data SAVED_EVAL_RUN --out ANALYSIS/NEW_SELFCHECK
python analyze_fact_revision.py --smoke-run SAVED_SMOKE_RUN --out ANALYSIS/NEW_SMOKE_AUDIT
```

`--selfcheck-data` is optional. It adds corruption checks on detached in-memory copies of actual saved samples, never editing their source artifacts. Smoke mode accepts its own recorded reduced budget and source commit; do not pass the final expected commit when auditing an earlier smoke run. Its outputs always say `main_certified: false`.

`plot_fact_revision.py --analysis VERIFIED_OUTPUT --out ANALYSIS/NEW_FIGURES` reads certified tables only and creates PNG/PDF figures with full accuracy axes and paired difference intervals. It rejects smoke or failed audits. No plot is generated until main data are available. Checkpoint-specific intervals describe sampled development episodes; they do not estimate training-seed variability. Stale unmarked replay remains a conflicting-input diagnostic, and K4 partial-block delay comparisons remain exploratory.
