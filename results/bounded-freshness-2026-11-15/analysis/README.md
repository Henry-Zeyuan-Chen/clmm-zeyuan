# Independent analysis handoff

Frozen executable source: `9ab3901860548d2d87a9afc61e8ce58f950a03ce`. Main data SHA256: `1fcc5d7a2a475ce18a98019f0d571aa426420dfa047bc435e5292856b0376a70`. All 12 original C9 step-1200 local-delta checkpoints are retained, with zero additional optimizer steps.

Read `result_interpretation.md` for the evidence scope and `summary.json` for machine-readable co-primary entries (`category`, `estimate`, `lower`, `upper`). The D128 revised effect is +10.47 percentage points, with a 97.5% paired-history interval of [+7.76,+13.28]; stable is −0.015 points [−0.127,+0.061]. Both bounded schedules pass 24/24 own-D0 gates each; both literal schedules pass 0/24 each. This is revision/interference repair evidence rather than a comparison between two admitted retention conditions. The bounded arms do not outperform the simpler online/repeat anchors; their stable accuracy is lower, and revised differences versus those anchors have descriptive 95% intervals covering zero.

The bootstrap conditions on the two fixed parents and six fixed routing maps, using 128 paired histories, 5,000 PCG64 draws, seed 91204 and 97.5% intervals for the two co-primary endpoints. Balanced routing means 0.5 content plus 0.5 mean of all five random maps. No model, map, phase or gate failure was dropped. Secondary results and 95% intervals remain descriptive.

## Evidence files

- `initial_protocol_review.md`, `source_review.json`, `model_scale_audit.json`: prospective critique, frozen causal API review and 81,921-parameter/64-KiB component geometry.
- `packet_audit.json`, `engineering_packet_audit.json`: exact private-RNG generator reconstruction, training exclusions, checkpoint lineage and all 45 preserved-source/Git-blob hashes.
- `supplemental_exclusion_audit.json`: one postlaunch exclusion-coverage omission, exact prior paths/hashes, the 275-file numeric-history inventory and unchanged regeneration under expanded exclusions. Original packets remain immutable.
- `audited_signature_superset.json`: 89,094 signatures comprising the main launch union plus the omitted integration history. Before a later study, append this study's main histories and any later consulted inputs.
- `cases/*.json` and `cases/*.npz`: authenticated per-model raw audits and paired analysis vectors. All 433,152 score vectors, 13,192,704 applied writes, 73,728 chronological plans and 4,608 resource batches were checked.
- `gates.csv`, `aggregate.csv`, `contrasts.csv`, `per_condition.csv`, `diagnostics.csv`: final gate, accuracy, contrast and exposure tables, retaining every checkpoint/map/phase/arm.
- `replay_telemetry.*`, `plan_telemetry_per_history.csv`, `bounded_schedule_content_differences.csv`: model-free candidate, refresh, fallback and staleness reconstruction. Bounded continuous/periodic share candidates but can apply different values.
- `runtime_gate_audit.json`: runtime-only continuation gate and invocation status preservation. No engineering efficacy was inspected by this reviewer.
- `numerical_proof_review.json`: inspection of the coordinator's separate actual CPU proof. It covers the first 16 histories of neutral K16 online and phase-2 D128 with all six arms across all 12 checkpoints: 9,216 scores, exact error zero, resume checks and unchanged slow/RNG assertions. This is selected numerical reproduction, not a numerical rerun of the full grid. This reviewer ran no inference.
- `figures/manifest.json`, `figure_qa.json`: five visually checked figures in PNG, PDF and SVG, bound to their source tables and plotting script.
- `analysis_execution.json`, `final_artifact_inventory.json`: verification scope/environment and final file hashes. The inventory excludes itself and Python bytecode caches.

## Exclusion qualification

The launch collector enumerated prior `episodes.json` files plus two explicit training arrays. One seed-90323 history, embedded in two older `verification.json` files, was absent from that union. The main union plus that signature contains 89,094 entries and excludes all 1,280 main histories. Separately, the engineering launch union plus the same signature contains 88,934 entries and excludes all 160 engineering histories. Both exact regeneration audits reproduce all accepted histories unchanged. The main union intentionally includes the engineering histories. This coverage omission is preserved and disclosed; no packet, outcome or checkpoint was rewritten.

## Reproducing the saved-output analysis

Run with Python, NumPy and Matplotlib from this directory. The scripts deliberately import no model, evaluator or production generator/scheduler.

```powershell
python -B .\audit_prepared_packet.py --packet ..\packet --prior-root ..\..\clmm-revision-local-2026-11-15 --repo C:\Users\henry\Documents\Codex\clmm-replay --source-commit 9ab3901860548d2d87a9afc61e8ce58f950a03ce
python -B .\audit_supplemental_exclusions.py
python -B .\stream_telemetry.py
python -B .\run_analysis.py --final
python -B .\audit_runtime.py --final
python -B .\plot_results.py
python -B .\write_interpretation.py
```

These commands verify or summarize saved artifacts; they do not train or execute the model. Move complete raw roots and preserve relative sibling paths when relocating. The current study is now consulted evidence; exact whole-history exclusion does not create a new vocabulary or establish robustness when the latest real observation is unreliable. Full Titans sequence behavior and language-scale transfer remain outside this component study.
