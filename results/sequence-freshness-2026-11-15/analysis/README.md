# Independent sequence replay evidence review

All eight 839,248-parameter terminal models fail reusable two-key acquisition: 0/24 required gates, 0/128 joint-four swap success per model, and 0/64 complete revision-pair success under every normal arm. Both co-primary bounded-minus-literal effects are zero with intervals crossing zero. Integration and saved-artifact integrity pass; no retention/sleep efficacy admission follows.

Read [result_interpretation.md](result_interpretation.md) for the complete negative conclusion and limitations. The coordinator's CPU check preserves all 1,200 sampled predicted bytes but fails its original numerical tolerance in ten rows; this limitation is independently verified in [cpu_recheck_artifact_review.json](cpu_recheck_artifact_review.json).

## Final artifacts

- `audited/results_audit.json`: all 52,224 main score vectors, exact raw-prefix/label binding, independent metrics, paired fixed-case primary estimates and budgets.
- `audited/acquisition_gates.csv`, `counterfactual_metrics.csv`, `query_behavior.csv`, `memory_dependence.csv`, `revision_contrasts.csv`: all per-case and secondary development diagnostics.
- `final_run_provenance.json`: all twelve main checkpoints, terminal optimizers, parent identity, tensor shape/count/finite checks and 48 exact source Git blobs.
- `resources_and_midpoints_review.json`: eight recovery-only midpoint checks, 16,000 main updates/256,000 forks/29,169,664 training bytes, evaluation/resource coverage and runtime gate.
- `engineering_audited/` and `engineering_run_provenance.json`: separately scoped twenty-step engineering evidence, never pooled into main effects.
- `figures/`: three stable PNG/PDF/SVG numerical figures; source tables are preserved. `figure_review.json` records visual QA and hashes.
- `claim_evidence_map.csv`: exact supported and unsupported claims.
- `future_exclusion_signatures.json` and `future_exclusion_manifest.json`: future-only85,123-history union, including actually presented contexts; launch packets remain immutable.
- `final_analysis_inventory.json` and `.sha256`: final artifact inventory and evidence counts.

## Source, data and exclusion provenance

Executed source is `95da59531ccf5d5f3fe2729553e6d980da598352`. `packet-final` and `engineering/packet-final` preserve exact source snapshots. `source_amendment_review.json` proves original/main-final training tensors, semantic worlds, plans, exact samples and parents unchanged after the performance/exclusion amendment. `final_packet_review.json`, `final_saved_execution_review.json`, `final_packet_provenance_review.json`, `final_engineering_packet_review.json` and `final_exclusion_review.json` preserve prelaunch audits.

The original structural collector's JSONL/raw-prefix omission is documented. The pre-execution81,326 union is immutable. Work-directory CPU smoke adds21 signatures; `work_smoke_regeneration_review.json` proves actual frozen-producer regeneration under81,347 produces exactly identical fixed evaluation episodes. The original CPU smoke packet/run are archived at `../engineering-cpu`, with66 file hashes. The failed `packet-executed` preparations remain preserved; their missing-filename error is not a model outcome.

`initial_protocol_review.md`, `wrapper_review.json` and helper/selfcheck records preserve the prospective review. Earlier `smoke_*` artifacts concern the separate eight-step CPU engineering run. They are not main efficacy evidence.

## Reproduction

Use Python with NumPy, Matplotlib for plotting, and PyTorch only for CPU tensor inspection. Main saved-data/statistics audits import no model code. The bridge review imports policy modules only; synthetic selfchecks use producer data functions to create fixtures.

```text
python audit_packet.py --training-worlds ../packet-final/training_plan.json.gz --training-arrays ../packet-final/training.npz --evaluation-worlds ../packet-final/episodes.json --exclusions ../packet-final/excluded_signatures.json
python audit_saved_plans.py --packet ../packet-final
python audit_results.py --packet ../packet-final --case <four-initial-and-eight-terminal-case-folders> --out audited --require-complete
python audit_provenance.py --packet ../packet-final --repo <frozen-LF-source-checkout> --case <same-case-folders> --out final_run_provenance.json
python secondary_analysis.py --audited audited
python analyze_behavior.py
python plot_results.py --audited audited --out figures
python audit_resources.py
python audit_cpu_recheck.py
python build_future_exclusions.py
python finalize_analysis.py
```

Whole-episode paired bootstraps use seed91304 and5,000 draws. Co-primary intervals are97.5%; secondary intervals are nominal95%. These condition on fixed checkpoints and cannot measure general training variability. The reviewer performed no model construction, training, inference, GPU or SSH work during this review. CPU checkpoint/optimizer tensors were inspected without building a model. Scheduler closure and the actual CPU inference were performed separately by the coordinator.
