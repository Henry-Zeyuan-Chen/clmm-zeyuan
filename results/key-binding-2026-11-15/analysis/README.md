# Independent analysis artifacts

The scientific source is frozen at `5e8c61829ff85ecea3c9314d4ccd8dfef3268b68`. The analysis performs no model construction, inference, GPU work or SSH. Tensor loading inspects saved tensors only. Actual CPU inference was performed separately by the coordinator and audited from its saved artifacts.

Start with `independent_results_review.md` and `claim_evidence_map.csv`. All four terminal models fail all six acquisition admission groups; both co-primary joint-four endpoints are at zero. Degenerate bootstrap intervals are not equivalence evidence.

## Evidence chain

| Artifact | Scope |
| --- | --- |
| `prepared_packet_audit.json` | Independent exact regeneration of all three packets, bytes, labels, exclusions and budgets |
| `frozen_provenance_audit.json` | All 50 source files per packet match frozen Git blobs; parent tensors and protected user document verified |
| `execution_amendment_review.json` | Original runtime-only failure and prospective two-GPU execution amendment |
| `main_output_audit.json` | All 81,920 full score vectors; predictions, metrics, parity, batch indexing, budgets, gates and primary intervals |
| `main_tensor_audit.json` | Four initial/four terminal/four recovery checkpoints and eight optimizer states |
| `engineering_validity_audit.json` | Four 640-vector GPU calibrations and one 80-vector CPU smoke; validity only |
| `cpu_proof_artifact_audit.json` | Original GPU batches and 2,560 saved CPU recomputations; unchanged tolerances and prelaunch script hash |
| `behavior_controls.json` | Requested-key changes, observed-value selection, swaps, retrieval ablation and descriptive training windows |
| `future_exclusion_signatures.json` / `future_exclusion_manifest.json` | 100,526 histories from actual presented bytes plus the supplied prior union |
| `../provenance/prior_exclusion_signatures.json` | Exact preserved prior exclusion input; packet manifests retain its original path/hash |
| `figures/figure_manifest.json` | Three visually checked numerical figures, PNG/PDF/SVG exports and source CSVs |
| `analysis_inventory.json` | Final analysis-only file hashes; excludes itself and Python bytecode caches |

Each figure comes from `main_output_audit.json` or saved training logs through `plot_binding_results.py`. The primary figure zooms the acquisition panels to 0–1% and explicitly states that the 80% admission threshold lies outside this range. All raw score vectors remain in the run directories.

## Reproduction entry points

Scripts accepting command-line paths can be rerun against the saved packet and run tree. `audit_binding_packet.py` checks prepared packet regeneration. `audit_binding_outputs.py` recomputes the saved-output audit and prespecified paired-world bootstrap. `audit_binding_tensors.py` inspects checkpoints and Adam states without constructing models. `plot_binding_results.py` renders figures from a passed main audit. `audit_cpu_proof_artifacts.py` checks saved proof vectors without inference. `finalize_binding_review.py` reconstructs future exclusions from actually presented raw bytes; `write_binding_report.py` writes this readable report and visual QA metadata.

Some provenance and packet manifests retain original absolute Windows paths. The exact prior exclusion input is preserved under the package provenance directory; on relocation, resolve manifest inputs by their recorded SHA256 and corresponding packaged engineering/CPU packets. Original launch files are immutable.

The fixed codec is 2,978 bytes of external preprocessing state, separate from model parameter count. Passing saved-vector and numerical validity does not qualify retention, sleep or an atomic replay adapter. No further model training was used to select these results.
