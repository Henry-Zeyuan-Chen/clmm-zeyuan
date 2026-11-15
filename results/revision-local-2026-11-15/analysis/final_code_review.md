# Independent final implementation/protocol review

Frozen executable commit: `c9f5c153f786f1b82ac383d01c1d18c6c595946b`.

No default-path implementation blocker found in `RevisionConfig`, `RevisionAdapter`, the revised evaluator, or `scripts/revision_local_experiment.py`. The rule matches `clip(d*(H-e*P_xH)+eta*x*y)` with protected inactive modules. The four-rule zero-state write is identical, the EMA arm preserves the old operation order, and readout remains `W0 + alpha*H`; no decoded-overwrite guarantee follows. The denominator uses the declared fixed epsilon. Rule identity participates in the config fingerprint and fast-state schema3 rejects cross-rule states.

The runner's frozen phase transfers all nine dense-trained parent tensors unchanged. Its training phase requires original parent training.step0 and begins the EMA/local-delta pair from identical tensors, using shared training arrays and fixed routing tables. The optimizer, batch, steps, loss and clipping match the declared protocol. Initial checkpoints are serialized at step0 before the separate training record is later updated. Each optimizer step creates zero H. No rule/map/checkpoint selection depends on outcomes. Both phases save source, data, scores, predictions, plans, resource accounting, dedicated-format weights and relevant counters.

The final phase-data helper explicitly excludes exact old90502/new90802 training histories and prior90723 development histories; trained90923 additionally excludes the full prospective frozen90823 histories. The independent raw analyzer reconstructs these choices without importing the runner/evaluator. Final confirmation remains development composition evidence in a fixed familiar codebook; it is not an untouched external benchmark.

## Checks completed by this reviewer

- `revision_formula_final_audit.json`: all48 rule/parent/map constructions passed independent looped formula checks on CPU, exact inactive/input preservation, exact common first write, bitwise legacy EMA equality, unclamped raw projection identity, finite/correct zero-key behavior, strict cross-rule state rejection, exact state roundtrip, exact nine parent tensors and fixed maps, 65536bytes per stream, unchanged global CPU RNG. Exact final revision source SHA is `2ebfaf8dcc7ccd58af4aca2df2fde20a24fa0994cf519276ea6df7bcab94ccbc`.
- `final_source_audit.json`: isolated LF checkout is clean at the frozen commit; all155 tracked Python/project dependency source files checked are byte-identical to their recoverable Git blobs. This is stronger than a matching file hash without recoverable source.
- `smoke_raw_audit.json`: nine existing CPU smoke evaluation directories, 7488 scorevectors. Independently reconstructed prediction/obsolete/MSE arithmetic, saved summaries, exact online plans, rolled/reset source IDs, per-query decay/projection exposure, retained write/decay/projection counts, physical state size and serialization boundaries. This is artifact validation, not an efficacy result.
- `old_revision_raw_audit.json`: independently checked all48 old cases'6144 short revised queries;143 errors, all obsolete, with corresponding stable queries correct. No model rerun was used.

The root coordinator separately reported181 tests passing; this reviewer did not execute that full suite and does not count it as an independently rerun check.

## Remaining evidence and interpretation limits

Full frozen/trained raw phase audits, actual saved initial/final tensor/optimizer audits, paired contrasts and figures remain pending phase completion. Prepared scripts are `analyze_phase.py`, `audit_saved_weights.py`, and `plot_phase.py`, with independent helper modules. They retain every fixed parent/map, calculate the declared four lower.0125 inequalities, preserve all original gates, and show per-parent/map results.

A raw-trace correction can improve revisions while damaging correlated stable keys. Equal persistent state, writes, labels and optimizer steps do not mean equal FLOPs. Main matched training can repair the baseline to a revision ceiling; in that outcome a unique rule-superiority claim should fail even if both methods qualify. The four factorial controls support mechanism contrasts conditional on their fixed geometry, not biological novelty, causal mediation by routing collisions, or a complete Titans result.
