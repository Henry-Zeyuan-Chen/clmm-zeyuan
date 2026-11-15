# Bounded streaming scheduler review

Reviewed `WORK/kv_scheduler.py`, its tests, the identical repository copy, and `docs/KV_CONTROL.md` on 2026-11-15. Scheduler SHA256: `8c082b77871adfa08264507f5e37f9b78cded0d5c3b4798f599801d5c06923dd`.

**Decision:** no material scheduler logic or causal-storage blocker identified. All four documentation integration findings were corrected and rechecked against the final documentation. No outstanding review guidance remains. This review did not change repository files, allocate a GPU, or retrain/run a model.

## Storage and semantics

Persistent policy state retains at most eight literal `(source,key,value)` FIFO records and four frozen selections. It never retains an episode/history or resolves replay values through external source-index lookups. Online and repeat retain neither FIFO nor snapshot. The temporary bounded lookup and emitted event lists add transient overhead; the eight/four claim describes retained records, not an exact count of Python objects or constant serialized bytes. Source counters grow logarithmically with stream length.

Only real arrivals refresh FIFO age. A snapshot is sampled with replacement from current latest-per-key records before the block's first arrival, remains immutable through that block, and is discarded afterward. A genuine within-block revision therefore leaves an old value in the frozen snapshot by design; it may be replayed after the new observation. The current experiment avoids such delay-time label conflicts. Future method integrations should retain or explicitly change this policy rather than silently rewriting snapshot labels.

JSON state carries the literal records, phase/counters, configuration fingerprint and PCG64 state, with bounded schema validation and defensive copies. Mid-block continuation is supported. `finish()` flushes only the arrived prefix of a partial periodic block and then closes the scheduler; a paused session should export state without finishing. The live implementation pre-draws a full block even for a terminal partial block. Its emitted events match the frozen plan; no final RNG-state equality with an offline partial-block generator is claimed.

An independent literal-record probe passed: a mid-block revision preserved the old frozen value, the FIFO retained the new value, JSON resume reproduced all remaining writes, and the next block used then-current latest values. Evidence is `streaming_probe.json`. The reviewed tests cover offline-event parity, partial blocks, mid-block JSON resume, 10,000-arrival record bounds, invalid-state rejection, and atomic rejection of invalid arrivals. The implementer owns the final test rerun.

## Parity and empirical scope

The coordinator-produced `streaming_parity_audit.json` reports exact equality for 17,424 saved plans and 405,222 writes across main, initial, smoke and dense artifacts, including 3,900 resumed policies. Its scheduler hash matches the reviewed source. I inspected that record; I did not rerun the coordinator's full parity job. This is a **post-experiment event-equivalence check**, not a model rerun or new efficacy experiment. It does not retroactively replace the frozen evaluator implementation used for the saved model runs.

The scientific claims in `KV_CONTROL.md` match the independently audited evidence: initial accuracy was already at ceiling, outer training improves MSE, the orthogonal grid has no replay-accuracy headroom, the dense primary interval includes zero, and the full byte-model transfer gap remains unresolved. Budget, fixed-codebook and higher-K acquisition caveats are retained.

## Documentation follow-up

1. **Resolved and rechecked:** resume now explicitly requires the slow checkpoint, adaptive fast payload **and scheduler JSON payload**, captured after every returned event has been applied. Saving only fast state would lose the FIFO, pending frozen selections and RNG continuation.
2. **Resolved and rechecked:** the snippet now writes and reloads `policy_state_path`; it originally only JSON-roundtripped an in-memory payload while writing only fast state to disk.
3. **Resolved and rechecked:** the admission measure is now named **K2 all-correct episode accuracy**, preventing a substitution of per-query accuracy.
4. **Resolved and rechecked:** the final documentation explicitly distinguishes terminal `finish()` from pause/export, scopes partial-block parity to emitted events rather than final RNG-state identity, and states that frozen snapshots may replay an old value after a within-block revision. Future revision studies must declare whether those stale snapshots persist or are invalidated.

The final archive/bundle should contain the later scheduler/integration commit as well as frozen `ff62df0`, because that frozen model/evaluator commit predates the live scheduler import and test paths. The coordinator owns packaging, combined-state integration demonstration, and scheduler-test completion.
