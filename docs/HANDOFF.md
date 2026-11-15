# CLMM working handoff

## Resume here

**Completed November 15, 2026: shared key/value representation positive control, reusable component weights, persistent state API and bounded streaming replay.** Local branch `kv-representation-control`. Start with [KV_CONTROL.md](KV_CONTROL.md), [MEMBER_C_STATUS.md](MEMBER_C_STATUS.md) and [results/kv-control-2026-11-14](../results/kv-control-2026-11-14). The task began November 14; artifact paths retain that start date. Verify Git status before editing. No task process or GPU allocation remains active; no future monitoring was scheduled.

Zeyuan is Member C: shared pretraining feasibility, reusable baseline weights and common evaluation. Member A owns sleep consolidation; Member B owns selective Hebbian updates/routing. The aim is to compare acquisition, genuine revisions and unchanged-fact retention. Sleep timing needs matched replay opportunities; the Hebbian rule and selective routing need separate ablations. A failing acquisition baseline cannot support a retention claim.

## Current evidence and decision

- Two simplified nonlinear memory initializers, seeds 90301/90305 with inner step 0.5, pass all three short acquisition/revision gates at 128/128, then pass fresh confirmation at the same score. Initial checkpoints **already** pass short accuracy; outer training lowers reconstruction MSE. This is not proof that training caused acquisition accuracy.
- Orthogonal K8/16/32 × D0/16/64 pressure is 100% on the fixed four targets in every ordinary arm for both initializers. It is ceiling limited; no measured periodic-replay accuracy advantage.
- A separately specified, single correlated-key linear calibration produces interference. K8 D0 four-target all-correct is 31/32. Prespecified K8,D64 stable periodic-minus-continuous is +4.69 percentage points, paired-episode 95% interval [0,10.94]. It is inconclusive, with one codebook and no training. Geometry and state size both changed relative to the orthogonal control. Larger-K D0 failures limit retention interpretation.
- The complete byte model remains unqualified for fresh two-key assignment transfer. The typed component is useful shared scaffolding; it does not resolve language encoding/generalization.

Next useful experiment: **Member C integrates a small declared correlated/capacity-limited setting into the nonlinear adapter, retaining acquisition and per-K D0 gates plus untrained anchors. Freeze a setting with real accuracy headroom before teammate efficacy comparisons.** Do not repeat an unlimited byte overfit or orthogonal ceiling sweep. Member A/B can already implement against the common API and scheduler independently.

## Code, artifacts and recovery

- Frozen main model/data/evaluation source: `ff62df0f85ff7cbf62f3209755ea066959aa7085`.
- Post-run bounded scheduler/integration source: `c25c047`. `clmm/pretrain/kv_scheduler.py` stores literal FIFO/snapshot records, not indices requiring complete history. All saved plans have exact event parity. `scripts/kv_streaming_demo.py` verifies both disk-backed fast state and scheduler resume after a partial block. New source is included in the final bundle; checking out ff62 alone will not supply it.
- Local output root: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-kv-control-2026-11-14`.
- Work/launch scripts: `C:/Users/henry/Documents/Codex/2026-10-22/i/work/clmm-kv-control`.
- Final weights: `s05-90301/checkpoint.pt` and `s05-90305/checkpoint.pt` under the output root. Initial weights, optimizer state and original commands are retained. `checkpoint_admission.json` states exact supported abilities and exclusions.
- Archive: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/CLMM_KV_Control_2026-11-14.zip`; adjacent `.verification.json` records observed archive verification and final handoff commit. `clmm-kv-control.bundle` inside the archive contains complete branch history and the unchanged baseline tag. `MANIFEST.sha256` covers archived files. The compact Git index omits full raw data and weights; raw auditors need the archive/extracted output root.
- Independent main checks: 47,808 saved score vectors, 304,704 causal writes and 918 serialization batches. Selected CPU reload: 540 answers, every argmax matches, max score error 3.58e−7. Same-environment GPU reload: 944 answers and all coordinates exact. Separate dense audit: 9,792 score vectors exactly replayed.
- Final focused tests: 329 passed. Separate dense tests: 18 passed. Live policy: 17,424 saved plans / 405,222 write events exact, including main, smoke and dense records; 3,900 JSON resumes. Each learned initializer passes the one-history all-arm streaming integration demo. These checks do not claim retraining reproduction.

The first launcher failed while serializing a NumPy boolean after Delta and the first neural run completed. `pipeline.attempt1.*`, partial `selection.attempt1.json` and original launcher source are preserved. The corrected launcher validates source, successful metadata and exact commands before reusing completed runs; no retraining or protocol change occurred. The independent storage-accounting audit's first assertion failure is also retained; final physical counts include live scan backing storage and distinguish compact exports.

## Compute status

Insomnia allocation **13914162, ins093, RTX A6000**, one GPU/four CPUs/16G RAM, one-hour ceiling, was explicitly released after **15m55s** (0.265 allocated GPU-hours). Pipeline and selected inference recheck completed. `cluster-sacct.txt`, `cluster-squeue-final.txt`, `gpu-driver.txt` and `cluster-observations.json` record closure. Remote files remain at `/insomnia001/depts/edu/users/zc2819/clmm/kv-control-2026-11-14`. No renewal, desktop helper selection change or unrelated-job cancellation occurred. The supplemental calibration and final integration checks used local CPU.

## Earlier milestones to preserve

- [SHARED_PRETRAIN.md](SHARED_PRETRAIN.md): two compact complete byte-model parents, seeds 68101/68105, modest language-context memory effect. Parent weights remain at sibling output `clmm-shared-pretrain-2026-11-14/lm-SEED/checkpoint.pt`.
- [PAIRED_ACQUISITION.md](PAIRED_ACQUISITION.md): single-fact recall/revision work; two-key selection fails.
- [FIXED_SET_GENERALIZATION.md](FIXED_SET_GENERALIZATION.md): all four models fit 32/32 fixed contexts; training-disjoint fresh follow-ups score only 1/128 and 3/128. Frozen source `5bccffd`, previous handoff `ea967d7`.
- Earlier structured-memory work in `scripts/research_memory.py`, `scripts/memory_pressure_probe.py` and [MEMORY_PRESSURE.md](MEMORY_PRESSURE.md) uses different representations/settings. Do not relabel it as complete sequence-model weights or conflate it with this typed control.

## Team choices and boundaries

Record whether the typed component suite is the main mechanism test while the complete sequence model is developed separately. Settle the shared pressure representation/capacity and whether the main comparison matches write opportunities, memory bytes or actual compute; expose remaining differences. Frozen block snapshots may replay an old value if a genuine revision arrives within that block. The present pressure delay revises no target, so a future in-block revision study must declare whether to retain or invalidate stale snapshots. `finish()` is terminal; use export/import, not finish, merely to pause a session.

No Google Doc edits, teammate messages, GitHub push, PR or merge were performed. Local work and commits are ready for review. Do not move/delete `baseline-v0.6.0`, whose target remains `35a549f137548d48aecc26e33c6f8a34a6efc97c`. Append consequential actions to `docs/EXPERIMENT_LOG.jsonl`; reconcile this handoff against real files and scheduler state on resumption.
