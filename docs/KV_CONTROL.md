# Shared key/value memory control

Completed November 15, 2026 on local branch `kv-representation-control`. Zeyuan's Zeyuan deliverable is a common representation, reusable memory initializers and a persistent evaluation interface for the sleep and Hebbian tracks. This is a **memory-component positive control**. The complete byte-model association-transfer failure in [FIXED_SET_GENERALIZATION.md](FIXED_SET_GENERALIZATION.md) remains unresolved.

## Results and decision

| Check | Observed result | What it supports |
|---|---|---|
| Nonlinear memory, initializers 90301/90305 | Both score 128/128 all-correct in K2 acquisition, K4 acquisition and genuine K2 revision; repeated on fresh confirmation histories | A usable closed-vocabulary component baseline for both tracks |
| Before versus after outer training | Initial models already score 128/128 in all three short cells; training reduces read-coordinate MSE | Improved reconstruction precision; training did not produce the observed short-task accuracy |
| Orthogonal pressure, K8/16/32, D0/16/64 | Both models are 100% on the same four targets in online, repeat, continuous and periodic arms | Correct state/evaluation plumbing, but no accuracy headroom for comparing replay methods |
| Supplemental correlated-key linear control | Short gates pass; K8 D0 four-target all-correct is 31/32 | An interference-producing instrument with adequate initial K8 acquisition |
| Its prespecified K8,D64 stable-target comparison | Periodic 20/64 versus continuous 17/64; difference +4.69 percentage points, paired-episode 95% bootstrap interval [0, 10.94] points | Inconclusive timing comparison for one fixed codebook; no established periodic-replay advantage |

The last comparison was specified **after seeing the orthogonal ceiling**, before running the supplemental configuration. It changes both key geometry and fast-matrix dimension: 64 normalized Gaussian keys in 32 dimensions, rather than orthogonal 64-dimensional keys. It is an untrained linear instrument, not a trained nonlinear or Hebbian result. Its interval resamples 32 paired episodes, 5,000 draws with seed 90404; it does not estimate variability across codebooks. Larger-K cells with weaker D0 acquisition cannot isolate delay retention. Immediate repeated writes are idempotent for this unit-key, step-one linear rule, so matched write opportunities do not mean equal numbers of nonzero gradient updates. No configuration or seed search followed this result.

## Common representation and models

`clmm/pretrain/kv_control.py` exposes 64 key IDs and 16 value IDs. Keys are fixed unit one-hot vectors in 64 dimensions; values occupy the first 16 coordinates. The decoder returns those 16 raw coordinates. There is no learned classifier, byte parser or transformer encoder. These changes are broader than a tokenizer ablation, so this does not establish that tokenization caused the earlier failure.

- **Delta:** zero-initialized linear matrix, half-sum squared-error gradient write with step one and no decay. Orthogonal keys guarantee exact latest-value retention by construction. Use it for plumbing and update checks.
- **Neural:** vendored `NeuralMemory` and a 64→128→64 `MemoryMLP`, one head, chunk one. Identity Q/K/V, no normalizations, momentum, memory residual or retrieval gate. Half-sum squared-error inner loss, step 0.5, decay logit −80 (no effective float32 decay). Only the initial memory matrices are trained offline; the last initial matrix starts at zero. No vendored source changes.

The neural adapter registers 32,898 slow parameter elements; 16,384 are trainable offline. These simplified settings intentionally make the memory mechanism inspectable. They are not a claim of paper-faithful Titans reproduction.

Both training invocations use 1,200 AdamW updates, batch 32, learning rate 0.001, no weight decay and gradient clipping at one. The outer objective is mean squared error of the read coordinates against the one-hot answer. Per initializer: **177,728 supervised answers and 278,112 observed writes**. Actual totals are 355,456 answers and 556,224 writes. Evaluation directories also save the planned training arrays for exclusions; those are not extra training invocations.

Training uses K2/4/8, real revisions in half the batches and D0/4 outside-key distractors. All 64 keys and all 1,024 atomic bindings appear in training. Evaluation excludes exact ordered training histories, but local subsets and atomic bindings can recur. The supported generalization is to **fresh complete histories/compositions**, not unseen atomic bindings, new keys or language.

## Qualification and evaluation

The first candidate, inner step 0.5, passed both original initializers and was selected; the conditional step-one alternative was never run. Final checkpoints are fixed endpoints, not the best observed checkpoint. Each of three admission cells has 128 episodes. K2 acquisition and K2 revision require ≥90% all-correct, K4 requires ≥80%, and the lower 95% bound of normal-minus-rolled K2 all-correct episode accuracy must exceed zero. Cyclic rolled-state controls use eight blocks of 16 episodes for the bootstrap, preserving donor dependence.

Development seed 90303 passes at 100% in both models. Fresh confirmation seed 90313 also passes at 100% in every cell. Confirmation normal-minus-rolled differences are 0.9922 [0.9766,1] and 0.9844 [0.9609,1]. These are development checks of fixed checkpoints, not an untouched final test or uncertainty over training procedures. At a ceiling, a degenerate empirical interval does not imply zero population error.

Pressure seed 90323 supplies 32 episodes per K,D cell, shared across models and arms. Prefill writes K facts and revisions of positions 2/3. Two stable and two revised targets have distinct final values; delay writes never revisit any initial key. Save/load/import occurs at the prefill boundary. The same four targets form the retention denominator at every delay; the optional recent distractor is reported separately. Stable versus revised targets differ in age and FIFO residency, so compare schedules **within** each stratum.

| Arm | Writes per episode | Replay policy |
|---|---|---|
| Online | K+2+D | Lower-budget anchor |
| Repeat | K+2+2D | Repeat the current arrival immediately |
| Continuous | K+2+2D | One frozen-snapshot sample after each delay arrival |
| Periodic | K+2+2D | The same four samples in the same order after each four-arrival block |
| Boundary reset | Executes prefill then discards its state | Separate diagnostic, not equal effective-state work |

Continuous and periodic sample with replacement from a latest-per-key FIFO of eight pairs at the **start** of each four-arrival block. Real repeated observations/revisions refresh FIFO age. Replay does not refresh age. Both use identical write rules and gates. This measures a specific buffer and schedule; four draws need not be four distinct records. It does not implement Member A's complete proposed sleep mechanism.

The original `kv_streams.replay_plan` keeps evaluator traces and source indices for reproducibility. The post-experiment `clmm/pretrain/kv_scheduler.py` supplies a deployable `StreamingReplay` with literal bounded records: at most eight FIFO pairs and four snapshot selections, no complete history. Online/repeat retain no replay buffer. JSON export/resume includes counters and PCG64 state, including mid-block resumes. Its emitted writes are checked against the frozen saved plans; this new implementation does not retroactively change the model runs. `finish()` closes the stream; use export/import rather than finish to pause a session. For a terminal partial block, emitted events match the offline prefix; final RNG-state identity is not claimed. A frozen snapshot can contain an old value if that key is revised within the block. Current pressure delays revisit no initial target; future revision studies must declare whether stale snapshots persist or are invalidated.

## Shared adapter and persistence contract

All method tracks should implement these operations:

```python
state = model.initial_state(batch)             # empty session, learned initializer
state = model.write(keys, values, state)        # int64 tensors [B], observed facts only
scores = model.read(query_keys, state)          # int64 [B,Q] -> raw scores [B,Q,16]
state = model.detach_state(state)
payload = model.export_state(state)
state = model.import_state(payload)
```

`read` must be pure and receive only keys and fast state. Exported state has a validated schema, method/config fingerprint, batch size, write count, weights and auxiliary tensors. It contains no history, query labels or replay plan. A replay-session resume requires the corresponding slow checkpoint, its fast payload **and** the scheduler payload. Save them after applying every event returned by a `prime`, `step` or `finish` call. Method states and byte-model checkpoints are not interchangeable. A new Hebbian architecture must supply its own explicit state/checkpoint loader; matching dimensions alone is insufficient.

Example using the saved initializer and a real mid-block scheduler checkpoint:

```python
import json
import torch
from scripts.kv_representation_control import load_checkpoint
from clmm.pretrain.kv_scheduler import StreamingReplay

model, provenance = load_checkpoint(checkpoint_path, torch.device("cpu"))
model.eval()
model.requires_grad_(False)                     # freeze slow weights during evaluation
state = model.initial_state(1)
policy = StreamingReplay("periodic", seed=123)

def apply(event):
    global state
    with torch.no_grad():
        state = model.write(torch.tensor([event["key"]]),
                            torch.tensor([event["value"]]), state)

for key, value in [(0, 2), (1, 3), (0, 4)]:
    apply(policy.prime(key, value))             # common online-only prefill
for event in policy.step(2, 5):
    apply(event)
torch.save(model.export_state(state), fast_state_path)
with open(policy_state_path, "w") as handle:
    json.dump(policy.export_state(), handle)
state = model.import_state(torch.load(fast_state_path, weights_only=True))
with open(policy_state_path) as handle:
    policy = StreamingReplay.from_state(json.load(handle))
for key, value in [(3, 6), (4, 7), (5, 8)]:
    for event in policy.step(key, value):
        apply(event)
for event in policy.finish():
    apply(event)
with torch.no_grad():
    predictions = model.read(torch.tensor([[0, 1]]), state).argmax(-1)
```

The saved `pressure-s05-SEED/evaluation/example_boundary_state.pt` and `example_final_state.pt` are batch-16 examples for online K8,D16 and its recorded episodes. They are not the single-stream state from this snippet.

Actual unique adaptive tensor backing storage per stream is 16 KiB for Delta, 128 KiB after a live neural write and 64 KiB after compact neural export/import. A live scan view retains unused backing storage; serialization compacts it without changing predictions or continuation. Fixed codebooks add 20 KiB per adapter. Counts exclude Python overhead, allocator reservations and training graphs. Report adaptive tensor bytes, slow parameters and replay pairs separately. Dense calibration has a 4 KiB fast matrix, a different resource setting.

## Evidence, commands and recovery

Frozen main model/data/evaluator commit: `ff62df0f85ff7cbf62f3209755ea066959aa7085`. Original baseline tag remains `35a549f137548d48aecc26e33c6f8a34a6efc97c`. Small result index: [results/kv-control-2026-11-14](../results/kv-control-2026-11-14). Full artifact root:

`C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-kv-control-2026-11-14`

Final reusable nonlinear weights are `s05-90301/checkpoint.pt` and `s05-90305/checkpoint.pt`; initial checkpoints and optimizer states are retained. `checkpoint_admission.json` records hashes and qualifications. Full data, episode definitions, replay plans, score vectors, source snapshots, environment and exact commands are retained. The adjacent `CLMM_KV_Control_2026-11-14.zip` includes a complete Git bundle and these raw artifacts; the Git index contains compact summaries rather than weights/raw traces.

Independent checks cover 47,808 main score vectors, 304,704 planned writes and 918 serialization batches. A separately differentiated nonlinear update matches current-state half-SSE SGD exactly. A CPU reload reproduces all 540 selected answers (maximum raw-score difference 3.58e−7); the same-environment GPU reload reproduces all 944 selected answers/score vectors exactly. These are artifact/inference checks, not retraining. Dense calibration separately replays all 9,792 score vectors exactly, 94,656 planned writes and 162 round trips. Smoke runs are recorded separately and do not count as efficacy evidence.

Run focused tests from the repository:

```powershell
python -m pytest tests/test_clmm_kv_control.py tests/test_clmm_kv_streams.py tests/test_clmm_kv_scheduler.py -q
python scripts/kv_streaming_demo.py --checkpoint <saved-checkpoint.pt> --out <new-output-directory>
```

The focused suite passes 329 tests; the separate dense adapter passes 18. The live scheduler matches all 17,424 saved plans (405,222 emitted writes, including smoke and dense records), with 3,900 mid-block JSON resumes. Both learned initializers pass the disk-backed integration demo in every ordinary arm with exactly matching live/offline scores. This is a compatibility check on one fixed history, not a new efficacy estimate.

For source/run reproduction, use a clean clone of the bundled branch, check out `ff62df0`, follow SETUP.md and relocate the exact commands in `pipeline.json`. Use a fresh output directory. The frozen run environment is Python 3.11.5 / PyTorch 2.11.0+cu128, RTX A6000. The supplemental dense script and tests are preserved in `launch/`; its own hash supplements the frozen evaluator commit. Analysis instructions and scripts live under `analysis/` in the full artifact root.

Insomnia allocation 13914162 on ins093 was released after 15m55s (about 0.265 allocated GPU-hours). No task job remains active. The first launcher attempt failed on a NumPy-boolean JSON serialization after two model runs had succeeded; the corrected launcher reused those completed runs after validating source and commands. `pipeline.attempt1.*`, partial `selection.attempt1.json` and original launcher source are retained. This was an operational failure, not a dropped model result.

## Next shared milestone and team choices

1. **Zeyuan:** integrate correlated/capacity-limited representations into the nonlinear adapter at a small declared configuration budget. Keep short acquisition and per-K D0 gates; freeze one useful pressure setting before evaluating teammate methods. The current dense control identifies a feasible interference regime but does not select a trained neural baseline. Include an untrained initializer anchor so outer-training value remains measurable.
2. **Member A:** connect the sleep update to the adapter and live scheduler; compare against continuous replay using identical observed histories, buffer, sampled records, extra writes and gating. Record any change from this snapshot schedule before interpreting results.
3. **Member B:** connect the Hebbian state/update behind the same API. Evaluate the update rule alone and selective routing separately. Log actual tensor storage, number of updates and any extra slow training rather than calling unlike resources matched.
4. **Team decision:** adopt this typed component suite as the common mechanism test while keeping the full-sequence baseline a separate milestone; settle the pressure representation/capacity, and whether the main comparison matches write opportunities, memory bytes or measured compute. Report whichever budgets remain unequal. These are decisions to record, not reasons to block adapter work.

The full-model track still needs a stronger encoder/association curriculum and successful fresh two-key selection before claiming language-level retention or sleep/Hebbian gains. The class interim report can now cite real acquisition, persistence and interference experiments with these limits; it need not be written before the method integration.
