# Shared sleep and Hebbian adapter integration

Completed November 15, 2026 on local branch `adapter-integration`. Frozen experiment source: `a41bb097af3e7eed356d13e24e4a0d5bc6677b14`. This is Zeyuan's Member C foundation for reusable weights, persistent memory and common evaluation. The fetched team repo (`origin/main` at `1e8de19`, only remote branch) contained empty `clmm/sleep` and `clmm/hebbian` directories. These implementations are therefore **explicit reference component adapters**, ready for teammates to replace or extend. They are not their final methods or complete Titans/MAC/attention integration.

## What now works

- `clmm/sleep/adapter.py`: `SleepSession` consumes observed key/value pairs, updates the supplied memory, and maintains the bounded replay policy. Online, repeat-current, continuous and periodic modes share the existing causal scheduler. Export/import resumes model fast state and the policy together, including pending mid-block replay. Functional writes are applied before a session state is committed; reads receive only keys and fast state.
- `clmm/hebbian/adapter.py`: four associative modules, nonlinear key/value encoders, a plastic layer and decoder. The bounded Hebbian update uses observed presynaptic/postsynaptic activity without an online autograd call. A matched analytic-gradient update and dense/top-one access configurations separate the rule from access.
- `clmm/pretrain/adapter_registry.py`: explicit new checkpoint loading and slow-weight forks. Original neural weights are never loaded into an incompatible Hebbian architecture; the literal rank-32 codebook is the shared representation.
- `scripts/adapter_integration_experiment.py`: one evaluation path for all adapters, with acquisition/revision gates, unchanged-fact retention, rolled/reset controls, causal schedules, state handoffs, raw scores and resource records.

## Pilot results

The bounded run completed all eight independently trained associative variants (four systems × two original seeds), two existing neural anchors and eight fixed-slow-weight comparisons. No configuration or seed was selected from these outcomes. Each new trained system passes its short acquisition/revision and K8 D0 admission in both seeds.

At K8, after 64 distractors, **online stable-fact accuracy** averages:

| Independently trained system | Accuracy |
|---|---:|
| Associative gradient, dense access | 6.25% |
| Associative gradient, top-one access | 36.72% |
| Hebbian EMA, dense access | 5.47% |
| Hebbian EMA, top-one access | 21.48% |

The prespecified development-pilot contrast, Hebbian top-one minus dense, is **+16.02 percentage points [8.98,23.44]**. However, access also helps the gradient system by +30.47 points, and trained Hebbian top-one is **15.23 points below** gradient top-one [−22.27,−8.59]. The useful result is selective access under this setup, not a demonstrated advantage of the Hebbian rule. These are 64 paired histories, two fixed initializers and one codebook; intervals resample the same history IDs across both models, 5,000 draws, seed 90604. They are unadjusted development intervals, not population-training uncertainty or a confirmatory final test.

At K16, dense gradient/Hebbian systems fail their own D0 four-target gate (about 44–48% all-correct); top-one variants pass at 63/64 in both seeds. Dense K16 curves remain acquisition/interference diagnostics. The earlier neural baseline's qualification does not transfer to a new architecture.

The exact initializers are retained. Hebbian top-one starts at 24.22% late stable accuracy and ends at 21.48%; the paired change is −2.73 points [−8.20,2.73], so no late-training improvement is established. Gradient top-one improves from 24.61% to 36.72%, +12.11 points [4.29,20.31]. Do not credit all observed recall to outer training when initialization already provides it.

## Separate the mechanisms

Independently trained encoders, plasticity coefficients and rates diverge, so the four trained systems do not isolate only the online update equation. Eight fixed-weight comparisons address that limitation: each gradient/dense endpoint supplies exactly the same slow tensors to all four configurations, with new zero traces and no further adaptation.

Under those fixed slow tensors, K8,D64 online accuracy is 6.25% gradient/dense, 47.27% gradient/top-one, 6.25% Hebbian/dense and 43.75% Hebbian/top-one. The top-one Hebbian-minus-gradient difference is −3.52 points [−7.81,0.78], leaving no established rule advantage in this conditional test. The unchanged gradient/dense fork reproduces **7,296 saved score vectors bit for bit** across the two parents. These are frozen-transfer diagnostics, not independently adapted Hebbian models or a new best-checkpoint search.

Top-one access changes both writing and reading, and protects inactive traces from decay by construction. It does not isolate read routing, write routing, interference and forgetting frequency from one another. Learned eta is about 0.082 for dense Hebbian versus 0.130 for top-one Hebbian, and about 0.072/0.112 for gradient dense/top-one. The frozen-weight forks hold eta and the other slow tensors fixed; even there, active-module exposure and read access differ. Zero **final-query-state** saturation is recorded; that does not prove no transient clipping occurred.

## Sleep integration and matched budgets

The sleep controller pauses arrivals and replays observed pairs through the same `model.write`. For the shared neural memory this performs reconstruction-gradient writes; for the Hebbian adapter it performs its declared local update. The controller does not secretly insert a gradient optimizer into the Hebbian rule.

All modes use online-only K+2 prefill. Online executes K+2+D writes; repeat/continuous/periodic execute K+2+2D. Continuous and periodic share FIFO8, block4, frozen with-replacement draws and identical selected records/order, differing only in placement. Replay never refreshes FIFO age. The same four initial targets form the delay denominator; recent distractors are separate, and delay observations do not revisit an initial key.

Fixed-two-seed secondary periodic-minus-continuous K8,D64 stable contrasts are +1.17 points for gradient/dense, −0.78 gradient/top-one, −0.39 Hebbian/dense, +0.78 Hebbian/top-one and −0.39 for the neural anchor. All corresponding intervals include or touch zero. This pilot establishes no sleep-timing advantage. The full per-condition results, including differences against the lower-budget online anchor, remain in the saved tables; no unfavorable cell was removed.

## Reference choices and source

The Hebbian component uses effective weights `W0 + alpha * H`. With `pre=tanh(key @ encoder)` and `post=tanh(value @ value_encoder)`, active traces update as `H' = clamp((1-eta)*H + eta*pre⊗post, -1, 1)`. Eta is one learned sigmoid scalar, initialized at 0.1. The gradient control replaces the outer product with the negative exact gradient in H of half-sum squared reconstruction error, keeping the same leak and bounds. Outer training differentiates through either analytic update; evaluation freezes all slow tensors.

This follows the fixed-plus-plastic framework and EMA form in [Miconi et al., equations 1–2](https://proceedings.mlr.press/v80/miconi18a/miconi18a.pdf), with explicit component changes: value-clamped postsynaptic activity, bounds and selective access. The router is fixed cosine similarity to literal key prototypes 0/16/32/48; the proposal's learned router is not implemented. Dense access reads/writes all four modules; top-one reads/writes one and leaves others unchanged, including no decay. Encoders/decoders, W0 and alpha are learned offline; trace H is the session state.

All four variants have the same 81,921 slow trainable elements and initial tensor bits for a given seed. Training uses the common train-seed-90502 arrays, 1,200 AdamW updates, batch32, lr0.001, no weight decay and clip norm1. Per model: 178,944 supervised answers and 279,968 observed writes. Eight main runs total **1,431,552 answers and 2,239,744 writes**. The GPU throughput probe and two-step CPU smoke are separate engineering runs, not a ninth main trial. Final checkpoints are fixed endpoints; no best-epoch selection occurs.

## Using the adapter contract

```python
import hashlib
import torch
from clmm.pretrain.adapter_registry import load_adapter_checkpoint
from clmm.sleep import SleepSession

model, provenance = load_adapter_checkpoint(checkpoint_path, torch.device("cpu"))
model.eval().requires_grad_(False)
session = SleepSession(model, batch_size=1, arm="periodic", episode_seeds=[123])
session.prime(torch.tensor([0]), torch.tensor([2]))
session.prime(torch.tensor([1]), torch.tensor([3]))
session.observe(torch.tensor([4]), torch.tensor([5]))
# Save only after all writes returned by an operation have been applied.
slow_hash = hashlib.sha256(open(checkpoint_path, "rb").read()).hexdigest()
torch.save({"slow_checkpoint_sha256": slow_hash,
            "session": session.export_state()}, session_path)
saved = torch.load(session_path, weights_only=True)
assert saved["slow_checkpoint_sha256"] == slow_hash
session = SleepSession.from_state(model, saved["session"])
scores = session.read(torch.tensor([[0, 1]]))
```

`prime` is the common prefill; `observe` accepts later arrivals and scheduled replay. `finish()` flushes a terminal partial periodic block and closes the stream; export/import pauses it. A session payload contains model fast state and bounded policy, not slow weights or a complete event history. Its config fingerprint alone does not authenticate learned slow values. Match the exact slow checkpoint SHA in the run metadata/registry. Saved `example_prefill_session.pt` and `example_midblock_session.pt` are plain session payloads for recorded batch16 periodic K8,D16; keep them with that run's checkpoint.

Replacement adapters must supply functional `initial_state/write/read/detach_state/export_state/import_state`, matching config dimensions/fingerprint, a monotone write counter and plain tensor/container payloads. Writes must not mutate the incoming state or slow model; reads must be pure and receive only keys and fast state. New architectures need their own loader; the existing registry explicitly refuses arbitrary neural-to-Hebbian weight transfer.

## Resources, checks and recovery

Each associative fast trace costs 65,536 tensor bytes per stream, regardless of top-one or dense access. Slow parameter backing is 327,684 bytes, fixed codebook/router backing 21,504 bytes (router1,024), and training-only optimizer tensors 655,392 bytes. These are measured tensor stores, not file sizes. Optimizers are not retained during evaluation. Counts exclude Python/allocator overhead. Top-one changes active updates1 versus4 but the implementation computes vectorized dense candidates; there is no fourfold speed or memory claim. Batch wall times include validation, serialization and controller overhead, not bare-kernel latency.

62 integration tests and an independent analytic/state proof pass. The full artifact audit checks **160,768 main score vectors, 1,107,712 causal writes and 2,960 serialization batches**. Selected inference rechecks cover all 18 main models: 3,616 saved predictions and all score coordinates match exactly on GPU; CPU argmaxes all match with max coordinate error 9.54e−7. These checks are not retraining or full-grid numerical reproduction. Figures are generated from raw outputs and visually inspected.

New Insomnia job13918002 on ins081 was explicitly cancelled after **16m50s**, following successful retrieval/checks. Prior job13914782 is now verified TIMEOUT after1h3s, with its optional portable-GPU check passed and retrieved. No task allocation remains active and no monitoring was scheduled. A tar root-directory-mtime warning is preserved in `transfer-verification.json`; complete GZIP extraction, matching remote/local SHA and the independent file-level audit establish that expected artifacts were retained.

Complete artifacts: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-adapter-integration-2026-11-15`. Compact index: [results/adapter-integration-2026-11-15](../results/adapter-integration-2026-11-15). The adjacent `CLMM_Adapter_Integration_2026-11-15.zip` contains weights, optimizer states, inputs/plans/scores, source snapshots and a complete branch bundle. Its verification JSON and internal manifest identify the final package. Follow `analysis/README.md` to reproduce analysis without training.

## Team decisions and next work

The teammates can now replace the reference backend/controller behind the shared contract. Before a stronger biological or sleep claim, settle learned versus fixed routing, top-k/module counts, the value-clamped Hebbian rule versus alternatives, and the full sequence-model bridge. A useful next diagnostic compares content routing with another one-module allocation rule at matched active-update exposure, and controls how inactive decay is handled. This can distinguish routing content from reduced forgetting frequency.

Member C should keep initializers and matched training budgets visible. The present training horizon is much shorter than D64 evaluation; the absence of a Hebbian late-training gain motivates a separately declared longer-horizon curriculum rather than selecting a favorable frozen fork on these histories. Any new training horizon changes total writes/labels/optimizer opportunities and must expose those budgets. The complete byte-model key-selection failure and paper-style model roadmap remain separate unfinished milestones.
