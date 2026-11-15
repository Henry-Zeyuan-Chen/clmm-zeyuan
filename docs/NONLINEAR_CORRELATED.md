# Nonlinear correlated-key shared baseline

November 15, 2026. Branch `nonlinear-correlated-control`; frozen experiment source `b5f27ff4738ded4832538573d454d2461786d38b`, parent `ffaffed`. This continues Zeyuan's Member C work: reusable component initializers and a common state/evaluation contract for the sleep and Hebbian tracks. It does not qualify the complete byte model for language or fresh two-key transfer.

## Calibration and scope

The first declared candidate, **rank 32**, passes both initializer seeds (90511/90515). Each trained model scores 128/128 all-correct on K2 acquisition, K4 acquisition and genuine K2 revision, plus 32/32 K8 four-target episodes before the calibration delay. Stable-fact accuracy after 64 distractors is 39.06% and 35.94%, giving acquisition plus measurable interference. Rank 16 and rank 8 were not run because the ordered stopping rule selected rank 32.

Fresh confirmation, seed 90513, repeats 128/128 on each short cell and 128/128 four-target K8 D0 episodes for both models. Online stable accuracy at D64 is 40.23% and 37.50%. Selection uses only acquisition and online-only pressure, with no replay outcomes available. `selection-before-confirmation.json` and `comparison-freeze.json` preserve the decisions. Confirmation is development evidence, not a search over additional settings.

The prespecified fresh-history primary shows **no established periodic advantage**: trained periodic-minus-continuous stable accuracy at K8,D64 is −0.59 percentage points, paired 95% interval [−2.15,+0.98]. The two individual checkpoints give −1.56 points [−3.91,+0.39] and +0.39 points [−1.56,+2.34]. This is an informative non-ceiling comparison, not evidence of equivalence or a result about every sleep mechanism.

| K8,D64 stable accuracy | Trained mean | Exact untrained mean |
|---|---:|---:|
| Online, fewer writes | 39.06% | 29.10% |
| Repeat current, matched extra opportunities | 40.43% | 14.45% |
| Continuous replay | 35.35% | 21.29% |
| Periodic replay | 34.77% | 21.88% |

Means average two fixed initializers on the same 128 histories (two stable queries each); seed×history rows are not treated as independent units. Training improves online stable accuracy by 9.96 points [5.08,14.84], while late online coordinate MSE worsens from 0.0560 to 0.0703. Both initializers already pass short accuracy, so training's demonstrated value here is delayed classification, not producing short-task acquisition. Report accuracy and reconstruction error separately.

Secondary comparisons show continuous and periodic replay below the lower-budget online arm by 3.71 and 4.30 points respectively. Repeat-current is 1.37 points above online. These fixed-history secondary intervals do not include zero, but they are not multiplicity-adjusted primary claims, and online has fewer writes. The simple eight-record FIFO/snapshot replay rule therefore supplies no positive sleep-method result. All 256 stable queries are evicted at the final FIFO; 102 had earlier replay. This motivates a later coverage/policy ablation rather than more timing tuning on the same histories.

Actual comparison D0 acquisition admits K8 (128/128 both) and K16 (125/128,124/128 four-target all-correct). K32 fails (75/128,77/128), so its curves remain acquisition/interference diagnostics. No setting or checkpoint was reselected after replay outcomes.

## What changed and what stayed fixed

`clmm/pretrain/kv_correlated.py` reuses the previous nonlinear adapter's engine, current-state gradient write and read implementation. It replaces only the fixed key codebook: draw one 64×64 Gaussian matrix using a private CPU generator, take the first r columns, normalize each row and pad to 64 dimensions. The declared ranks are nested slices of that one draw before normalization; no codebook-seed search occurs. The codebook seed is 90501. Rank 32 has SHA256 `701cbfb55a6be11487bb02b21a2392374a361bd2b4e003225e4b72b0acaceb0a`, covering shape, dtype and canonical tensor bytes.

The nonlinear memory remains 64→128→64, with 32,898 registered parameter elements and 16,384 offline-trainable initial-memory elements. Identity Q/K/V, one head, chunk one, no norms, momentum, memory residual or read gate; inner half-sum squared-error step 0.5 and decay logit −80. The fixed 16-way value decoder is unchanged. Private codebook generation leaves the same per-seed MLP initialization/global RNG as the orthogonal baseline and other ranks. All candidates have the same registered model geometry; the key subspace changes effective representational capacity and correlation.

Each actual training run uses 1,200 AdamW updates, batch 32, lr 0.001, no weight decay, clip norm one and MSE of the 16 read coordinates against one-hot answers. Both seeds see the same train-seed-90502 arrays. Per seed: **178,944 supervised answers, 279,968 observed writes**. Actual total: **357,888 answers, 559,936 writes**. Evaluation-only directories save the same planned arrays for exclusions; those are not extra training invocations. Final checkpoints are fixed endpoints; initial checkpoints and optimizer states are retained.

All 64 keys and atomic bindings are familiar. Evaluation excludes exact ordered training histories, including for untrained anchors, but local subsets/bindings can recur. The supported population is new complete closed-vocabulary histories, not new atomic facts, unseen keys or language.

## Acquisition and replay contract

Short admission requires ≥90% all-correct K2 acquisition/revision, ≥80% all-correct K4 acquisition and a positive lower 95% bound of normal-minus-rolled K2 all-correct episode accuracy. The rolled bootstrap resamples eight cyclic-roll blocks of 16 episodes, 5,000 draws, seed 90504. Calibration additionally requires ≥80% K8 D0 four-target all-correct, stable D64 accuracy in [15%,85%] and a ≥10-point stable decline from D0, in both trained seeds. If those gates fail, the design specifies a diagnostic fallback rather than unrestricted tuning. No fallback was needed here.

Fresh comparison seed 90523 supplies 128 episodes per K8/16/32 × D0/16/64 cell. The same histories and fixed four initial targets are shared across trained/untrained models and schedules; recent-distractor accuracy is separate. Acquisition for a K is adequate only when **both trained checkpoints have ≥80% four-target all-correct at actual comparison D0**. Weaker cells remain acquisition/interference diagnostics. Stable and revised targets differ in age/FIFO residency, so cross-stratum gaps do not isolate revision cost.

Prefill is online-only: K facts followed by genuine revisions of positions 2/3, then exact fast-state serialization. Delay observations never revisit any initial key. Online executes K+2+D writes; repeat, continuous and periodic execute K+2+2D. Continuous/periodic share a latest-per-key FIFO of eight records, four frozen draws with replacement before each four-arrival block and the same selected records/order. Continuous inserts one extra write after each arrival; periodic inserts all four at block end. Repeat-current matches extra opportunities; boundary reset is a separate effective-state diagnostic. All write rules/gates stay identical. Equal opportunities are not a claim of equal nonzero gradients or hardware cost.

The prespecified primary is K8,D64 stable **periodic-minus-continuous** accuracy, shown for each trained checkpoint and their fixed-two-seed mean. The bootstrap resamples the same 128 whole history IDs across arms and both seeds, 5,000 draws with seed 90504. It describes sampled-history uncertainty for these checkpoints and one codebook, not training-procedure or codebook variability. Other contrasts/K/D cells are secondary or exploratory. Initializers are exact untrained anchors, evaluated even if their acquisition is weak; they are not additional trained seeds.

## Integration for Member A and Member B

Use the correlated loader, not the earlier orthogonal or byte loader:

```python
import torch
from clmm.pretrain.kv_correlated_portable import load_portable_checkpoint
from clmm.pretrain.kv_scheduler import StreamingReplay

model, provenance = load_portable_checkpoint(checkpoint_path, torch.device("cpu"))
model.eval().requires_grad_(False)
state = model.initial_state(batch)
state = model.write(key_ids, value_ids, state)  # int64 [B], observed facts only
scores = model.read(query_key_ids, state)      # int64 [B,Q] -> [B,Q,16]
torch.save(model.export_state(state), fast_state_path)
state = model.import_state(torch.load(fast_state_path, weights_only=True))
```

The correlated checkpoint format is `clmm.kv.correlated.v1`, method `neural_correlated`. State/config fingerprints include the actual codebook identity; wrong-rank, wrong-codebook and orthogonal states are rejected. A complete replay-session resume requires the corresponding slow checkpoint, fast payload and scheduler JSON, captured after every event returned by `prime`/`step` has been applied. `finish()` closes a stream; export/import pauses it. The bounded scheduler retains ≤8 literal FIFO pairs plus ≤4 frozen selections, no complete history. Its snapshot may contain an old value if a within-block revision occurs; current delays avoid such target conflicts.

Use the **portable loader**, added after the model runs in `4b119ef`. The frozen constructor rejects local regeneration because PyTorch 2.11 cluster versus 2.15 Windows normalization differs by at most 2.98e−8, changing the hash. The portable loader retains the saved literal vectors and exact hash/fingerprint; numerical tolerance checks only the seed/rank construction, never changes a stored value. It also separates CPU-aliased template/initial-parameter storage before strict loading, preserving every saved slow tensor. The frozen experiment source, trained weights, selected setting and predictions were not changed. See `portability-amendment.json` for the observed failed local reload and bounded fix.

Member A can place its sleep update behind this write/read/state contract while retaining the continuous matched-budget anchor. Member B can implement its update rule first and selective routing separately. Share encoding, decoder, observed histories and acquisition checks, while reporting extra slow training, adaptive state, pair buffers and update opportunities. A different Hebbian architecture needs an explicit checkpoint/state adapter; matching dimensions is insufficient.

Ordinary live nonlinear post-write storage is 128 KiB per stream; compact export/import is 64 KiB. These are actual unique tensor backing stores including auxiliary/scan views, not summed aliases. Fixed codebooks add 20 KiB per adapter. Reset initial states can share slow-initializer backing across a batch; rolled controls can hold equal but unaliased weight/aux copies. Use the recorded per-arm `resources.csv` rather than assuming every diagnostic has the ordinary alias layout. Counts exclude Python overhead, allocator reservations and training graphs.

## Recovery and next milestone

Full root: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-nonlinear-correlated-2026-11-15`.

Final component weights: `r32-90511/checkpoint.pt` and `r32-90515/checkpoint.pt`; corresponding `initial_checkpoint.pt` files are the untrained anchors. `checkpoint_admission.json` records hashes and actual qualifications. Comparison directories contain batch-16 boundary/final-state examples for online K8,D16 and their full episodes. The compact Git index is [results/nonlinear-correlated-2026-11-15](../results/nonlinear-correlated-2026-11-15). The adjacent `CLMM_Nonlinear_Correlated_2026-11-15.zip` will contain the full recoverable experiment and branch bundle; its adjacent verification JSON records completed archive checks.

Focused CPU checks passed 352 tests; the separate portable adapter passed 21. Independent CPU checks across all three codebook ranks and both initializers verified nested codes, aligned MLP initialization/RNG, exact current-state SGD, pure reads and exact persistent continuation. The CPU smoke is execution-only and does not count as calibration evidence. Independent artifact reconstruction covers 143,616 main score vectors, 1,536,000 planned writes and 2,232 serialization batches, plus selection/source/data/checkpoint checks. These are artifact checks, not retraining.

A separate strict-loader GPU reload reproduces all 2,432 selected answers and raw coordinates exactly. The portable Windows CPU reload reproduces all 2,432 argmaxes with maximum score error 3.58e−7, including actual bounded live scheduling and prefill/mid-block model-plus-policy resume. Delay-zero groups resume only at prefill. The later portable GPU check has now been retrieved and also reproduces all 2,432 queries exactly. On retry, Slurm confirms job13914782 ended TIMEOUT after1h3s, with all model/check steps completed; no job remains active. `cluster-closeout-addendum.json` supersedes the prior unknown status. The original e3b744c archive remains a historical snapshot.

The next scientific milestone is to compare the teammates' concrete sleep/Hebbian variants against the frozen shared baseline, with their own acquisition and resource checks. Broader language encoding/pretraining remains separate from this component result. Keep the earlier byte-transfer failures and orthogonal ceiling visible when writing the class progress report.
