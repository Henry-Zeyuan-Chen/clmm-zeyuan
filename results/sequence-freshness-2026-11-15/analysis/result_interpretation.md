# Independent result interpretation

The bounded raw-record integration is operational, but none of the eight unchanged-size sequence models acquires a reliable reusable two-key rule under this fixed recipe. All 24 required acquisition gates fail. No sequence checkpoint is admitted to retention or sleep efficacy comparisons.

## Acquisition and counterfactual results

Each number below counts complete distinct-value episodes; each episode has two independently answered queries with identical wording apart from the requested key. The admission threshold is at least 103/128 both-correct (80%). Joint four requires both the original and value-swapped pair to be entirely correct.

| Parent | Segment | Training policy | Fresh train /128 | Dev seen /128 | Dev heldout /128 | Joint four /128 |
|---|---:|---|---:|---:|---:|---:|
| 68101 | 32 | literal | 2 | 0 | 1 | 0 |
| 68101 | 32 | bounded | 1 | 2 | 0 | 0 |
| 68105 | 32 | literal | 0 | 1 | 1 | 0 |
| 68105 | 32 | bounded | 0 | 0 | 2 | 0 |
| 68101 | 128 | literal | 0 | 1 | 0 | 0 |
| 68101 | 128 | bounded | 0 | 0 | 0 | 0 |
| 68105 | 128 | literal | 0 | 1 | 0 | 0 |
| 68105 | 128 | bounded | 1 | 0 | 0 | 0 |

On fresh training-key/seen-wording episodes, every generated answer lies in that context's two current values, but each model returns the identical byte for both requested keys in 124–128 of 128 episodes. Per-question accuracy is 49.61–50.39%, versus 3.91–6.64% for query-only controls. These saved predictions show access to context values without reliable key selection; they do not identify an internal storage mechanism or prove an always-first/last rule. The first-value rate varies by model. All eight joint-four counts are 0/128.

The two declared bounded-training minus literal-training co-primary both-correct effects, averaged across the four fixed parent/geometry cases, are:

- Fresh train keys / seen wording: 0.0000 percentage points, adjusted 97.5% paired interval [-0.9766, +0.8813].
- Dev keys / jointly heldout fact and query wording: 0.0000 points, adjusted 97.5% interval [-0.9766, +0.9766].

Intervals use 5,000 whole-episode PCG64 draws, seed 91304, after averaging fixed-case differences within each episode. They describe history uncertainty conditional on these checkpoints; they do not estimate training-procedure variability. A zero effect at an acquisition floor, or degenerate secondary intervals from all-zero observations, is not population equivalence. These replay-ineligible K2 evaluation prefixes are identical across replay policies: the primary contrasts measure effects induced during adaptation, not direct test-time freshness repair.

Both native32 and independently trained native128 models fail. Native128 acquisition fits inside direct attention, and all 6,144 acquisition normal/reset pairs of 256-logit vectors in the six native128 initial/final cases are exactly equal as expected. The wider models' failure prevents isolating a neural-memory-only bottleneck. Nonzero memory gradients at every one of 16,000 main optimizer steps verify gradient plumbing; they do not establish useful key binding.

## Revision diagnostics

Every terminal checkpoint has 0/64 both-correct revision pairs under all six normal arms on both seen and heldout wording. Bounded replay can increase partial per-question revised accuracy; the figure deliberately shows this partial metric while stating that every complete pair fails. The 240 nominal secondary contrast rows retain all training policies, arms, template groups and obsolete-answer rates. These are stale-conflict/revision diagnostics of an unqualified baseline, not successful retention, sleep efficacy, or full-model replication of the earlier typed-component result.

The full transformer consumes the raw replay text, including attention, embeddings, feedforward computation and native neural memory. The fixed preprocessing parser recognizes the four supplied fact grammars, while the learned model receives unchanged original wording. Template transfer is conditional on this provided parser. The controller retains at most twelve raw records/384 payload bytes at rest and may use thirteen/416 bytes in a transition lookup, apart from metadata and complete model input/activation costs. Queries and oracle targets never enter replay selection. Full prefixes are recomputed independently per answer; persistent deployed sessions are not tested.

## Integrity, selection and resources

Independent saved-artifact checks pass all 52,224 main 256-logit vectors, exact row/prefix/target binding, full-vocabulary predictions, legal-answer mass, NLL, paired summaries, resource-row coverage, source/parent identities and checkpoint/optimizer tensors. Prediction bytes agree exactly with independent raw-score recomputation; the largest continuous-score rounding difference is 1.11e-6. All 48 executable source snapshots match Git blobs at `95da59531ccf5d5f3fe2729553e6d980da598352`.

All four initial states exactly equal their original parents. The eight terminal states retain 839,248 registered elements and original tensor shapes. AdamW state covers 830,960 active elements, with the same eight unused original-memory-model/frequency tensors across all trained models. All terminal optimizer states are at step 2000. The eight recovery-only midpoint checkpoints and optimizers are finite at step 1000; no midpoint is selected or evaluated as an efficacy endpoint.

Main totals are 16,000 optimizer updates, 256,000 supervised answer forks, 29,169,664 training input bytes, and 16,000 nonzero memory-gradient updates. K1 deliberately duplicates its one query to retain two forks per context. Per-case curriculum is 256 K1, 1,564 K2 acquisition and 180 revision updates. Evaluation totals are 52,224 answers, 10,269,696 input bytes and 3,264 resource batches. Separately audited engineering runs contribute 1,088 score vectors and two twenty-step runs; they are excluded from the main effect estimate.

The runtime-only preflight projection is 2,821.6675 seconds against the 4,200-second gate. Main execution takes 1,542.8670 seconds; the complete remote pipeline takes 1,583.0394 seconds, below the 5,000-second pipeline cap. Scheduler release/accounting is owned and recorded separately by the coordinator; this reviewer performed no GPU, SSH or model inference work.

## Numerical reproduction limitation

The coordinator's prespecified CPU check covers 1,200 selected scores from all twelve main models. All 1,200 predicted bytes match saved GPU predictions, and checkpoint/RNG purity checks pass. The original numerical acceptance nevertheless fails: maximum absolute logit error is 0.0187702179 against 0.002, and maximum probability error is 0.0007565087 against 0.0002. Ten rows in two terminal models fail at least one threshold (six logit failures, six probability failures, with two overlapping). All are 351-byte normal revision probes; every selected acquisition row remains within tolerance.

The post-failure diagnostic uses the exact original GPU cohorts. CPU scores equal the prior subset-CPU scores exactly on the ten flagged rows, so selected-subset batching does not explain the discrepancy. Single-row versus batch CPU perturbation is at most 0.00043869. Hardware and Torch version differ (GPU2.11 versus CPU2.15); the cause is not isolated. No tolerance was relaxed. This is a documented numerical reproduction limitation, distinct from the passing saved-GPU integrity audit. The reviewer audited these arrays and comparison artifacts without executing a model.

## Exclusion history and future use

The original 42,416-signature collector omitted JSONL/raw-prefix histories. A pre-execution amendment used an independently audited 81,326-signature union; original and final model inputs, semantic worlds, samples, plans and parents remain identical. A separately located CPU smoke adds 21 signatures. Actual frozen-producer generation under the wider 81,347 union reproduces every fixed evaluation episode exactly, with zero overlap. Original packets, failed preparation attempts and all failures are preserved.

`future_exclusion_signatures.json` contains 85,123 signatures for subsequent studies: the later 81,347 base plus all now-consulted real and actually presented raw-record histories from main and engineering training/evaluation. It is explicitly future-only and never rewrites a launch packet or its union.

The supported next decision is to repair and qualify reusable key-conditioned acquisition before scaling or making retention claims. The result is specific to this representation, objective, curriculum and 2,000-step budget. It neither proves that larger models are necessary nor that this architecture cannot learn the task.
