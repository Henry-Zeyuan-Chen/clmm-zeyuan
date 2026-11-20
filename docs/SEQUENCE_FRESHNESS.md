# Bounded freshness in the complete byte-sequence baseline

Status: experiment complete, November 15, 2026. Branch `sequence-bounded-freshness`; executed source `95da59531ccf5d5f3fe2729553e6d980da598352`. Integration and matched training ran successfully, but all 24 acquisition gates fail. The baseline is not qualified for retention/sleep conclusions. This follows the typed-memory [bounded freshness study](BOUNDED_FRESHNESS.md); no model size increase was made.

The model remains the original 839,248-registered-parameter byte-MAC approximation: token embeddings, two attention/feedforward layers, persistent tokens, native neural memory and full 256-byte answer head. The two original language checkpoints 68101/68105 initialize every paired case with identical tensors. Native attention 32 and wide 128 are trained independently. Neither geometry changes parameter count; 128 gives an acquisition condition where ordinary attention sees both original facts.

`SequenceReplay` wraps the existing `StreamingReplay` controller. It accepts only observed 32-byte fact records, parses their key/value for FIFO selection, and retains exact raw source bytes rather than canonicalizing wording. Queries, desired labels, revision markers and episode objects are not policy inputs. State retains the union of FIFO 8/snapshot 4 payloads: at most 12 records/384 raw bytes plus controller metadata. A transition can temporarily use 13/416 bytes; do not call 384 bytes a universal peak. Full LM input and activation/cache storage are separate costs. Default literal and opt-in bounded policies share candidate draws, keys and byte/update opportunities. On eviction, bounded uses the original sample.

Replay records traverse the entire unchanged transformer. This is a full-sequence record-replay reference, not memory-only sleep or a paper-faithful MAC claim. The scheduler recognizes all four provided observation grammars; heldout-template results therefore test neural reading under that provided parser, not generalization of an unknown-format parser. No cross-session complete-cache persistence is claimed by these in-prefix recall probes.

Eight fixed adaptations cross 2 initializers × 2 trained geometries × literal/bounded periodic replay. Each gets 2,000 AdamW updates, eight contexts/two independent answer forks per context, 32,000 supervised forks and 3,646,208 input bytes. Fresh random assignments use 600 training keys and seen wording. Fixed curriculum: 256 one-fact updates (duplicate the query for equal fork count), 1,024 two-fact updates, then 720 updates mixing 3 two-fact acquisitions to 1 both-revised/two-distractor context. Two current target values stay distinct after each true revision. Only final step 2000 selects a model; step 1000 is recovery-only. No extra alignment labels or new learned modules are added.

The policy primes the first min(2,n) real records, then observes block 4. This intentionally differs from the older fact scheduler's empty first block. Plain K2 acquisition has 95 input bytes and no extras; revision contexts have 6 real records+4 extras+31 query bytes=351. NPZ arrays are padded for storage only and sliced to actual length before forward. Literal/bounded training differs in 180 revision updates; no-extra acquisition probes measure adaptation spillover/generalization, not direct test-time freshness repair.

Relation to the team's final proposal: Zeyuan supplies shared weights, training feasibility and common acquisition/retention/revision evaluation. The Hebbian proposal additionally requires replacing the adaptive memory, selective local traces and frozen evaluation parameters; this study does not implement that replacement. The sleep proposal calls for memory reconstruction updates during a pause, followed by recall after attention/buffer context is gone. Our raw-record full-transformer replay is an explicit reference control, not that final sleep intervention. Complete cache carry across sessions, retention beyond the buffer and corruption recovery remain separate unqualified requirements. Acquisition is the current gate before those claims.

Fresh evaluation uses 128 paired worlds for training-key seen wording, value-swapped counterfactuals, dev-key seen wording and the same dev semantics in heldout fact/query templates; revision probes use 64 paired worlds. Both questions share wording except key and are separately forwarded without the other's answer. Initial/final complete models retain online/reset and query-only controls; revision probes include all six replay arms. Report full 256 argmax, both-correct, per-rank accuracy, NLL, legal mass and actual key-switch responses. A changed prediction alone does not establish key-conditioned correctness.

History exclusions hash the complete ordered observed key/value sequence, independently of wording. The training-key probe can reuse individual key/value facts; its complete assignment history and the paired swapped history are excluded from actual prior/new training. Dev keys are disjoint from training keys. Do not describe all test facts as individually unseen or the jointly heldout wording probe as an untouched future test after this study.

Two prespecified contrasts compare bounded-trained minus literal-trained BOTH-correct accuracy on fresh training-key seen assignments and dev-key heldout templates. Resample 128 whole-world IDs jointly across fixed parents/geometries using 5,000PCG64 draws 91304; two 97.5% percentile intervals implement the Bonferroni adjustment for two tests. Per-case admission requires>=80% both-correct on fresh train-seen/dev-seen/dev-heldout. Native 32 memory dependence is separately checked against reset. Keep all failures and scope results to the fixed models/histories. If acquisition fails, report a working integration and unqualified baseline rather than a sleep/retention result.

Prelaunch review found repeated lazy NPZ decompression and missing old JSONL/seed/raw-prefix exclusion coverage. Source `95da595` caches the selected arrays once. The independent 81,326-signature union includes actual older raw training contexts; regenerated `packet-final` and `engineering/packet-final` preserve all original worlds/arrays/plans/samples/parents exactly. Original d0b7455 packets remain for provenance. Earlier `packet-executed` attempts used a shortened audit filename, failed before model training and remain preserved with logs.

238 scoped tests pass. A CPU prototype completes 8 updates and 100 scores with exact reload/pure-evaluation checks; its short curriculum does not include revision training. Main and 20-step engineering packets independently reconstruct, with zero engineering-training/main-evaluation overlap. Source snapshots contain 48 files matching executed Git. All 52,224 saved GPU score vectors, 12 checkpoints and 8 terminal optimizer states pass the independent saved-output/provenance audit. Recomputed metrics differ by at most1.11e-6; predicted bytes agree exactly. Coordinator separately recomputes the two primary intervals and all budgets from raw CSV, with exact agreement. All 16,000 main optimizer steps report finite nonzero neural-memory parameter gradients; that alone does not prove useful online-memory learning.

Actual CPU reload/inference covers 1,200 selected queries across all 12 models and every group/arm/condition, with unchanged weights/RNG and identical predicted answers. The original strict score thresholds, 0.002 logit/0.0002 probability absolute error, remain unmet: six logit failures plus four probability-only failures, ten 351-byte revision rows across two terminal models. Maximum logit error 0.0187702, probability error 0.000756509. Exact original GPU batch membership produces the same selected CPU outputs, so subset batching does not explain the mismatch. Windows CPU Torch 2.15 and Linux GPU Torch 2.11 differ in software and hardware; the cause is not isolated. Keep `cpu-recheck/verification.json` as `numerical_review_required`; selected prediction agreement is established, full numerical portability is not. No thresholds were relaxed and no new GPU run was launched to replace the failure.

Compute: one 90-minute 2 A6000 allocation, 8 CPU/16 GiB, no renewal/helper selection change. Job 13922356 on ins080 passed import and runtime admission: padded projection 2,821.67 s<=4,200 s. Two geometry preflights use 20 updates plus N16/revision 8 evaluation; hard pipeline 5,000 s. Complete pipeline 1,583.04 s, main 1,542.87 s. The allocation was cancelled after 1,634 s, 0.9078 allocated GPU hours; later accounting/queue checks confirm closure and no active user jobs. All conditions were kept. Root owned resources; independent reviewer performed source/data/raw arithmetic checks without model execution.

The separately located CPU smoke contributes 21 additional history signatures beyond the supplied 81,326 union. The later 81,347 union has zero overlap with all 448 unique main evaluation histories; actual executed-source data generation reproduces every evaluation episode unchanged under it. `analysis/work_smoke_regeneration_review.json` records this additive coverage check. Original launch packets remain immutable; `engineering-cpu` preserves 66 hash-verified smoke files. For future studies use the 85,123-signature union in `analysis/future_exclusion_signatures.json`, including now-consulted real and actually presented replay histories; keep launch files unchanged.

The replay bridge can be reused without query access:

```python
from clmm.pretrain.sequence_replay import SequenceReplay

policy = SequenceReplay("periodic", seed=7, freshness="bounded")
emitted = []
for raw in observed_records[:2]:  # each record is exactly32 bytes
    emitted.extend(policy.prime(raw))
for raw in observed_records[2:]:
    emitted.extend(policy.observe(raw))
emitted.extend(policy.finish())
prefix = b"".join(event["record_bytes"] for event in emitted) + query_bytes
```

Use `freshness="literal"` for the matched frozen-snapshot control. `export_state()`/`SequenceReplay.from_state()` resume the controller and retained raw records; they do not persist a complete transformer's caches. `clmm.pretrain.load_checkpoint(path,map_location="cpu")` loads the saved sequence model/config/payload. Its weights are diagnostic until acquisition admission passes.

## Results and next decision

Every entry below is BOTH-correct worlds out of 128, using the full 256-way output head. The per-case admission threshold is 103/128 on each of the first three columns; zero of 24 gate cells pass.

| Parent / trained segment / replay | Train keys, seen | Dev keys, seen | Dev keys, heldout | Joint original+swap four answers |
| --- | ---: | ---: | ---: | ---: |
| 68101 /32/literal |2|0|1|0|
| 68101 /32/bounded |1|2|0|0|
| 68105 /32/literal |0|1|1|0|
| 68105 /32/bounded |0|0|2|0|
| 68101 /128/literal |0|1|0|0|
| 68101 /128/bounded |0|0|0|0|
| 68105 /128/literal |0|1|0|0|
| 68105 /128/bounded |1|0|0|0|

Bounded-training minus literal-training co-primary effects are 0.000 pp on fresh train keys, adjusted 97.5% CI[−0.9766,+0.8813]pp, and 0.000 pp on dev keys/heldout wording,[−0.9766,+0.9766]pp. These are paired-history intervals conditional on the four fixed parent/geometry pairs, not training-procedure uncertainty or equivalence evidence at this floor. Training-key individual-answer accuracy is 49.61–50.39%, while requested-key prediction changes occur in only 0–4 of 128 pairs. Both original and swapped question pairs are jointly correct in 0/128 worlds for every checkpoint. Every saved answer in the fresh training-key probe is one of the two observed values, but 124–128 of 128 pairs use the same predicted byte for both keys. The model accesses context values without reliably selecting the requested association.

All eight models also score 0/64 BOTH-correct under every normal revision arm in both wording groups. Bounded replay changes partial per-question revision accuracy, but none of these unqualified checkpoints establishes revision or retention efficacy. Wider 128 attention does not repair fresh key selection under this recipe. This implicates key-conditioned reading as the immediate diagnostic target; it does not identify tokenization, capacity or any particular layer as the cause.

Keep the existing qualified typed-memory reference for A/B's controlled component comparisons. Zeyuan’s next sequence task should isolate key/value binding with a controlled representation or input-format contrast at fixed resources, retaining fresh assignment, value-swap and heldout-template gates. Do not add longer delays, increase parameters, select a better midpoint or keep training on these consulted histories as an unplanned remedy. A scale increase needs a separate protocol and a reason supported by the binding diagnostic.

Team choices to settle before final method claims: A's intended memory-only reconstruction sleep versus this full-record control; B's intended biological update versus the existing local-delta reference; whether to retain the byte input path or compare an explicit key/value representation; and whether the latest observation is authoritative under corruption. For this study, latest truth, known fixed parser and original-record fallback on eviction are explicit adopted assumptions. Only periodic replay is crossed during adaptation; six schedules are evaluated afterward, so this does not compare separately trained continuous/periodic procedures.

Artifacts: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-sequence-freshness-2026-11-15`. Downloaded outputs are `gpu-output/result`; executed packet is `packet-final`, original pre-amendment packet is `packet`. Source/launch work is sibling `work/clmm-sequence-freshness`. `validation.json` separates completed saved-GPU checks, failed acquisition and CPU numerical limits. Three numerical PNG/PDF/SVG figures were visually reviewed. The compact Git index is `results/sequence-freshness-2026-11-15`; the adjacent `CLMM_Sequence_Freshness_2026-11-15.zip` and verification file preserve full inputs, checkpoints/optimizers, source bundles, failed preparations and hash/CRC checks. Preserve the existing SLEEP_MULTIREVISION.md user edit. No push, teammate message or GoogleDoc change was made.
