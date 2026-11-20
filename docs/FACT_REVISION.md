# Shared fact, revision and distractor evaluation

Zeyuan explicitly clarified on November 14 that he is **Zeyuan: Pre-training & Evaluation**. Member A owns the sleep mechanism and Member B the biologically inspired/Hebbian module. This branch supplies common data, reusable baseline weights, reference replay schedules and measurements that both method tracks can use. It does not assign either teammate a new task or present the reference scheduler as their finished method.

## Why the shared checkpoint uses bytes

A byte is also a token; the difference is the vocabulary and segmentation. Our existing baseline treats each UTF-8 byte as one of 256 symbols. This was an engineering choice for the first complete-model pilot: the repository already contained enwik8, encoding is deterministic without an extra tokenizer artifact, and the small vocabulary keeps the input/output layers inexpensive. It was not an attempt to reproduce the paper's text representation. Titans uses the Llama 2 tokenizer with a 32K vocabulary and 4K-token training sequences ([paper v1, §5.1](https://arxiv.org/html/2501.00663v1#S5.SS1), checked November 14).

The current model has independent embedding and output matrices of width 128. Holding every other parameter fixed gives the following **configuration-size calculations**, not trained alternatives:

| Vocabulary | Embedding + output parameters | Total model parameters |
|---|---:|---:|
| 256 bytes, current | 65,536 | 839,248 |
| 2,048 subwords | 524,288 | 1,298,000 |
| 4,096 subwords | 1,048,576 | 1,822,288 |
| 32,000 subwords | 8,192,000 | 8,965,712 |

Subword tokens can cover more text per sequence position, but the actual compression depends on the tokenizer and corpus. A fixed number of bytes is not the same context budget as the same number of subwords. These byte scores cannot be compared directly with the paper's token perplexities. Also, replacing the vocabulary would change the checkpoint's embedding/output shapes and interpretation; it is not a drop-in change to the two saved models.

This experiment keeps the existing tokenization so that the checkpoint and replay comparison remain controlled. A next shared language baseline should evaluate a modest, saved subword tokenizer (for example 2K or 4K BPE) trained only on the training portion, then train compatible weights. The paper's FineWeb-Edu corpus and training scale are recorded in [SHARED_PRETRAIN.md](SHARED_PRETRAIN.md); a sampled, versioned corpus would be a separate pilot. Measure tokenization coverage and actual throughput before selecting its budget. Full paper-scale pretraining is not established as feasible by this small run.

## Fixed experiment

Source commit: `646c670`. The two parent checkpoints are the fixed seeds 68101 and 68105 from the shared byte-language-model pilot. The prospective protocol is preserved in `results/fact-revision-2026-11-14/design.json` once results are recorded.

Both receive the same 1,600-step task-adaptation budget, batches of 16, example seeds and query selections. All outer weights are adapted with answer-position cross-entropy over all 256 bytes. The loss-independent curriculum moves from one fact to up to eight facts, revisions and distractors. Final checkpoints are retained without selecting the best validation step or seed. Training uses only the online stream; replay is an inference intervention.

Keys are three-digit strings, split by a fixed hash permutation into 600 training, 200 development and 200 reserved test identities. Each position contains every digit in every split. Values are fresh random letters a–p; revisions always change the value. Two template families are used: two seen training templates, and two heldout templates assessed separately. The experiment runner does not evaluate the reserved test namespace.

Each fact occupies exactly 32 bytes, aligned with the attention segment and memory refresh boundary. The query occupies a new 31-byte prefix; its answer is the next byte and is excluded from model inputs. The model must use neural memory to carry facts across records. The complete prefix is recomputed independently for each query; this does not yet test a deployed persistent session or variable-length prose.

Short diagnostics test one-fact acquisition, two-fact acquisition, a single revision, and a paired relabeling of all values that leaves query words unchanged. They compare ordinary inference with memory reset between attention segments. Full-vocabulary accuracy, answer NLL, probability assigned to the 16 legal answers, and forced-choice accuracy are all saved. Forced-choice accuracy alone cannot establish successful answer generation.

The pressure grid uses 4/8/16 initial facts, revises half, then adds 0/8/32 novel-key distractors. Each episode supplies one value-independent sampled stable query, one revised query and, when present, the most recent distractor query. Stable, revised and recent groups have different ages/exposures and are reported separately.

## What matched replay means here

The working buffer holds the latest observed record for at most eight keys, using FIFO order with recency refresh on a real revision. At each four-record block start, the scheduler samples one record per upcoming online event from that already available snapshot. Continuous replay inserts those selected records after successive online events. Periodic replay inserts the **same records in the same order** at block end. Neither arm may read a future record. The first empty-buffer block gets no extras.

The repeat-current control adds the same number of records but repeats each current observation. Thus continuous, periodic and repeat-current match input-byte counts, query position and memory-chunk/update opportunities at completed evaluation boundaries. The online baseline has fewer updates and is explicitly unmatched. Wall time and actual input lengths are also recorded; equality of logical budgets does not imply identical hardware runtime.

This is raw-record replay through the entire transformer, including embeddings and attention. It is a reference for common evaluation, not memory-only consolidation. The buffer stores at most 256 raw text bytes plus up to four snapshot selections and bookkeeping; evaluator traces and reconstructed full prefixes use additional storage and are not available as a retrieval database to the query.

Replay text carries no version marker. Reintroducing an obsolete assertion after a revision can therefore create a conflicting last assertion. Revised-answer and obsolete-answer rates diagnose that situation; they do not by themselves identify a consolidation failure. Stable old facts are the primary comparison. A version-aware replay protocol would require separate implementation and adaptation.

Online histories are paired across delays. K4 has a partial block after its six initial/revision records, so longer delays change that partial block's periodic replay placement. Its delay curve is exploratory. The K8 primary and K16 settings align at full-block boundaries.

## Analysis and interpretation

The fixed primary comparison is periodic minus continuous **full-vocabulary stable-fact accuracy at K8, D8**, with 128 episodes per checkpoint. Other pressure cells use 32 episodes. Confidence intervals resample complete paired episodes 5,000 times, separately for each fixed checkpoint. K2's two questions stay together in the bootstrap; two checkpoints are not pooled into a training-procedure uncertainty estimate.

Before interpreting retention, each adapted checkpoint must reach at least 80% full-vocabulary two-fact acquisition on seen templates and have a positive lower paired interval over segment-reset inference. The one-revision diagnostic is checked separately. Failed acquisition makes pressure results diagnostic of a weak baseline, rather than evidence that replay cannot help. A periodic advantage is a useful lead only if both checkpoints pass acquisition, both primary intervals are positive and both point advantages are at least five percentage points. The wider grid and heldout-template results are exploratory.

## November 14 results

Both complete-model adaptations and their full evaluations finished on an RTX A6000 in Insomnia. Independent reconstruction of the saved data, source snapshots, logits, labels and budgets verified both fixed seeds. This verification establishes the integrity of the recorded experiment, not a successful model.

| Fixed parent | K2 acquisition before | K2 acquisition after | K8/D8 continuous stable accuracy | K8/D8 periodic stable accuracy | Periodic minus continuous, 95% paired interval |
|---|---:|---:|---:|---:|---:|
| 68101 | 0/128, 0% | 6/128, 4.69% | 3/128, 2.34% | 3/128, 2.34% | 0.00 pp [0.00, 0.00] |
| 68105 | 0/128, 0% | 6/128, 4.69% | 4/128, 3.13% | 1/128, 0.78% | -2.34 pp [-5.47, 0.00] |

The adapted models assign almost all probability to legal answer letters, but their K2 accuracy is around the 6.25% uniform 16-answer reference. Resetting memory makes no difference to K2 accuracy. Both acquisition gates fail. The grid therefore does **not** establish whether replay helps a model that can acquire facts; these are floor diagnostics. A zero observed paired bootstrap interval is not proof of population equivalence. All four arms, revision/obsolete-answer rates, full-vocabulary and forced-choice losses, and timing are retained in the artifact tables.

A post-hoc view of the training logs gives a useful lead. Both models average 100% on fresh one-fact/no-delay training batches in steps 221–320. On the 44 one-fact/no-delay batches sampled after step 640, their average accuracy is only 17.76% and 11.65%. The actual number of revisions in these one-fact examples remains zero because the generator uses `floor(K * revision_fraction)`. These are training-log slices, not intermediate heldout validation results; no intermediate checkpoint was saved. The decline motivates checking acquisition throughout the curriculum, but does not establish its cause.

### Separate attention-access diagnostic

The conditional diagnostic was specified before either control run, after weak main training behavior was visible. Starting from original parent 68101, two models received exactly the same 1,000 steps, 16,000 answers and K2/no-revision training arrays. Their initial learned tensors were saved and independently verified equal to each other and the original checkpoint. Only the local attention segment length differed: 32 or 128. All 95 input bytes fit inside the longer window. Source: `ed2a8dc`; fresh development seed 78123.

| Control | K1 seen accuracy | K2 seen accuracy | Single-revision seen accuracy |
|---|---:|---:|---:|
| Attention segment 32 | 0/64, 0% | 67/128, 52.34% | 31/64, 48.44% |
| Attention segment 128 | 64/64, 100% | 65/128, 50.78% | 31/64, 48.44% |

The paired K2 difference (128 minus 32) is -1.56 percentage points, with a 95% whole-episode interval of [-5.47, 2.34]. Wider direct attention did not resolve two-key acquisition under this budget. The 128-window reset is exactly a no-op on all 768 paired short queries, as expected. The 32-window control drops from 52.34% to 4.69% under reset on K2: memory carries useful answer information in that simpler curriculum, but accurate key-specific retrieval remains unresolved. This one-parent exploratory control has different training from the main run and does not rescue its replay comparison.

These controls trained only on two distinct facts without revisions. K1 changes sequence length, and the revision task reuses a key, so those two rows measure transfer beyond the control training distribution.

A post-hoc query-sensitivity analysis identifies a concrete failure. In the 29 K2/seen episodes with different fact values and identical query wording apart from the requested key, **neither control changes its predicted answer in any episode (0/29)**. Thus neither answers both questions correctly in those pairs. Restricting to all 62 distinct-value episodes, every answer equals one of the two observed values; both controls select the first value on 59/124 questions and the last value on 65/124. They are not simply an always-last rule, but these observations do not show reliable key selection. On the single-revision seen diagnostic, both output the current value on 31/64 examples and the obsolete value on 33/64, with no other outputs. This analysis describes saved development predictions; it is not a new confirmatory endpoint or a causal identification of the internal mechanism.

### Reproducibility and scope

The scoped suite passes 170 main tests and 21 supplemental-control tests. The independent artifact analyzers additionally pass 21 and 8 synthetic selfchecks. An original-environment GPU inference rerun reproduces all 256 logits exactly for 128 selected answers, including acquisition/reset and continuous/periodic conditions. This is a bounded inference check, not a second training reproduction.

A separate CPU check keeps the same 32 predicted answers but fails its predeclared 0.001 absolute-logit tolerance (maximum difference 0.02495). It changes device, Torch version and batch geometry, so no one of those factors is identified as the cause. Both results are preserved: GPU Python 3.11.5/Torch 2.11.0+cu128 and local test Python 3.13.3/Torch 2.15.0+cpu. Do not claim cross-environment numerical identity.

Allocation 13911359 was cancelled deliberately after all work and output retrieval; accounting records show 55 minutes 19 seconds allocated and an empty user queue afterwards. The main and supplemental execution steps completed successfully. The prior allocation 13905404 is now confirmed timed out, and its bounded auxiliary diagnostic completed before the previous SSH disconnect; its metrics are not part of this experiment. No automatic allocation renewal occurred.

### Next Zeyuan implementation

1. Diagnose **key-conditioned acquisition** before expanding replay experiments. Keep the current serialization fixed initially; require changing the queried key to change the answer when the two facts have different values. Report accuracy separately for the first and second fact and compare with always-first/always-last-value rules. A model that emits a seen value without using the key is not a valid retention baseline.
2. Use a bounded K1/K2 curriculum with paired query supervision, explicit stage-end checkpoints and fixed development checks at each stage. Save every declared checkpoint, retain a final-step primary result and report stage trajectories; do not select a convenient checkpoint after inspecting the pressure outcomes. Compare any changed curriculum against the current one under a declared training budget.
3. Once both fixed parents pass K2 acquisition and revision checks, rerun the existing matched continuous/periodic/repeat-current protocol. Preserve the stable-fact primary outcome. Settle version handling before interpreting revised facts, because stale unmarked replay is conflicting evidence.
4. In parallel with method integration, measure a train-only 2K/4K subword tokenizer on a versioned language sample, document bytes-per-token and throughput, and train new compatible shared weights only after selecting the budget. This is a separate language-baseline track; it should not silently change the current controlled experiment.

Team choices to record are the ultimate tokenizer/corpus, the target state/read/write interface, version-aware versus unmarked replay, and the exact Hebbian/sleep adapters. These do not prevent Zeyuan from repairing and validating common acquisition now.

**Follow-up:** The [paired-key acquisition diagnostic](PAIRED_ACQUISITION.md) has now run both-query supervision against a matched duplicate-query control for both original parents, with fixed stage checkpoints and a wide-attention auxiliary. Single-fact recall and revisions work under the narrower recipe, but key-specific two-fact recall remains unqualified. The proposed curriculum diagnostic above was tested; it did not establish a successful multi-key baseline.

## Integration handoff

Member A can replace the reference replay execution with its memory-only sleep implementation while consuming the saved causal stream and selection plan. Member B can use the same prompts, queries, parent/adapted checkpoints and metrics with an explicitly compatible memory adapter. Frozen replacement measures transfer; a fair learned-method comparison also needs matched adaptation. Neither track should silently change label visibility, query context, replay exposure or checkpoint-selection budget.

The next shared-baseline priorities are successful acquisition before larger retention claims, a tokenization/corpus choice for the language-scale pilot, and an explicit state/read/write interface for persistent evaluation. These are Zeyuan deliverables supporting the two method tracks.
