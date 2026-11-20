# Paired-key acquisition diagnostic

This is Zeyuan's Zeyuan pretraining/evaluation work. The previous [fact/revision experiment](FACT_REVISION.md) failed its acquisition gate. A separate simpler control returned observed values but did not change its answer in any of 29 development pairs where only the requested key changed. This diagnostic tests whether supervising both keys in each context improves that behavior.

## Fixed comparison

Source `d82030974596207a37044ba3951fbbbe242e2d00` adds `scripts/fact_paired_acquisition.py` and data-contract tests. The original two shared checkpoints, seeds 68101 and 68105, each receive two runs with attention segment 32:

- **Paired:** eight real contexts per batch, with both query keys independently supervised in each two-fact context.
- **Duplicate:** the identical eight contexts, each with one uniformly sampled query repeated twice.

The arms match original weights, optimizer settings, base events, lengths, 16 answer positions per update and 1,600 updates. Query exposure differs deliberately. No target from one query is appended to the input of the other. Both use ordinary full-256-byte cross entropy, without an auxiliary discrimination loss. In expectation their two-fact objectives agree; any benefit could result from per-context balance or lower gradient noise, rather than a different population objective.

Two-fact training contexts have distinct initial values, selected by deterministic rejection from the existing generator, and identical query wording apart from the key. Accepted underlying episode seeds and complete online records are preserved. Revisions may make final values coincide. Training uses only the existing 600-key train namespace. Original 32-byte record/31-byte query encoding remains fixed.

The schedule is 200 one-fact steps, 1,000 two-fact steps, then 400 steps with a uniform batchwise choice of one fact, two facts, a single revised fact, or two facts with one revision. AdamW uses learning rate 0.0003, weight decay 0.01 and gradient clipping 1. Training seed 89102 is shared across arms. All stage endpoints (200/1200/1600) are saved with optimizer states and evaluated. Step 1600 is the primary endpoint; earlier checkpoints describe the curriculum and never replace an unfavorable final result.

A fifth run uses paired supervision and attention segment 128 from parent 68101. All training and short evaluation inputs fit inside that window. This diagnoses acquisition with direct attention; its memory-reset ablation is expected to be exactly a no-op. It is separate from the four memory-dependent comparisons.

## Evaluation and decisions

Fresh development seed 89103 defines the existing acquisition/revision/counterfactual suite (64 episodes per cell), plus 128 strict distinct-value two-fact episodes per template family. Strict pairs change only the requested key. Report per-query accuracy, each rank's accuracy, prediction changes and **both questions correct**. An always-first/last-value rule gets 50% per query but 0% both correct on these distinct-value pairs. Query-change rate alone cannot establish correct retrieval.

The primary effect is paired minus duplicate final-step strict seen-template both-correct accuracy, separately for each original parent. Bootstrap 5,000 whole paired episodes using seed 89104. Intervals describe each fixed checkpoint and sampled development episodes, not training-seed variability. A useful pairing lead requires both lower bounds above zero and both point effects at least five percentage points.

Before continuing replay, both final paired-32 models must reach 80% natural K2 accuracy, 80% strict seen both-correct, a positive lower interval over reset-memory K2, and 80% seen single-revision accuracy. Passing the wide-attention control alone is insufficient. Reserved test keys remain unused. The new development seed provides new assignments within a previously development-exposed key namespace.

## Execution and results

All five runs finished on an Insomnia RTX A6000. Each used exactly 25,600 supervised answers and 2,331,136 input bytes; all 15 stage checkpoints and optimizer states are retained. The independent audit verified every input/label, semantic pairing, source-to-Git hash, initialization tensor, checkpoint configuration and saved prediction. Warmup checkpoint tensors are bit-identical across paired/duplicate arms for each parent, as expected from identical K1 training. The auxiliary 128-window model's reset logits are exactly unchanged.

| Final model | Natural K2 per-query accuracy | Strict both queries correct | First-key accuracy | Second-key accuracy |
|---|---:|---:|---:|---:|
| Duplicate32, parent 68101 | 52.34% | 0/128, 0% | 1.56% | 98.44% |
| Paired32, parent 68101 | 51.56% | 1/128, 0.78% | 3.91% | 96.88% |
| Duplicate32, parent 68105 | 52.34% | 0/128, 0% | 5.47% | 94.53% |
| Paired32, parent 68105 | 52.34% | 1/128, 0.78% | 3.91% | 96.88% |
| Paired128 auxiliary, parent 68101 | 53.13% | 1/128, 0.78% | 17.97% | 82.81% |

The primary paired-minus-duplicate effect is +0.78 percentage points for each fixed parent, with a 95% paired-episode interval of [0.00, 2.34]. This fails the declared useful-lead criterion. Paired32 predictions change in only 1/128 strict key-switch pairs for each parent; duplicate32 changes in 0/128. The final behavior strongly favors the second/recent value, and does not establish reliable key selection. The wide-attention auxiliary also fails, so these results do not isolate the difficulty to cross-segment neural memory. The experiment does not identify whether more training, a different representation, model capacity or a different objective would resolve it.

The narrower curriculum does yield useful single-key checkpoints. Both paired32 final models achieve 64/64 (100%) seen-template one-fact accuracy. They achieve 61/64 (95.31%) and 63/64 (98.44%) on seen-template single revisions. Natural K2 accuracy has a positive lower paired interval over reset (39.84 and 39.06 percentage points). Thus the path can carry useful information and handle single-key updating, while failing arbitrary two-key retrieval. These observations are not a causal effect of pairing: the new curriculum/learning rate differ from the older experiment, and both new arms share them.

Both final paired32 models fail the natural K2 and strict both-correct requirements, although they pass the revision and reset-difference requirements. **Replay continuation is not qualified.** Larger fact counts or delays would currently mix the key-selection failure with retention. Earlier stages remain in the tables: paired32 both-correct at step 1200 is 1/128 and 2/128; the auxiliary peaks at 3/128 there. Those exploratory stage values do not replace the primary final result. Zero-event empirical bootstrap intervals are not proof of zero population error or equivalence.

### Validation and resource use

The 122-test scoped suite passes, including 33 new data/matching tests. A separate reviewer found no demonstrated answer-position or causal-boundary bug; suggested provenance hardening was incorporated before the frozen commit. The artifact auditor passes 25 synthetic corruption/selfchecks and independently verifies 128,000 training answers, 23,040 gate predictions and 15,360 strict predictions. It does not execute the model or reproduce training.

A bounded GPU inference reload check reproduces every saved logit exactly for 160 selected final-checkpoint answers across all five runs, ordinary and reset conditions. Source and device/framework provenance accompany that check. All numerical plots are generated from the audited tables and exported as PNG/PDF.

Allocation 13911789 on ins083 was cancelled deliberately after completion, GPU checking and retrieval. Accounting records 26 minutes 25 seconds allocated (about 0.44 GPU-hours). The main pipeline and inference step completed with exit 0. A lightweight inspection step returned exit 2 because an optional old cache directory did not exist; this was not a training failure. Unrelated jobs and the desktop job selection were left untouched, with no automatic renewal.

### Next decision

The next useful diagnostic is **trainability versus generalization**, before another replay sweep. Use a tiny fixed set of two-fact contexts with both queries and a declared finite overfit budget; compare training-set both-correct with fresh assignments on the same keys and on heldout keys. Require a direct-attention positive control first. If that model cannot fit the fixed examples, inspect optimization/representation/model capacity. If it fits them but fails fresh assignments, the main gap is learning a reusable key-matching rule. This is a proposed next diagnostic, not an experiment already run.

Keep the current byte format and single-key checkpoints available for the sleep/Hebbian tracks, but label the two-key baseline unqualified. Tokenization/corpus choices and a stronger language-pretraining baseline remain separate Zeyuan decisions; no paper-faithful Titans reproduction or sleep benefit is established here.

The prospective design, all successful/failed runs, full training arrays/plans, raw logits, independent artifact analysis and source snapshots are retained under `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-paired-acquisition-2026-11-14`. No report prose or teammate method module is required to run this diagnostic.

**Completed follow-up:** The [fixed-set/assignment-transfer diagnostic](FIXED_SET_GENERALIZATION.md) now establishes perfect finite-bank fitting in both geometries and parents, with failed novel-assignment transfer. Its conditional fresh-assignment training branch also fails the training-disjoint transfer target. The next baseline work is representation/association training and the shared persistent-state interface, rather than a further replay sweep on these unqualified models.
