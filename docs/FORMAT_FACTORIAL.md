# Fact/query format factorial — complete, November 16

The unchanged 839,248-parameter model has a format-transfer problem in both
visible input blocks. In the four fixed full-value continuations, changing only
fact wording reduces joint success by **47.46 percentage points**; changing
only query wording reduces it by **35.35 points**. This supports a controlled
format-training experiment before enlarging the model or testing retention.

## Results

Each entry is mean joint-four-answer success across four fixed checkpoints,
on the same 128 fresh development-key worlds. Both requested answers must be
correct before and after record-order reversal. Predictions use the full
256-byte head. These tasks have different inherited training budgets; rows
are descriptive components, not equally trained competing methods.

| Task | Original F0Q0 | Fact change F1Q0 | Query change F0Q1 | Both F1Q1 |
| --- | ---: | ---: | ---: | ---: |
| Natural record selection S | 97.85% | 16.80% | 90.82% | 15.23% |
| Given-index decoding D | 100.00% | 100.00% | 97.66% | 96.48% |
| Ordinary value recall V | 72.07% | 24.61% | 36.72% | 16.60% |

V's two primary development-key contrasts have 97.5% paired-world percentile
intervals, adjusted for two comparisons: fact loss **47.46 [45.70, 48.83] pp**;
query loss **35.35 [28.91, 41.90] pp**. Worlds are averaged over four fixed
checkpoints before resampling. Intervals condition on those checkpoints and
this codec/template/sampling scheme; they do not estimate training variance.
The secondary correctness interaction is +27.34 [22.85, 31.84] pp (95%). Its
conjunctive metric and floors preclude a neural-mechanism interpretation.

Keep every V checkpoint visible (joint successes out of 128):

| Parent / encoding | Original | Fact change | Query change | Both |
| --- | ---: | ---: | ---: | ---: |
| 68101 / atomic | 0 | 0 | 0 | 0 |
| 68101 / digits | 127 | 0 | 59 | 0 |
| 68105 / atomic | 116 | 0 | 50 | 0 |
| 68105 / digits | 126 | 126 | 79 | 85 |

Fact transfer collapses in two fitted V cases; one fitted digit case preserves
it. Query transfer hurts all three fitted cases. The weak atomic case remains
at floor and supplies no equivalence evidence. Strong given-index decoding
and poor natural selection under changed facts motivate format-robust
key-to-record reading, without identifying its internal implementation cause.

All 24 capabilities finish on two Windows CPU threads in 1,449.99 seconds,
including the engineering canaries, within the 2,700-second inference bound.
Insomnia was unreachable in this session; no GPU allocation was made.
All 138,240 main distributions, byte/label joins, batching and erasure pairs
pass independent numerical audit. Independent raw-logit statistics agree
exactly with coordinator answer-byte joins (192 cells, 96 controls, 36 contrasts).
The 164 focused tests passed before source freeze. Three figures and their
source tables are in [results/format_factorial_2026-11-16](../results/format_factorial_2026-11-16).
The same-CPU reload proof passes all 13,824 selected predictions / 864 actual
batches across every capability, with bitwise-identical logits and argmax,
and unchanged slow weights/RNG. Its selection was frozen before main
completion. This is same-device repeatability, not cross-device equivalence.

## Next controlled training experiment

Retain the same model, attention length 128 and all four main natural-selector
checkpoints as paired initializers. Train ordinary V for 6,000 new updates in
each of two arms: the existing familiar-format recipe, and uniform mixing of
the four fact/query family cells. This is eight runs / 48,000 new updates,
with identical semantic histories, answer supervision, update counts, batch
shapes and input-byte budgets between paired arms. Use terminal weights;
retain the weak parent/encoding rather than selecting successful models.

Generate training, engineering and confirmation histories against the new
790,004-history exclusion union, with all four counterfactual views excluded
together. Freeze genuinely new confirmation wording before training: the
current template families are consulted development data. Keep fact-only,
query-only and combined shifts as separate endpoints, with full-head scoring,
fresh assignments, order/value swaps, erasure controls and initial controls.
Use a bounded GPU runtime preflight when Insomnia is available. This training
has not been run as part of the frozen-weight diagnostic.

This remains Member C's acquisition qualification. Keep the already qualified
typed-memory reference available for A/B. Robust ordinary sequence reading
must precede a separately trained native-32 baseline and stateful retention,
revision and corruption after context/buffer exit. Neither attention-128
lookup nor an explicitly supplied correct slot establishes those abilities.
Team choices still include whether ordinary natural queries are required for
the main project claim, which A/B adapters are final, and the retention task.

## Frozen prospective design

Member C's next diagnostic separates the two changes confounded in the
previous held-out condition. Each frozen checkpoint sees the same fresh
histories in F0Q0(original fact/query families), F0Q1(query change only),
F1Q0(fact change only), and F1Q1(both). These template families have already
been consulted; this is a fresh-history exploratory diagnosis, not a new
unseen-template confirmation. No training or model resizing occurs.

The twelve trained cases retain all4 main natural selectors,4 main given-index
decoders, and4 paired full-value continuations, including the weak atomic case.
The two original language parents supply12 task-unadapted capability controls
(3tasks×2encodings×2parents); they are not random untrained networks. Every
model retains839,248 parameters, full256head and128 attention. Inputs remain
95bytes. D retains its explicitly supplied correct visible index.

There are128 fresh development-key worlds(primary),32 training-key worlds,
four counterfactual views and two semantic query forks. The two original-view
F0Q0 plumbing controls are no_facts and cue_erased. All normal cells cover all
four views; controls do not. This yields34,560 frozen packet rows and138,240
saved score vectors across24 capabilities/14 distinct checkpoint files.
All semantic four-view closures reject the previous789,300-history union and
the new engineering histories. Individual fact novelty is not claimed.

Episode invariants and global template dictionaries remain unchanged. The new
renderer assembles individual32-byte tagged record blocks and31-byte query
blocks with independent family choices, preserving labels/identities/slot tags.
The diagonal cells reproduce the old renderer exactly; mixed blocks are tested.

The primary metric is joint-original/order four-answer correctness on the
development keys. Primary V contrasts are F0Q0−F1Q0(fact-format loss) and
F0Q0−F0Q1(query-format loss), averaged across four fixed models per world before
5000 paired-world PCG64/91804 resamples. One shared index matrix is reused;
97.5% intervals implement two-primary Bonferroni. Correctness interaction
y11−y10−y01+y00 is secondary95%, as are S/D/initial and train-key results.
Intervals condition on the fixed checkpoints. Floors do not prove equivalence.
Both-erased forks have identical inputs/different labels, so their structural
both-correct zero is a leakage/parity check. Format effects bundle words,
padding and key position; they do not identify an internal neural mechanism.

Insomnia was not reachable during this session's probes. Local CPU uses two
threads with a45-minute inference cap, source/data/checkpoint hashes, no weight
updates, and a representative engineering runtime gate before any main output.
Either all24 main capabilities fit or the main phase stops; no outcome-driven
sample/model reduction. Data and saved distributions receive independent audits.
No native32/retention claim or new training belongs to this diagnostic.

Artifacts: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-format-factorial-2026-11-16`;
launch work in sibling`work/clmm-format-factorial`. Root owns all execution;
an independent reviewer owns analysis-only reconstruction and statistics.
Preserve the pre-existing SLEEP_MULTIREVISION edit and baseline tag. No push,
teammate message or Google Doc edit is authorized by this phase.
