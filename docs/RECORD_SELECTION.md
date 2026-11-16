# Record selection and value decoding — completed November 16

The unchanged839,248-parameter model learns record selection on familiar
formats and value decoding when the current index is explicitly supplied.
Ordinary key-to-value recall remains unreliable. All four main given-index
decoders pass full admission; none of the other twelve main checkpoints does.
No ordinary-value or retention baseline is admitted. Keep the qualified
typed-memory reference available for A/B's mechanism experiments.

Executed source: `708b038de7fb63506aa5f184513177684ab372ad`, branch
`sequence-record-selection`. Later commits add analysis/handoff, not training
source. Artifact root:
`C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-record-selection-2026-11-16`.
Read `result_summary.json`, `shared_checkpoint_registry.json`,
`analysis/final_independent_review.md`, and `coordinator-analysis-v3/` there.

## Completed protocol and results

Sixteen main cases cross two original language parents, digit/atomic keys,
and S/S+/D/V objectives. Every case gets20k updates,320k supervised answer
forks and30.4M input bytes. Inputs remain95 bytes, attention128, full256-way
head unchanged. Decoder D receives the correct current record index as a
visible input cue; S+ changes wording/endpoint placement. These are diagnostic
bundles, not isolated neural mechanisms. No model enlargement occurred.

The final set uses256 fresh worlds/group, two requested keys and four views:
original, record reversal/retagging, value exchange, and both. Query identities
stay fixed. The held-out group changes both fact and query templates while
preserving development-key assignments. Full-history novelty does not imply
every individual/masked fact is novel. All terminal choices and conditional
training precede the12:45:26UTC decision lock and final invocations.

Joint scores below require all four answers correct across the original and
reversed views. Percentages average four fixed parent-by-encoding checkpoints
on256 paired worlds; these are not1024 independent training replicates.

| Objective | Familiar development formats | Held-out fact + query formats | Admitted |
| --- | ---: | ---: | ---: |
| Natural selection S |96.29%|15.53%|0/4|
| Endpoint-aligned selection S+ |65.92%|5.76%|0/4|
| Given-index decoding D |100.00%|97.46%|4/4|
| Ordinary key-to-value V |13.38%|2.93%|0/4|

Admission requires205/256 BOTH-correct in all12 group/view cells and205/256
joint-original/order in all3 groups, per checkpoint. S familiar joint counts
are238–254/256; held-out formats prevent admission. All D familiar counts
are256; held-out counts are256,256,252,234. One atomic S+ case fits training
keys but scores9/256 on familiar development keys: endpoint alignment is not
a reliable repair under this recipe.

Prespecified familiar-development contrasts: S+−S **−30.37pp**
(97.5% interval[−32.62,−28.13]); D−V **+86.62pp**
([84.86,88.38]). These use5000 PCG64 seed91604 paired whole-world bootstrap
draws after fixed-model averaging, with two-primary Bonferroni adjustment.
Intervals condition on these checkpoints, not training-procedure variability.
D−V includes additional index information and a different query/training
objective; it does not identify a pure decoder-component effect.

## Conditional continuation

Both encodings met terminal-gate eligibility with natural S. All eight
6000-update continuations ran. Both arms inherit identical matching20k S
weights and reset AdamW. SD alternates3000 S/3000 D updates; full-V receives
6000 V updates on the same worlds/forks/95-byte budget. Composed inference
uses one checkpoint and two actual forwards; only its predicted ASCII0/1
chooses a frozen D query, otherwise the index is erased. Truth never routes.
Its190 input bytes/answer differ from single-pass V's95.

| Parent / key format | Familiar joint count: SD two-call / V one-call | Held-out formats: SD / V |
| --- | ---: | ---: |
|68101 / digits|255 /254|3 /2|
|68105 / digits|251 /250|134 /189|
|68101 / atomic|202 /0|0 /0|
|68105 / atomic|220 /222|0 /0|

Denominators are256. The+19.73pp familiar average comes entirely from the
68101/atomic case; the other three differences sum to zero. Held-out average
is−5.27pp. There is no uniform advantage, and no ordinary/composed continuation
passes full admission. Three full-V continuations nevertheless learn strong
familiar recall. These results do not establish architectural impossibility.
Without a matched26k all-V-from-language-parent control, they also cannot
attribute gains over the20k main V cases to selector pretraining.

## Checks, resources and reuse

165 focused tests pass on the frozen source. Independent data reconstruction,
decision-seal verification and full raw auditing pass:136 evaluations,
2,334,720 saved answer distributions and all routing arrays. Another5760 GPU
engineering vectors pass separately. Tensor audits cover68 checkpoints and
66 optimizer states including calibration. All368k main/bridge updates have
finite nonzero aggregate memory-parameter gradients; this establishes plumbing.
Independent compact-byte and coordinator statistics agree exactly.

Both cluster CPU and Windows CPU reproduce12,544 selected queries/784 original
GPU batches across48 terminal capabilities/all24 terminal weights. Original
0.002 logit/0.0002 probability bounds, argmax/routing and weight/RNG purity all
pass. Windows maxima are0.000418663/0.0000162125 for answers and
0.000120163/0.00000274181 for selector scores. This covers selected batches,
not every score or the preceding study's long-revision caveat.

All24 weights, exact config/source/codec hashes, training budgets and
capability cells are retained in the registry. The four main D weights are
**given-index decoding controls** with source facts visible inside attention.
They are not general key-value or retention weights. Three inspected
PNG/PDF/SVG figures have source CSVs; v3 adds a standalone bridge legend/margins
without changing any table. Future work should exclude789,300 consulted or
reserved histories; current final histories are now development evidence.

GPU job13928152 completed13:38:41UTC on ins093:2A6000/8CPU/32GiB,
23,535s =13.075 allocated GPU-hours. CPU-only13928622 completed exit0 after
849s,8GiB/four allocated scheduler CPUs (two requested). Both jobs are closed.
No renewal, unrelated cancellation, helper-selection change, push or baseline
tag move occurred. Local disk/launcher outages did not restart the batch run.

Full cluster evidence stays at
`/insomnia001/depts/edu/users/zc2819/clmm/record-selection-2026-11-16`.
The verified cluster recovery ZIP is3,060,477,420bytes/3699 entries, SHA256
`6965d2cd8cdf031589af8b9e083241157473fae747fe0ade09aaf4d437f598b8`.
It includes source.bundle, complete packets, raw scores, weights, optimizers
and closed GPU logs. Active post-log/caches are excluded. Local compact and
numerical transfers passed SHA/CRC checks. The raw ZIP predates Windows proof,
final figures/docs; the final handoff supplement supplies them. Original local
packet caches were trimmed only after verified remote copies.

Next Member C work: ordinary-format robustness and a separately trained
native32 acquisition baseline at the same model size, followed by explicit
stateful delay/revision/corruption tests after facts leave attention/buffers.
Current95<128 controls cannot establish retention. Team choices remain the
main evidence scope (controlled component/full sequence), format-transfer
admission, and whether two-call inference belongs in deployment comparisons.
Model enlargement has not been shown necessary. No native32 extension ran
inside this frozen study.

## Original prospective plan (preserved)

Member C's next diagnostic splits the two-record failure into record selection
and given-index value decoding, keeping all 839,248 registered parameters and
the full 256-byte output head. The run uses both original language parents,
digit and atomic key encodings, and four matched-update objectives: natural
selection, endpoint-aligned selection, explicit-index decoding and ordinary
value recall. All use 95-byte tagged-record inputs in a 128-byte attention
segment. This is acquisition inside the attention window, not retention.

The held-out group clones dev-seen key/value assignments but changes both
record and query template families. It measures their combined format shift;
it does not isolate query wording from record wording.

Frozen protocol and live/recoverable evidence:
`C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-record-selection-2026-11-16/`.
The session record will name the exact source, packets, Slurm allocation and
remote recovery archive. Full raw score arrays stay on Insomnia because the
local drive has about 4 GB free. Essential weights, compact predictions,
independent audits and selected numerical reproduction batches are synced.

Main training is prospectively 20,000 updates per condition, with a globally
matched 10,000-update resource fallback decided before main efficacy output.
Final histories are sealed until terminal-gate conditional bridge decisions
and any eligible bridge training finish. A selector/decoder curriculum is
eligible only under the explicit paired-parent gates in the frozen protocol;
all eligible encoding/parent/arm cases run or the entire bridge is skipped
for resources. Optimizers reset equally. Composed inference routes by the
predicted byte only and costs two full forwards per answer.

Visible index supervision, endpoint formatting and additional decoder cues
are diagnostic bundles. Masked-record oracle results are OOD references.
Neither gate passes nor narrow bootstrap intervals establish general memory
capacity, scientific equivalence, a Titans reproduction or biological novelty.
Keep the previously qualified typed-memory baseline available to A/B.

Hard compute stop: 2026-11-16 14:52:49 UTC; user research window ends 15:52:49.
At most two GPUs, no automatic allocation renewal, failures retained.
