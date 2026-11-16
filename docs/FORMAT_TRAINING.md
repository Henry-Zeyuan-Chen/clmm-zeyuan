# Matched format training — complete, November 16

All eight fixed-size runs finish their full 6,000 updates. Mixed wording
improves the prespecified average on all three new-wording endpoints, but
reduces familiar recall and has sharply different effects across cases.
**No terminal passes the full acquisition requirement (0/8).** These are
reusable diagnostic weights, not admitted sequence-retention baselines.

## Results

Full-256 joint4 requires both requested answers correct before and after
record-order reversal. Entries average the four fixed initializer/encoding
pairs on 256 fresh development-key worlds. F2/Q2 wording was never used in
training or engineering neural execution. F1/Q1 is now part of mixed training.

| Wording | Familiar-only | Mixed | Mixed minus familiar |
| --- | ---: | ---: | ---: |
| Familiar F0Q0 | 65.04% | 54.30% | −10.74 pp |
| New facts F2Q0 | 29.88% | 45.90% | +16.02 pp |
| New queries F0Q2 | 28.52% | 45.12% | +16.60 pp |
| Both new F2Q2 | 20.70% | 41.02% | +20.31 pp |

The three primary adjusted 98.3333% paired-world intervals are respectively
[13.38, 18.65], [12.89, 20.12] and [17.25, 23.34] pp. All three meet the
frozen positive-lower-bound rule for the fixed-case average. The familiar
cost is secondary: −10.74 pp, nominal 95% interval [−12.99, −8.59]. Intervals
condition on the four retained cases, exact training stream, codec and
template families; they do not estimate training-procedure variability.

New-fact / new-query / both-new effects by initializer, in percentage points:

| Initializer / encoding | F2Q0 | F0Q2 | F2Q2 |
| --- | ---: | ---: | ---: |
| 68101 / atomic | 0.00 | 0.00 | 0.00 |
| 68101 / digits | +82.42 | +80.47 | +83.59 |
| 68105 / atomic | −25.00 | −30.47 | −14.06 |
| 68105 / digits | +6.64 | +16.41 | +11.72 |

Both digit pairs improve, with 68101/digits dominating the aggregate. The
68101/atomic pair remains at zero joint success on every group/cell in both
arms, including fresh histories with training keys. Its last-500-update NLL
is about 0.72 and BOTH-correct training success about 0.2%, consistent with
an unresolved fitting problem under this budget. The other atomic mixed run
fits less well than its familiar control and regresses on every new-wording
primary. This does not establish a universal encoding advantage, a neural
mechanism or a need for a larger model. Encoding is a bundled representation
intervention, not a tokenizer replacement.

The mixed digit candidates are close to the gate, but failures remain:
68101/digits passes 89/90, missing training-key F2Q0 joint4 (102/128 versus
103 required); 68105/digits passes 86/90, missing three training-key query-2
joint4 cells and development F2Q2 (204/256 versus 205). No failed endpoint is
waived or successful initializer selected. All four pre-V S-as-V controls
score zero joint4/joint8. The fresh paired control and four-view training
augmentation differ from the preceding bridge study, so cross-study changes
are not isolated training-recipe effects.

## Execution and shared evidence

Model size stays **839,248 registered parameters**; original model code (52
files) is unchanged. All eight runs have 96,000 supervised answers and
9,120,000 input bytes each: 48,000 new updates, 768,000 answers and 72,960,000
input bytes overall. Persistent slow weights remain unchanged during
evaluation; normal within-prefix virtual memory updates stay active.

Executed source is `ae55882b86edc838d2768dca48b8caf8516c5648`. Slurm
13933115 is COMPLETED / exit 0:0, elapsed 3,368 seconds, two RTX A6000s:
**1.8711 allocated GPU-hours**. The original pending 4h request was reduced to
2h before any neural execution, with preserved source/metadata and independent
resource rebind. The actual-time runtime gate passes. All terminal hashes
lock at 19:34:56 UTC before reserved-wording execution. No job remains active.

117 scoped tests pass. Independent audit checks all 350,208 confirmation and
1,152 engineering full-256 vectors, exact data/label joins, lock timing,
training budgets, batch coverage and input collisions. Coordinator joins
match all 18 contrast arrays/intervals, 216 normal cells, 48 controls and 720
gate decisions exactly. Actual tensor checks pass all eight terminal models
and optimizer states: fixed parameter count, finite tensors, changed memory
weights and 6,000 AdamW steps. Each terminal's same-environment one-query
checkpoint reload is exact; an additional selected-batch GPU reload was not
run before the allocation closed.

The prespecified Windows CPU proof passes all 14,592 selected actual GPU
queries / 912 batches across all 12 capabilities. Argmax is identical; max
absolute logit difference is 0.00014687, full-probability difference 0.00001525,
within frozen .002/.0002 limits. This is selected-batch portability, not
all-query or bitwise cross-device equivalence. Four numerical figures,
source tables, individual cases and qualifications are in
[results/format_training_2026-11-16](../results/format_training_2026-11-16).

The artifact root contains all eight diagnostic capability cards in
`shared_checkpoint_registry.json`, exact inputs, copied initializers, complete
weights/optimizers, raw distributions, independent review, source and recovery
manifests. The future consulted-history union contains **985,652** signatures.
New family2 wording is now consulted evidence; it cannot be reused as an
untouched wording confirmation in later adaptive studies.

## Next Member C decision

Keep the qualified typed-memory reference available to A/B. Before harder
sequence retention or enlargement, test fitting stability at this same size:
hold mixed wording/data/supervision fixed and compare constant learning rate
with a prespecified 500-update linear warm-up to 3e-4, retaining all four S initializers and
matched 6k budgets. The large initial value loss, atomic fitting floor and
unfinished mixed atomic learning curve motivate this optimization diagnostic;
they do not prove its mechanism. Freeze fresh histories and a new wording
family before a future confirmation. This follow-up is planned, not run here.

The team should settle whether the primary memory claim needs digit and
atomic representations plus broad paraphrase transfer, or a canonical digit
interface. That scope choice must precede a new study; it cannot reclassify
the current 0/8 all-format admissions. Native-32 acquisition, persistence
after attention/buffer exit, and intended sleep/Hebbian sequence adapters
remain unqualified. No report, GitHub push or Google Doc edit is made here.

## Frozen prospective plan

Member C compares familiar-only versus mixed fact/query wording at the same
839,248 parameters and attention length 128. All four exact 20k natural-selector
checkpoints (68101/68105, digit/atomic keys) initialize both arms. Each arm gets
6,000 new ordinary-value updates, AdamW 3e-4 / weight decay .01, clipping 1,
and a reset optimizer. Eight runs total 48,000 new updates. Every run receives
96,000 answers and 9,120,000 input bytes (16 queries of 95 bytes per update).
No index/oracle is supplied to the ordinary-value model.

The semantic histories, counterfactual views, query ranks and targets are
identical within each pair. Familiar uses F0Q0 throughout; mixed uses exactly
two contexts per F0Q0/F0Q1/F1Q0/F1Q1 cell per update. Independent fixed RNGs
generate semantic worlds, views and cell permutations. All world four-view
closures reject the previous 790,004-history union; engineering training,
engineering evaluation, main training and confirmation are mutually disjoint.

New family 2 wording is frozen before model execution: facts `entry {key}
stores {value}.` and `{key}: stored code is {value}.`; queries `lookup code
belonging to {key}: ` and `report {key}'s stored code: `. Exact newline/padding
bytes live in the frozen protocol and renderer. Family 2 never enters
engineering neural execution or training. Confirmation model outputs require
a lock binding all eight complete terminal checkpoints and four initializers.
Use terminal weights only; no result-driven adaptation or dropped weak cases.

Confirmation uses 256 development-key and 128 training-key worlds, all nine
fact/query family cells, all four order/value views, both requested keys, and
original-F0Q0 no-facts/cue-erased controls. Twelve V capabilities (eight finals,
four initial S weights evaluated as V) yield 350,208 full-256 score vectors.
Three co-primary mixed-minus-familiar joint4 contrasts are new facts F2Q0,
new queries F0Q2 and both F2Q2 on development keys. Average four fixed paired
case differences per world before 5,000 shared PCG64/92004 world resamples.
Use exact 1−.05/3 percentile coverage, linear interpolation and tails .05/6.
An endpoint requires a positive adjusted lower bound; a broad three-shift
claim requires all three. Inference conditions on fixed cases/training data.
Other cells, familiar cost, joint8, per-case and initial scores are descriptive.

Qualification retains 80%: each terminal must reach 205/256 development and
103/128 training both-correct in all nine normal cells and four views, plus
joint4 in each group/cell. No failed endpoint is waived; nonsignificance does
not establish unchanged familiar recall. This is visible-context acquisition,
not native-32 persistence, retention or final A/B method efficacy.

Insomnia is connected through the existing SSH alias after session network
permission. Request at most two GPUs / four hours (8 allocated GPU hours),
four CPUs and 16G RAM. Engineering has two independent 64-update runs; its
timings must project the entire main train/evaluation workload within the
remaining allocation before main starts. No automatic renewal or sample/run
reduction. Root owns Slurm and all neural work; reviewer runs data/math only.

Artifacts: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-format-training-2026-11-16`.
Work: sibling `work/clmm-format-training`. Protocol, log and SESSION identify
actual state; never infer completion from this prospective document. Branch
`sequence-format-training` preserves the original baseline tag and the existing
SLEEP_MULTIREVISION edit. No push, teammate message or Google Doc edit occurs.

Resource update before any model execution: same pending job13933115 was
reduced from4h to2h after the long request slipped intoNovember18. Source
ae55882 reads actualSlurm time and retains the complete-workload gate; a
resource-parser test passes. All training/data hashes are unchanged and
independently rebound to the new source. Current packet metadataSHA is
38f92aade1fcb91b6a5da4294478cfdf9ec9fa5439eaa064f3c30c674d967dc6.
