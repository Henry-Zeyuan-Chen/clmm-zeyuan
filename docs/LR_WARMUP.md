# Constant LR versus 500-step warm-up — prospective, November 16

Member C tests fitting stability without resizing the 839,248-parameter model.
Both arms restart from all four exact 20k natural-selector checkpoints
(68101/68105 × digit/atomic keys), with empty AdamW states. Both use exactly
the same mixed-format arrays, query order and labels for 6,000 updates:
96,000 answers / 9,120,000 input bytes per run. All eight cases are retained.

Constant LR is 3e-4 throughout. Warm-up uses 3e-4×min(update/500,1), indexed
from update 1: first LR6e-7, peak on update500, unchanged thereafter. There
is no LR-area compensation. Nominal sums are1.8 and1.72515; AdamW decay also
depends on LR. Interpret any result as an optimizer-schedule recipe effect,
not proof of isolated early gradient shock or backbone drift. Weight decay,
clipping, supervision, registered parameters and optimizer-update counts match.

Training mixes F0Q0/F0Q1/F1Q0/F1Q1, two contexts each per update. Independent
fixed generators draw semantic worlds, counterfactual views and cell assignments.
Every four-view key/value history rejects the prior985,652 signatures and
engineering/main/confirmation closures are mutually disjoint. Individual facts
may recur. First-minibatch logits are saved; paired initial-function differences
must be ≤1e-5 before main and again before confirmation.

New family3 facts are `item {key}: its code is {value}.` and
`{key} is paired with code {value}.`; queries are `fetch code tied to key
{key}: ` and `{key}: what code is stored? `. Exact newline, padding and alias
bytes are frozen in protocol/data. Episode invariants and original template
tables stay intact. Previously consulted family2 is omitted from this study.
Family3 never enters training or engineering neural execution; confirmation
requires all eight terminal6k states to complete and lock their hashes.

Confirmation uses256 training-key and256 development-key fresh worlds, nine
cells over families0/1/3, all four order/value views, both requested keys and
original-F0Q0 no-facts/cue-erased controls. Twelve capabilities (eight finals,
four S-as-V initial controls) yield466,944 full256 score vectors. Initial
controls are pretrained selectors, not random/untrained networks.

Three co-primaries compare warm-up minus constant joint4: train-key F0Q0
(fresh-assignment fitting), dev-key F0Q0 (key generalization), and dev-key F3Q3
(new fact+query wording). Average four fixed paired differences per world,
then use5,000 world bootstrap draws: PCG92205 for train,92204 for dev; matrices
are shared within group and independent across groups. Exact linear quantiles
.05/6 and1−.05/6 give98.3333% intervals. Each endpoint needs a positive lower
bound; all three are needed for a broad fitting+transfer benefit. Other
endpoints are descriptive95%. Fixed-case inference does not estimate training
procedure variance or establish uniform improvement/equivalence.

Qualification keeps the80% rule:205/256 BOTH-correct in each cell/view and
joint4 for all nine cells and both groups,90 gates per terminal. No waiver or
best-case selection. Current all-format failures remain failures. Native-32,
stateful retention and final sleep/Hebbian adapters are separate future work;
qualified typed references remain available for A/B.

Another user project has one GPU under job13933883; leave that allocation
untouched. This study requests a separate one-GPU/three-hour allocation with
two CPUs and8G RAM, keeping total user GPUs within the current two-GPU limit.
The pipeline supports serial workers, uses actual Slurm time with a3-minute
reserve, and must project the full48k-update/evaluation workload before main.
No automatic renewal, case reduction or outcome-driven tuning. Local artifact
cap3GB; remote6GB. Root owns all source/GPU work; reviewer uses data/math only.

Artifacts: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-lr-warmup-2026-11-16`;
work: sibling `work/clmm-lr-warmup`. Branch`sequence-lr-warmup`. SESSION and log
record observed state; this prospective note does not establish completion.
Preserve the existing SLEEP_MULTIREVISION edit and baseline tag. No push,
report draft, teammate message or Google Doc edit belongs to this task.
