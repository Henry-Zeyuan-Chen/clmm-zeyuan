# Matched format training — in progress, November 16

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
