# Independent prospective protocol review

Reviewed 2026-11-15 before seeing any outcomes for this experiment. This is a design review, not a completed artifact audit. Reviewed the main handoff, `SleepSession`, `StreamingReplay`, the old key/value stream generator, and the local-delta update source at repository HEAD supplied by the coordinator. No model inference, training, accelerator allocation, or checkpoint selection was performed by this reviewer.

## Estimand and controls

The proposed comparison isolates placement of an identical sequence of extra literal writes through the same fixed learned model. Continuous and periodic replay must share real arrivals, block-start FIFO snapshots, PCG64 draws, selected source IDs, literal key/value pairs, and selected order. Only the position of extra writes among actual arrivals changes. Repeat-current matches total write opportunities and online supplies the lower-write anchor. Logical write matching is not a claim of matching end-to-end latency or FLOPs.

The coordinator refined the query set before any outcomes: six original targets, comprising early stable positions 0/1, initially FIFO-eligible stable positions 12/13, and revised positions 14/15. The stable co-primary endpoint averages all four stable targets, with early and eligible groups separately reported. The common neutral K16 gate requires at least 80% of histories to answer all six original targets correctly. All six current query values remain distinct after every genuine change. This refinement addresses the reviewer observation below and does not select on model performance. D64/D128 add one separate recent-distractor query; it never enters either co-primary endpoint.

The analysis estimates performance over the sampled paired histories conditional on two fixed learned parents, the fixed content map, and the five fixed random maps. The two parents and five random maps are not extra independent history samples. The proposed balanced aggregate is the mean over the two parents of `0.5 * content + 0.5 * mean(all five random maps)`, then mean across both phases and paired histories. This differs deliberately from an unweighted mean over all twelve checkpoints.

## Requirements to freeze before outcomes

1. Define the full real-arrival sequence and all block boundaries. Three eight-observation revision windows separated by sixteen distractors only between consecutive windows imply 56 post-prefill observations, hence total writes of 72 + D online and 128 + 2D in each extra-write arm. D=0 means no final distractor tail; it is not an acquisition-only baseline.
2. Distinguish the common checkpoint admission gate from arm-specific D0 outcomes. A neutral online-only gate with all genuine revisions in the prefill can qualify fixed checkpoints without selecting on periodic or continuous performance. Report every arm's D0 failures and use the declared revision/interference interpretation when full retention admission fails.
3. Preserve every checkpoint, map, phase, history, and pressure cell. No gate-based removal of continuous/periodic damage and no favorable-map selection.
4. At each genuine revision, the new value must differ from every earlier value for that key. Preserve both any-obsolete and immediate-penultimate error indicators; these answer different questions after three changes. All six queried current values remain distinct at each revision stage, making reset/rolled controls interpretable.
5. Use actual arrival time to update the evaluator's world state. A frozen snapshot is allowed to become stale within its block. Its literal value must never be replaced by a future oracle value. The effect of stale snapshot revival is part of this timing intervention.
6. Keep the 16-fact prefill entirely online-only for every arm; scheduler priming must not inject extra writes. Evaluate endpoints after the terminal finish flush, and save the closed counters. All proposed cell lengths are multiples of four, but the audit should still assert this and verify no hidden flush writes.
7. Record target availability, selected target replay probability, buffer residence, stale extra writes at application time, and exposure after the final target-valued write. A source ID alone is insufficient to authenticate a replay; audit its key/value literal too.
8. Resample 128 history IDs jointly across all phases, delays, arms, parents, and maps using the declared 5000 PCG64 draws/seed. Two two-sided 97.5% intervals give at least 95% simultaneous coverage across the two declared endpoint contrasts by Bonferroni. Report other comparisons with their stated secondary uncertainty; do not use a secondary result to rescue a failed co-primary result.

## Mechanism and interpretation traps

Stable queried keys at initial positions 0 and 1 are outside the latest-eight FIFO after the 16-fact prefill and never refreshed in the proposed arrival stream. They cannot receive direct stable-target replay. Differences in their retention measure interference and ordering from writes to other keys. Stable positions 12 and 13 are initially in the FIFO and can receive direct replay while resident. Report these two groups and actual replay exposure separately; initial eligibility does not guarantee that a target was sampled. The primary stable average deliberately includes both groups.

The revised keys at positions 14 and 15 intentionally enter the buffer late. Their old values are explicitly refreshed before each change, so the window tests a snapshot that has had lawful access to the old world. This is a designed stale-replay challenge, not a representative frequency estimate for all naturally occurring revisions. Show phase-level outcomes because phase 0 and phase 2 alter how many current-world writes precede stale replay.

With local-delta erasure, same-key and overlapping-key writes can be noncommutative. Any periodic-minus-continuous effect is an ordering effect in this fixed reference implementation. It does not establish biological sleep, general continual learning superiority, or variability of training across new initializers.

Each content/random checkpoint was separately trained under its own routing map. The within-checkpoint timing comparison holds every slow tensor fixed, but a descriptive content-versus-random difference also includes the learned tensor differences induced by that earlier training. It does not isolate a map-only routing intervention. The fixed random family remains a useful control family for the timing effect.

The query/current-value distinctness guard does not imply all sixteen world values are unique. Classify obsolete errors per key. A prediction equal to another key's current value is not necessarily a valid prior version of the queried key.

Whole-history signature separation is necessary for the declared split but does not demonstrate token-level or codebook novelty. The study reuses the same closed key/value alphabet and learned representation. Preserve the full exclusion provenance for training 90502 and 90802, and evaluations 90723, 90823, 90923, and 91023; include the history's full stream in the signature rather than only seed labels.

## Planned independent artifact checks

Reconstruct the dataset constraints and exact stream prefixes; recompute query targets from last real arrivals; independently reconstruct scheduler writes and FIFO occupancy; compare C/P literal sequences and budgets; verify predictions against saved score-vector argmaxes and targets; recompute any-obsolete/penultimate errors; reconcile checkpoint inventory and hashes; recompute balanced primary/secondary contrasts with paired bootstrap; create figures from saved source tables. This is a full saved-artifact arithmetic audit, with numerical model inference left to the coordinator's separate validation.
