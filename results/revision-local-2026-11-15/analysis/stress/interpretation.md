# Exploratory old-value reinforcement stress

All 24 fixed trained checkpoints completed the exploratory follow-up successfully in **351.51 seconds of local CPU wall time**, below the global 600-second limit. No optimizer, new training, GPU allocation, checkpoint selection or model change occurred. The original trained joint primary criterion remains **failed**: this post-outcome stress study does not amend or rescue it.

The design was saved before stress inference in `design.json`. It was motivated by the observed short-revision ceiling. Each pressure history contains eight initial facts, old-value reinforcement R=1/4/16 for each of the two keys that will later be revised, one genuine conflicting observation per revised key, and D=0/64 later distractors. Reinforcement inserts exactly R−1 additional cycles of the original old bindings before the existing revisions. Original queries and distractors remain intact; no revision flag, query target or old-value oracle is passed to `write`. All methods receive identical extra observations.

All 128 base histories are paired across the six R/D variants, and all fixed parents/maps are retained. Joint rejection excludes any candidate whose six variants overlap the exact old/new training sets or any actual prior 90723, frozen 90823 or trained 90923 history. The independent audit reconstructed the streams and rejection rule exactly: 80,231 excluded signatures, 128 accepted base histories in 128 attempts, with zero overlap. All keys, values and bindings remain familiar; the novelty is the observation sequence.

## Declared exploratory focus: R16 D64

Differences are local delta minus EMA, in percentage points. Intervals are nominal 95% paired-history percentile intervals: 5000 PCG64 seed 91004 draws, resampling the 128 whole-history IDs jointly across the two fixed initializers and five fixed random maps. These are exploratory intervals, without confirmatory multiplicity adjustment or population uncertainty across training seeds/maps.

| Allocation | Endpoint | Local delta | EMA | Difference | Nominal 95% interval |
|---|---|---:|---:|---:|---|
| Content | Revised queries |99.61%|0.39%|+99.22|[+98.05,+100.00]|
| Mean5random | Revised queries |98.87%|2.03%|+96.84|[+95.70,+97.81]|
| Content | Stable queries |96.68%|18.36%|+78.32|[+73.63,+82.81]|
| Mean5random | Stable queries |97.58%|22.46%|+75.12|[+72.58,+77.66]|

Both route families pass the exploratory revised-query improvement criterion (lower bound >0) and stable noninferiority criterion (lower bound >−3 points). The baseline's D0 behavior is essential to interpreting these differences: at R4 and R16, **every EMA condition has 0% revised-query accuracy before any distractor delay**, while local delta is 100%. At R16 D0 all 3072 EMA revised queries return the obsolete value. EMA therefore has an immediate replacement failure under reinforcement, rather than a cleanly acquired revised memory that only fails later.

## Full reinforcement curve and qualification

The full curve is retained rather than selecting the strongest stress level. D64 revised-query accuracies are:

| Allocation and rule | R1 | R4 | R16 |
|---|---:|---:|---:|
| Content, EMA |30.66%|3.71%|0.39%|
| Content, local delta |99.61%|99.61%|99.61%|
| Mean5random, EMA |32.89%|4.38%|2.03%|
| Mean5random, local delta |98.87%|98.87%|98.87%|

At D0, R1 revised accuracy is 100% for content EMA and 99.80% for its random mean, then drops to 0% at R4/R16. Both local-delta families remain at 100% across R. Stable D64 EMA accuracy also declines with reinforcement: content 24.41→21.48→18.36%, random 30.16→26.21→22.46%; local delta stays at 96.68%/97.58% on these paired histories. Identical accuracies across R do not imply bitwise identical states or scores, nor independent replications: these variants share the same base histories.

All 24 conditions pass the original three short gates on the new 91023 histories. At R1 D0, the minimum EMA four-original-target all-correct score is 98.44%; local delta is 100% in every condition. At R4/R16 D0, EMA four-target all-correct is 0% in all 12 conditions; local delta remains 100%. These failures are preserved in `short_and_stress_d0_gates.csv`. Accordingly, the high-stress comparisons are **exploratory revision/interference diagnostics**, not an admitted EMA retention comparison. No gate was relaxed, and no failed map or parent was removed.

## Checks and scope

`raw_audit.json` verifies all 24 completed cases, 156,672 scorevectors, 1,078,272 causal write events, and 2880 resource/serialization batches. It independently reconstructs saved argmax/correct/obsolete/MSE/summary arithmetic, all online plans, query exposures, retained association/decay/projection counts, physical trace bytes and zero serialization errors. It verifies all input checkpoint SHAs against the complete trained registry, read/parameter/RNG-purity assertions, exact data construction and exclusions, driver snapshot SHA, and unchanged c9 source hashes against the previously verified Git blobs.

Actual driver SHA: `f7162bbf3d86be1da4dd663e52c2ca97d1434c1087e380ffc1876c35556a8ab3`. The exact driver snapshot, command, environment, design, episodes, source manifest, statuses and all per-case evaluation outputs are saved under `OUT/stress`; existing trained weights are referenced by SHA rather than duplicated. Numerical figures are generated from audited CSVs and their PNGs were visually inspected.

The stress result supports a specific failure mechanism: repeated old-value observations can dominate a later single EMA write, while key-local erasure supports replacement under this reference geometry. The compound local-delta rule also omits active bulk decay, so its late stable advantage must be interpreted with the earlier factorial controls. This is a controlled post-outcome follow-up in one familiar codebook, not a new confirmatory primary, a general revision benchmark, a biological novelty claim, or full Titans/teammate-method validation. Equal observations, retained module writes and persistent storage do not imply equal FLOPs. No additional follow-up was launched.
