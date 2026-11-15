# Independent bounded-freshness interpretation

All 12 fixed C9 step-1200 local-delta checkpoints are included, with no added optimization or model/map selection. The independent reviewer regenerated the data and reconstructed saved scores, event plans, refresh/fallback fields, query diagnostics and resource arithmetic without importing the model/evaluator or running inference. The source commit is `9ab3901860548d2d87a9afc61e8ce58f950a03ce`.

## Co-primary result

| D128 category | Bounded accuracy | Literal accuracy | Bounded minus literal (97.5% CI) |
|---|---:|---:|---:|
| Stable | 83.28% | 83.30% | -0.01 pp [-0.13, +0.06] |
| Revised | 93.93% | 83.46% | +10.47 pp [+7.76, +13.28] |

Revised accuracy improves relative to literal replay in this fixed comparison. The stable endpoint interval covers zero; it is not an equivalence test.

Each endpoint equally averages both schedules, both phases and the two fixed parents, with 0.5 weight on content routing and 0.5 on the mean of all five fixed random maps. Intervals use 5,000 paired whole-history PCG64 bootstrap draws (seed 91204), jointly retaining every condition; the two 97.5% intervals implement the declared Bonferroni adjustment. The 128 histories are the resampling units. These intervals condition on the frozen checkpoints and maps; they do not estimate training-seed or routing-map population uncertainty.

## Admission and repair

Common short/neutral admission passes for 12/12 checkpoints. The own-D0 six-target threshold fails in 48/144 arm/phase/checkpoint cells; all remain in the analysis. Overall classification: **revision/interference diagnostic; all conditions retained**.

| Arm | Own-D0 failures / 24 | Minimum D0 six-target all-correct |
|---|---:|---:|
| online | 0/24 | 89.84% |
| repeat | 0/24 | 89.84% |
| continuous | 24/24 | 35.16% |
| periodic | 24/24 | 28.12% |
| continuous_bounded | 0/24 | 86.72% |
| periodic_bounded | 0/24 | 86.72% |

D0 revised repair (secondary, nominal 95% CI): +34.77 pp [+29.98, +39.55]. D0 concerns repair under replay/arrival order, so it must be shown alongside later retention rather than treated as a clean retention effect.

## Secondary detail

| Family | D128 stable freshness | D128 revised freshness |
|---|---:|---:|
| content | -0.10 pp [-0.27, +0.00] | +10.60 pp [+8.11, +13.23] |
| mean5random | +0.07 pp [+0.01, +0.14] | +10.34 pp [+7.96, +12.73] |

D128 early-stable freshness: +0.06 pp [+0.00, +0.13]; initially eligible stable: -0.09 pp [-0.27, +0.02]. Early keys receive zero direct replay writes under all four replay policies, so early-group changes concern interference preservation rather than rehearsal.

| Category | Continuous freshness, D128 | Periodic freshness, D128 | Timing × freshness, D128 |
|---|---:|---:|---:|
| stable | +0.01 pp [-0.05, +0.06] | -0.04 pp [-0.17, +0.05] | -0.05 pp [-0.13, +0.00] |
| revised | +8.23 pp [+6.08, +10.45] | +12.71 pp [+9.83, +15.69] | +4.47 pp [+2.84, +6.32] |

| Bounded arm minus anchor, D128 | Stable | Revised |
|---|---:|---:|
| continuous_bounded minus online | -1.45 pp [-2.01, -0.92] | -0.56 pp [-1.38, +0.16] |
| continuous_bounded minus repeat | -1.45 pp [-2.01, -0.92] | -0.56 pp [-1.38, +0.16] |
| periodic_bounded minus online | -1.77 pp [-2.43, -1.13] | -0.62 pp [-1.50, +0.16] |
| periodic_bounded minus repeat | -1.77 pp [-2.43, -1.13] | -0.62 pp [-1.50, +0.16] |

Both bounded arms have lower stable accuracy than the online and repeat-current anchors. Revised differences versus those anchors have intervals covering zero. The primary benefit repairs an avoidable weakness of literal replay; it does not establish a replay advantage over these simpler anchors.

All secondary intervals are descriptive 95% intervals. Per-checkpoint/map/phase/delay outcomes, repeat-current and lower-write online anchors, obsolete-any-version errors and query exposure diagnostics remain in the CSV tables. They were not used to select a checkpoint, routing map or primary contrast.

## What is matched and what changed

All 768 pressure history/cell candidate sequences match across the four replay policies, and literal continuous/periodic have identical applied extra source/key/value order. Bounded continuous/periodic differ in 3,403 applied sources and 849 values because their FIFO lookups happen at different causal times. Calling bounded timing conditions identical-content would be incorrect.

Bounded replay has zero stale-at-emission extra writes in this capacity-eight/block-four grid. Fallback is applied and counted rather than dropping writes. Zero staleness here does not establish a guarantee under arbitrary larger blocks or eviction patterns. An extra write can be obsolete relative to a final target while still current when emitted; those observer labels are kept distinct. Source-only refresh changes provenance but gives the model the same key/value write; changed values are the effective freshness intervention.

The model, real arrivals, candidate-key order, write counts, adaptive trace bytes and retained record capacity match within each comparison. Applied values intentionally change. Bounded configuration adds fixed schema/freshness metadata, so serialized controller byte counts need not be identical. These matches do not imply identical FLOPs or total evaluator process memory.

## Provenance and resource qualification

The launch union contained 89,093 signatures from both exact training arrays and the prior `episodes.json` scan. A broader postlaunch structural search found one prior seed-90323 integration history embedded in two `verification.json` files. Its signature was absent from that union. The corrected coverage is 275 numeric-history JSON files: 273 numeric episode files plus two verification files; one additional episode JSON contains semantic byte-task data and no typed numeric histories.

The original launch packets and outcomes are unchanged. Adding the one missing signature to the main launch union gives 89,094 signatures and zero overlap with all 1,280 main histories. Separately, the engineering launch union plus that signature contains 88,934 signatures and has zero overlap with its 160 histories. The main exclusion union intentionally contains the engineering histories. Independent generation under each expanded union reproduces every accepted stream exactly. This is a documented launch-provenance coverage omission, with no observed data leakage or outcome change. Exact source paths/hashes, supplementary signature and reusable launch-union superset are saved in `supplemental_exclusion_audit.json`.

The runtime-only preflight padded projection was 2474.13 seconds, below the prospective 3,600-second continuation threshold. Engineering efficacy was not inspected. The main pipeline used two CPU workers with one Torch thread each and a 4,200-second hard limit. All 433,152 saved score vectors, 13,192,704 chronological writes and 4,608 resource batches passed arithmetic checks; this reviewer performed no model inference.

## Claim boundary

Correctness here is defined by the latest real observation for a key. Refresh is a causal recency policy under that authority assumption, not a learned test of whether an observation is trustworthy. No noisy, adversarial, delayed-authority or source-conflict observations were evaluated. The split excludes exact whole histories; the 64-key/16-value typed vocabulary is intentionally shared. The main split was fresh when declared and is now consulted evidence for this fixed study; it should not be reused as untouched selection-free evidence for a later intervention.

The component has 81,921 slow parameter elements and a 65,536-byte fast trace per stream. That is an inspectable capacity for this controlled finite KV diagnostic, not evidence for full Titans/MAC/MAG/MAL, language-scale transfer or biological novelty. The primary Titans v1 section 5.1 reports 170M–760M parameter models trained on 15B–30B tokens: https://arxiv.org/html/2501.00663v1#S5.SS1. Existing complete sequence-model acquisition and held-out assignment/template gates should be addressed before making a full-project retention claim or choosing a larger model size from these component outcomes.
