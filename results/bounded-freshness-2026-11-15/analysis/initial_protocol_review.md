# Prospective bounded-freshness review

Reviewed November 15, 2026 from the coordinator's preimplementation task packet. This is an independent protocol review; no current engineering or main-study efficacy outcomes were inspected. Reviewer owns only this analysis directory, and runs no model inference, training, or GPU jobs.

The question is whether resolving each originally sampled key from the current eight-record real-arrival FIFO, with original-record fallback on eviction, improves stable or revised retention relative to frozen literal replay under matched write/storage budgets. Both schedules use a four-record block-start snapshot, the same private PCG64 draws, and unchanged candidate order. All 12 fixed C9 step-1200 local-delta checkpoints remain included. There is no new optimization or checkpoint selection.

## Required matching distinctions

- Literal continuous/periodic can share the identical applied extra source/key/value sequence while differing in placement.
- Bounded continuous/periodic share sampled candidate records and sampled-key order. Their actually applied source/value sequences can differ because they resolve at different causal times. This is the intended timing by freshness interaction, not a violation of the planned match.
- Freshness must be resolved after the real arrival at that emission clock, including the terminal partial-block flush. Replay must neither refresh FIFO recency nor mutate the sampled candidates. No retained full-history/current-version map, query labels, skipped writes, or new draws are permitted.
- The comparison matches write opportunities, retained tensors and controller record limits. It does not automatically match FLOPs, interpreter overhead, or information content.

## Estimand and uncertainty

Co-primary endpoints are balanced-family bounded-minus-literal stable and revised accuracy at D128, averaged equally over two schedules, two phases and two fixed parents. Balanced family means 0.5 content plus 0.5 mean of all five fixed random maps. The paired experimental unit is the whole generated base history; resample the same 128 history indices jointly across every arm, delay, phase, checkpoint and route map. Use 5,000 PCG64 bootstrap draws at seed 91204 and 97.5% percentile intervals for each endpoint, giving the declared Bonferroni two-endpoint family. Other intervals are descriptive 95% intervals unless explicitly specified.

These intervals describe conditional history uncertainty for the frozen checkpoints/maps. Two parent checkpoints and five fixed random maps do not support population training-seed/map uncertainty. A positive mean does not show that every schedule, phase or checkpoint benefits; those disaggregations are secondary and must remain available.

## Interpretation safeguards

Common short/neutral gates and each arm's own D0 threshold must remain unchanged, and all failures stay in reported data. D0 revised effects concern repair under arrival/replay order; weak D0 disqualifies a clean retention claim for the affected condition. Stable early keys at original positions 0/1 are evicted before replay eligibility, so their outcomes concern interference preservation rather than direct rehearsal. Stale-at-emission and obsolete-relative-to-final-target are distinct labels; all old versions count as obsolete errors, not only the penultimate value.

Engineering seed 91203 N16 may determine runtime only. Main seed 91223 N128 must exclude both prior training arrays and all prior numeric KV episode files, including the preceding 91123 split and engineering artifacts. Exclusion reconstruction, generation, raw score argmax/MSE, chronological plans, gates, resources, source/checkpoint hashes and failed-run retention require inspection after the implementation is frozen.

## Model size and scope

81,921 learned slow elements and a 64 KiB persistent fast trace can be enough to test this controlled finite typed-KV mechanism if acquisition/admission holds. Model size alone does not invalidate the component comparison, and it does not establish full sequence-model behavior. The implementation is a reference associative component, not a full Titans/MAC/MAG/MAL reproduction. The Titans v1 primary source reports 170M/340M/400M/760M parameter scales and 15B/30B-token training in section 5.1: https://arxiv.org/html/2501.00663v1#S5.SS1 (checked November 15, 2026). Language-scale transfer and architecture integration remain untested here.

No prospective blocker found in the supplied design. Final assessment remains conditional on the actual patch, packet, runtime gate and saved raw outputs.
