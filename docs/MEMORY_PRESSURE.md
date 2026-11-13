# Memory-pressure experiments, 2026-11-13

This local `memory-pressure` branch extends the replay pilot with more facts, longer delays, and an update-matched control. It is an exploratory associative-memory study supporting the proposal's replay direction. It does not complete the paper-style Titans baseline in `ROADMAP.md`, settle the ambiguity register, or establish a sleep-specific advantage.

## What changed

`scripts/research_memory.py` now accepts configurable key/value vocabularies while preserving the old defaults. `scripts/train_memory_pilot.py` adds an explicit episode-length curriculum. `scripts/memory_pressure_probe.py` implements the paired pressure experiment. The vendored memory engine is unchanged. The implementation used for all three pressure grids is commit `d93ac37`.

The model has 256 keys, 8 possible values, width 64, a two-layer memory, and 62,730 registered parameters. Each condition activates 16, 64, or 256 distinct facts and applies 0, 64, 256, or 1,024 further observations. There are 256 episodes per fact count per checkpoint; delay conditions share prefixes. The three checkpoints see the same evaluation episodes. Different fact counts use different streams.

Every episode starts with one observation of each active key, followed by two target revisions. Four old target keys are then withheld from the observation stream. Their last-observation ages are matched across fact counts. Remaining keys recur with fixed final values, separating retention pressure from the stale-label replay failure in the earlier pilot. Four recent distinct keys are also queried. These keys were already observed: this measures recent-key recall, not acquisition of genuinely new facts.

## Matched comparison

All arms share the checkpoint, observations, reservoir schedule, and common online-only prefill. Outer parameters are frozen. Reads receive only query keys and fast weights.

| Arm | Writes during the delay |
|---|---|
| Online | One ordinary write per observation; lower-compute anchor |
| Repeat | Ordinary write plus an immediate repeat of the current fact |
| Continuous replay | Ordinary write plus one reservoir sample |
| Periodic replay | Eight ordinary writes followed by the same eight reservoir samples |

The final three arms receive exactly `K + 2 + 2D` memory writes at delay D, compared with `K + 2 + D` online. Auxiliary writes disable forgetting in all three arms. Continuous and periodic replay receive identical frozen tuples in the same order. Runtime is recorded, but equal update counts are not a promise of equal wall-clock cost.

The reservoir samples observation events using Algorithm R. It holds 32 working pairs plus an eight-pair block snapshot: a maximum of 40 logical pairs. Repeated keys can occupy several slots, and resident copies are refreshed when a key is observed. The snapshot is discarded before queries. Saved evaluator traces are larger than the policy budget and are not accessible to retrieval; this is not a 40-pair physical-memory claim.

## Training and evidence status

The first 400-step recipe, with an inner-step cap of 0.1 and constant 256-observation episodes, failed with a nonfinite loss at step 140. The failed run is preserved. A recorded amendment reduced the cap to 0.01 and used 100 steps at length 16, 100 at length 64, and 200 at length 256. Each successful run used batch size 64 and exactly 3,788,800 training observations. Both cap and curriculum changed together, so their separate effects are unknown.

The two originally specified seeds reached 904/1,024 (88.28%) and 356/1,024 (34.77%) heldout recall. A third seed was added through an explicit amendment after training variability became apparent; it reached 883/1,024 (86.23%). Resetting or exchanging fast memories reduced recall to approximately the 12.5% chance level. All successful final checkpoints and the failed run are retained. The added seed cannot replace either original seed in the continuation rule.

The primary comparison was fixed before the pressure results: 64 facts, delay 256, on old targets absent from the working reservoir at query but previously selected by the shared replay schedule. This subset contains 362 of 1,024 old queries in 209 of 256 episodes for every checkpoint. Intervals below use 5,000 paired episode bootstrap draws, recomputing the ratio of summed correct and eligible queries. They are descriptive intervals conditional on each checkpoint, not uncertainty across training seeds.

| Training seed | Primary periodic recall | Primary repeat recall | Difference, percentage points [95% interval] | Recent recall difference |
|---|---:|---:|---:|---:|
| 48101, original | 362/362 (100.0%) | 317/362 (87.6%) | +12.43 [9.24, 15.88] | 0.00 |
| 48103, original | 77/362 (21.3%) | 65/362 (18.0%) | +3.31 [-0.29, 6.94] | -8.89 [-11.23, -6.54] |
| 48105, added sensitivity seed | 358/362 (98.9%) | 304/362 (84.0%) | +14.92 [11.17, 18.85] | 0.00 |

The two stronger checkpoints reached 1,024/1,024 recent recall in every arm at the primary cell. The weak checkpoint reached 467/1,024 with periodic replay, 558/1,024 with repeat, and 687/1,024 online. The original continuation rule required at least a five-point primary gain with a positive lower interval and at most a two-point recent-recall loss in **both original checkpoints**. That rule failed. No fresh-stream confirmation was triggered, and the third seed does not change this decision.

The advantage is not confined to this subset: across all 1,024 old queries at the primary cell, periodic/repeat/online correct counts are 967/902/938, 217/166/182, and 935/864/901 for seeds 48101/48103/48105 respectively. At 256 facts and delay 1,024, the stronger checkpoints retain only 30.3% and 28.2% of all old targets with periodic replay (online: 22.8% and 21.8%). Thus the harder task exposes substantial forgetting even when replay helps. The weak checkpoint is near chance at long delays.

Periodic versus continuous replay differs by only +1, +1, and -1 correct answer out of 362 in the primary subset. This is evidence for using extra updates on past facts in some trained checkpoints; it does not establish a useful benefit from sleep timing. The online anchor remains necessary because repeating the latest observation can itself hurt retention.

A separately labeled posthoc check requires at least 64 additional observations since the end of the last block selecting the target. At the primary cell, the shared subset has 339 queries from 204 episodes. Periodic-minus-repeat differences are +12.68 [9.30, 16.20], +2.65 [-0.88, 6.16], and +15.34 [11.48, 19.48] points. This reduces concern that the first result is solely immediate rehearsal, but selection history and residency are descriptive strata, not randomized mechanisms. The raw stable/revised difference also mixes revision with different initial observation counts.

These are small synthetic memory experiments with frozen outer parameters, online-only training, and no language-model or MAC/MAG/MAL reproduction. The strong/weak checkpoint association does not establish a causal explanation for replay sensitivity. Multiple grid cells were explored; no pooled-seed or multiple-comparison-adjusted significance claim is made.

## What to do next

1. Stabilize the training recipe across fixed seeds before scaling the model. Compare training curricula under equal observation and outer-update budgets, retaining failures. The current seed association does not prove why replay works better in one checkpoint.
2. Run the saved auxiliary-forgetting diagnostic on the weak checkpoint. This changes only auxiliary decay from zero to the learned value, with the same original stream prefixes and online predictions as a check. It is a posthoc sensitivity test, not fresh confirmation. It was planned but not run before the allocation expired.
3. Test replay-aware training against online training with explicit observation and update budgets. Continue reporting both old retention and recent-key recall, and use fresh heldout streams once the policy is fixed. Reintroduce revisions only with a causal stale-value control.

Team choices remain: whether the project claims generic replay benefits or a distinct sleep mechanism; how to budget auxiliary forgetting, buffer capacity, and training compute; and which reference/retrieval conventions from the ambiguity register to adopt. Current settings are recorded experimental choices, not team-wide decisions.

## Verification and artifacts

The 91 reference, replay, vocabulary/curriculum, and pressure tests pass locally and on Insomnia. The pressure tests cover causal sample access, reservoir capacity, identical replay tuples, update counts, immutable query state, target age, and legacy behavior. Independent analysis reconstructs labels, Algorithm R admission, replay selections, residency, exposure, budgets, raw counts, and paired episode intervals. It checks saved predictions rather than independently rerunning neural inference.

Local checks used Python 3.13.3 and CPU Torch 2.15; GPU runs used Python 3.11.5 and Torch 2.11.0+cu128 on one RTX A6000. Exact environments are saved. This GPU environment is not the exact Python 3.13 lock environment in `SETUP.md`. All pressure jobs completed before Slurm allocation 13890691 reached its one-hour limit; no allocation was renewed.

The `baseline-v0.6.0` tag still resolves to `35a549f137548d48aecc26e33c6f8a34a6efc97c`. Work remains on a local branch; no push, merge, or PR is claimed. Repository summaries are under `results/memory-pressure-2026-11-13/`. Full raw traces, checkpoints, failures, source snapshots, analyses, figures, logs, and a Git bundle are under `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-pressure-2026-11-13` and its adjacent ZIP package.
