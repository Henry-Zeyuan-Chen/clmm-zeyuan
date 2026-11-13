# Replay pilot: research state, 2026-11-12

This exploratory branch asks whether periodic replay can preserve old and revised facts while accepting newer information. It supports the proposal's sleep direction without settling the team's choice between proposals. The main roadmap and ambiguity register remain the authority for the eventual paper-style baseline.

## Implementation and scope

`replay-pilot` branches from the tested local `reference-impl` work. The new code lives in `scripts/research_memory.py`, `scripts/train_memory_pilot.py`, and `scripts/replay_probe.py`; invariant tests live in `tests/test_clmm_replay_probe.py`. The vendored engine is unchanged.

The probe freezes outer parameters and compares online writes, continuous replay, and periodic replay. Both replay schedules consume the previous eight observations once, in the same key order, per eight new observations. The logical bank plus arrivals holds at most 16 pairs. The evaluator retains the full stream for reconstruction, so this is a policy-capacity constraint rather than a measured 16-pair system footprint. Queries read fast weights only. Old targets have left both buffers at evaluated boundaries.

Two independent diagnostic options are available:

- `--replay-decay-scale 0`: remove forgetting only on replay writes, preserving normal observation updates.
- `--replay-value-source latest-observed`: resolve a replay key's value from the bank and observations that have already arrived in the current block. This preserves replay keys and counts, but available values can differ across schedules. Default `stored` preserves identical replay examples across schedules.

Neither option is a team-approved final sleep algorithm. These experiments establish controls and failure modes before an efficacy claim.

## Evidence so far

Three existing 24-dimensional linear-memory checkpoints were evaluated on 512 shared streams. Each stream contains stable and revised targets followed by 64 distractor observations. Four recent keys are queried per episode; uncertainty resamples episodes, not the four correlated questions independently.

At the final boundary, stored periodic replay achieved 56.6–58.3% recent accuracy, compared with 98.7–99.7% online. An independent reconstruction found that periodic replay wrote an obsolete conflicting value last for 820 of 2,048 recent queries. The models selected that obsolete value in 810–813 cases. Causal label correction recovered recent accuracy to 93.9–96.8%, still below online. Disabling replay forgetting substantially reduced old-target damage but did not establish a general improvement over online. Raw outputs, paired intervals, and independently reconstructed event histories are saved.

An exposure-matched revision control also narrowed the earlier interpretation: the extra revision penalty at delay 31 was about 2.8–4.1 percentage points relative to a target seen once. The larger reaffirmation-versus-revision gap partly reflects seeing the current value twice.

The nonlinear transfer probe uses a 64-dimensional, two-layer memory with 47,370 registered parameters and repeated-key training episodes. A maximum inner step of 1.0 produced a nonfinite loss at step 37; that failed run is preserved. A cap of 0.1 stabilized two 400-step runs, with short-episode held-out accuracy 1024/1024 and 1023/1024. Resetting or exchanging fast memories reduced accuracy to roughly chance. Long-stream recent acquisition was substantially weaker even online (76.2% and 81.7%).

A further GPU model trained on 72-write episodes achieved 1023/1024 on its held-out episode task. At the longest replay probe, online recent accuracy reached 2044/2048 (99.80%). Stored periodic replay still fell to 1244/2048 (60.74%); label correction plus replay decay disabled reached 2042/2048 (99.71%). Revised old-target counts were 510/512 online, 504/512 with stored periodic replay, and 511/512 with both corrections. This establishes a stronger toy baseline and a reproducible stale-replay failure. It does not establish a sleep advantage: corrected performance is near a ceiling in this setting, and a one-example old-target difference is weak evidence.

## Limits and decisions

- These are small synthetic associative-memory experiments with 16 keys and 8 values. They do not reproduce MAC/MAG/MAL, language-model performance, or the full Titans paper.
- Training and replay evaluation have different update horizons. Outer parameters are frozen during the replay comparisons; the models were not trained to exploit sleep.
- Online uses fewer updates. Only stored continuous versus stored periodic matches replay examples and update counts. Timing also changes the age of the most recent replay.
- The linear and nonlinear probes differ in width, depth, training distribution, and inner-step cap. Their comparison establishes transfer of a failure mode, not an isolated depth effect.
- Windows used Torch 2.15 CPU; Insomnia used Torch 2.11 CUDA 12.8. Exact versions, source hashes, and resource metadata are saved. Seed and environment effects are not separated by the two short nonlinear runs.
- The longer training pilot matches architecture, initialization and outer steps but uses 4.5 times as many observation writes. It is an adequacy check, not a compute-matched horizon ablation.
- Earlier probe metadata contains an inherited “short unique-key” caveat. For nonlinear checkpoints, the correct training protocol is repeated keys; each training run's metadata is authoritative.

The team should settle A1/A2/A4 update and retrieval timing, A7/A9 learning-rate/loss scaling, baseline architecture scope, and the replay policy before formal comparison. Current settings are experiment choices rather than edits to the decision register. The `paper-baseline-v1` tag is still absent.

## Next research step

Use the longer-horizon model as the starting baseline and increase distinct-key count and retention delay until online performance leaves the ceiling. Then train replay-aware and online controls under explicit matched budgets. Measure recent acquisition and stable/revised retention separately while sweeping revision rate. Causal buffer correction is a required control for any claimed sleep benefit. Establishing novelty requires a separate primary-literature comparison.

## Artifacts and reproduction

Local experiment root: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-research-2026-11-12`. Its `README.md`, designs, raw CSVs, analysis scripts, checkpoints, environment records, and Git bundle form the reproducibility package. The package index lists completed and failed runs separately. `analysis/` contains independent label/count/event reconstruction; `figures/` contains plots and their regeneration scripts.

Run the implementation checks with `python -m pytest tests/test_clmm_reference.py tests/test_clmm_replay_probe.py -q`. Run a bounded nonlinear training pilot with `python scripts/train_memory_pilot.py --out results/deep-pilot --device cuda --max-inner-lr 0.1`. Every experiment's metadata records its exact arguments. Local commits can be shared as a Git bundle while organization write access remains unavailable.
