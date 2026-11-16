# Roadmap: from baseline to our modification

**Goal.** Before we start our modification, we will have a tested, paper-faithful set of Titans models (MAC, MAG, MAL, LMM) and same-codebase baselines, frozen under the tag `paper-baseline-v1`, with every deviation from the paper written down. We compare all of our modification results against that tag.

**Current working scope, November 14.** Following the supplied group-sync discussion, the immediate local milestone is one small, tested lucidrains-based sequence model and shared training/evaluation contract for both modification tracks. See [SHARED_PRETRAIN.md](SHARED_PRETRAIN.md). Full paper-style MAC/MAG/MAL/LMM reproduction below is deferred; these checkboxes and the `paper-baseline-v1` tag are not claimed complete. Experimental defaults remain distinct from reviewed team decisions in the ambiguity register.

**Priority update, November 16.** The [TA feedback plan](TA_FEEDBACK_2026-11-16.md)
requires explicit cross-segment state carry, a same-size unmodified memory-update
comparator, and recognizable benchmark foundations. First validate streaming
carry/reset/resume and train a learnable vanilla baseline; then connect intended
sleep/Hebbian modules to a shared, benchmark-backed evaluation. Short acquisition
and format diagnostics do not establish retention. Existing completed results,
deviations and unchecked paper-reproduction tasks remain unchanged. This update
starts no experiments or new allocation.

**Principle.** We follow the paper wherever it is explicit, use our [ambiguity register](ambiguity-register.md) decisions wherever it is silent, and depart from the literal equations only where they would let a token see later tokens. We record every departure in `docs/DEVIATIONS.md` with its register row ID.

**How we use this file.** Each unchecked task gets a GitHub Issue with one owner. We work on the branch named in parentheses, and tick the box in the pull request that finishes the task.

## Phase 0: Setup (done)

- [x] Private org repo mirrored from lucidrains/titans-pytorch v0.6.0; tag `baseline-v0.6.0`
- [x] Shared venv instructions, lockfile and `SETUP.md`
- [x] Baseline verification: lucidrains passes causality; every Neuranox variant fails (`results/`)
- [x] Ambiguity register in `docs/`
- [x] Our README, `clmm/` package, and lucidrains' top-level docs archived in `upstream/`

## Phase 1: Decide the register rows that block code

- [ ] List the compute each of us can use (laptops, university cluster, cloud credits). This sets D1 and D5.
- [ ] A1, A2, A4: chunk size b, gate granularity, when a token reads memory
- [ ] A5–A9, A13: memory MLP shape, learned `M_0`, gate form, loss scaling, surprise clipping
- [ ] B1–B3: convolution, SiLU and ℓ₂ order, output norm and gate
- [ ] C1, C2, C4, C10: MAC causality, what memory stores, the ⊗ gate
- [ ] D1, D5: model sizes and data budget
- [ ] Record each decision in the register through a pull request

Note for A2: truly per-token gates mean storing a full copy of the memory weights for every token, which is infeasible at 340M-model widths. Unless we erwise, we use small chunks (4–16 tokens) with per-token θ and record this in `DEVIATIONS.md`.

## Phase 2: Reference implementation (`reference-impl`)

- [ ] `clmm/reference.py`: plain loops applying eqs. 11–15 token by token, exactly as decided in A1, A2 and A4
- [ ] `tests/test_clmm_reference.py`: lucidrains' `NeuralMemory`, configured to match, agrees with the reference to about 1e-8 in float64. Settings: `mem_model_norm_add_residual=False` (A8), `store_memory_loss_fn` (A9), `batch_size=b` (A1), `chunk_size` (A2), `default_step_transform_max_lr` and `init_decay_bias` (A7)

Done when the test passes for several chunk sizes and memory depths 1–3. Every later engine change must keep it passing.

## Phase 3: Engine changes (`engine-flags`)

All new options default to off, so upstream behaviour and `tests/test_titans.py` stay unchanged.

- [ ] Separate q, k and v activations (today one `activation` setting is shared by all three)
- [ ] ℓ₂ normalization of q and k (B2)
- [ ] Causal depthwise convol the q, k, v projections (B1), carrying its last few inputs in the memory state so segmented runs match one-shot runs
- [ ] Output norm and gate before the final projection (B3)

Done when the upstream tests pass, the reference test passes with the flags on, and a segmented run equals a one-shot run with the convolution on.

## Phase 4: Paper-style models and baselines (one branch per file)

- [ ] `clmm/attention.py`: multi-head attention with RoPE, segment and sliding-window masks, persistent tokens (B4–B7, C7, C9, C11, C12)
- [ ] `clmm/mac.py`: segment loop that retrieves `h_t` from the previous state, attends over [P ‖ h ‖ S], stores `y_t` and gates the output (C1–C8, C10)
- [ ] `clmm/mag.py`: parallel attention and memory branches with the ⊗ gate (C10, C11, C13)
- [ ] `clmm/mal.py`: memory layer, then attention, then FFN (C12, C14)
- [ ] `clmm/lmm.py`: memory-only model (C15)
- [ ] `clmm/baselines.py`: Transformer++ and a sliding-window-only transformer, sharing our attention and FFN code (D2)mm/config.py`: size presets (D1)

Done when each model trains for 200 steps on the `data/` enwik8 sample with falling loss and no NaNs.

## Phase 5: Tests for every model (`tests-causality`)

- [ ] Port `scripts/verify_titans.py` into pytest tests for every `clmm` model: causality, cross-segment memory effect, segmented equals one-shot, `gradcheck` on a tiny model
- [ ] Optional: GitHub Actions running `pytest` on every pull request

## Phase 6: Small-scale fidelity checks (`bench-*`)

We can't match the paper's absolute numbers without its compute, so we check that its patterns reproduce.

- [ ] Synthetic recall (associative recall; passkey retrieval at 2K–16K): Titans models beat the sliding-window baseline with the same window
- [ ] Table 5 ablations: removing weight decay hurts most, then momentum, convolution and persistent memory; linear memory does worse than deep memory
- [ ] Small language-model run (about 50–125M parameters, 1–3B FineWeb-Edu tokens, scaled to our compute) for every model andbaselines; in the paper's Table 5, MAG had the best perplexity and MAC the best long-context accuracy
- [ ] Each result saved in `results/` with its config file and commit hash

## Phase 7: Freeze the baseline

- [ ] `docs/DEVIATIONS.md` lists every departure from the paper with its register row
- [ ] README "Status" updated to describe the frozen baseline
- [ ] Tag `paper-baseline-v1` on `main` and push it

## Ready for our modification when

- [ ] All tests pass on `main`
- [ ] Every model has a config and a saved results file
- [ ] `DEVIATIONS.md` is complete
- [ ] `paper-baseline-v1` exists, and our modification branches start from it
