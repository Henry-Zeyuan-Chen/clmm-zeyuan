# Prospective independent review

Reviewed on 2026-11-15 before any calibration-model outcome. Parent repository: `ffaffed543232a4bc49634aa73580cdd9190c574`; new branch: `nonlinear-correlated-control`. The reviewed `design.json` declares a finite rank sweep and separate online calibration, fresh confirmation, and fresh replay comparison. The reviewer owns only this analysis directory and will perform no GPU work, retraining, repository edits, or delegation.

Reviewed design SHA256: `7d2f65917b99170e61fd7aadb389260ccfd4a5340fb4a121eb04a84429c09f11`.

Before any main GPU outcome, the coordinator clarified actual-comparison D0 adequacy as four-target all-correct ≥0.8 in both trained checkpoints; if either fails K8 D0, the fixed-two-seed primary is diagnostic. The gate block bootstrap seed is explicitly 90504. Amended prospective design SHA256: `0afff638ed23e070376b4c15dac5df48d70123cc9916746556edff626b22f877`. Frozen implementation: `b5f27ff4738ded4832538573d454d2461786d38b`.

## Decision and scientific scope

The prospective design can identify a useful correlated-key nonlinear memory setting without selecting on a replay advantage. It retains the Member C component-foundation scope. It does not implement sleep or Hebbian modifications, resolve the byte-model transfer failure, or establish variability over new codebooks/training datasets.

The candidate order is rank 32, then 16, then 8. Every visited candidate uses both fixed initializers 90511/90515, the same 1,200-update budget, batch 32, and shared actual training arrays from seed 90502. Model dimensions, inner rule, outer objective, optimizer, step size, and registered memory shapes stay fixed. Only key geometry/effective input subspace changes. A first-eligible stopping rule with at most six training runs is a finite calibration budget, not a best-seed or best-checkpoint search.

## Important issues resolved before outcomes

1. **Nested codebooks:** construct one private-CPU-generator 64×64 Gaussian base at seed 90501, slice its first r columns, row-normalize, and zero-pad to 64 dimensions. Independently drawing a 64×r matrix for each candidate would not produce nested per-key vectors. Private codebook randomness must leave identical MLP initializer tensors for the same initialization seed across ranks; verify this from saved initial checkpoints.
2. **Selection unit:** every short and pressure criterion must pass in each original initializer. Do not allow a fixed-seed average to compensate for a failing seed. The short gate is all-correct episode accuracy, with the existing cyclic-roll-block interval. K8 D0 uses four-target all-correct; headroom/drop use only the two stable queries.
3. **Explicit fallback:** if no candidate is eligible, the design chooses the last candidate passing short plus K8 D0 acquisition in both seeds; otherwise rank 8. Any resulting replay comparison is labeled ineligible diagnostic. A confirmation failure also retains the planned comparison as diagnostic, with no reselection or resweep. Preserve weak, failed and interrupted runs.
4. **Fresh headroom:** the headroom window is selected from only 32 calibration histories. Confirmation/comparison online retention may move outside it. Record that result without further tuning or claiming the window was confirmed. Comparison outcomes must never select a new configuration.
5. **Paired inference:** the main primary resamples the same 128 whole-history IDs across both fixed seeds and all arms, then averages each history's two seed-specific differences. Treating 256 seed×history rows as independent would overstate replication. Report both seed effects and the fixed-seed average; the interval does not estimate training-procedure or codebook variability.

## Runner/evaluator inspection

Inspected `scripts/kv_correlated_experiment.py` and the minimal optional `datasets`/`pressure_arms` extension to `scripts/kv_representation_control.py`. No protocol blocker was identified in the draft. Calibration and confirmation retain normal/reset/rolled short controls while restricting pressure model execution to online only. Calibration/confirmation pressure is K8 D0/16/64. Comparison retains the full K8/16/32 grid and all five pressure arms.

The initial and final calibration evaluations reuse exactly the same generated histories. Training starts from declared fresh initialization; checkpoint warm starts are rejected in training mode. Initial and final checkpoints, optimizer, training arrays, labels, complete plans, raw scores, design, resolved arguments, source snapshots and hashes are captured. Checkpoint loading checks configuration, codebook identity and finite tensors. Evaluation asserts frozen slow state, pure reads, RNG preservation and exact fast-state round trips. The legacy default evaluator behavior is unchanged when the new optional arguments are absent.

The runner must be frozen before launch; this review is of a developing source draft. Adapter inspection, source identity, final selection-script logic, and observed artifact consistency remain to be checked when available. No outcome or qualification is implied here.

## Interpretation requirements

- The supported transfer population is fresh whole histories/compositions with familiar IDs/atomic bindings; exact ordered planned-training histories are excluded from every evaluation, including untrained anchors.
- Retention interpretation requires calibration eligibility, fresh confirmation, and adequate actual comparison D0 acquisition at the relevant K. Retain all initial-checkpoint comparisons regardless of their gates, but label weak-acquisition anchors as diagnostics.
- Use the same four initial targets across delays. Stable/revised/recent strata remain separate; observation age and FIFO residency prevent interpreting a cross-stratum gap as pure revision cost.
- Periodic versus continuous is the matched timing primary within a fixed representation/checkpoint. Online has fewer write opportunities, and equal opportunities do not imply equal nonzero gradients, wall time, or storage.
- Running exact saved initial checkpoints on the same fresh comparison histories supports an anchored initial/final analysis. It does not by itself isolate an explanation for any learned effect.
- Preserve the one-allocation, one-hour, no-renewal limit and all stopping/failure records. Exhaustion of the finite sweep is an informative negative calibration result, not permission for another undeclared search.

The later independent audit will reconstruct labels, exclusions, source/codebook/parameter identity, causal plans, raw-score metrics, final-only selection, confirmation, retention eligibility and paired primary intervals, then generate numerical tables/figures from those saved artifacts.

## Adapter and selector follow-up before outcomes

Inspected `clmm/pretrain/kv_correlated.py`, its tests, and `WORK/run_pipeline.py`. The adapter reuses the inherited nonlinear write/read mechanism; key vectors come from the required private single Gaussian base, and the configuration/state fingerprint incorporates the actual key tensor's shape, dtype and canonical float32 hash. Checkpoint codebooks are validated. The selector implements both-seed final-only gates, ordered stopping, declared fallback, confirmation without reselection, and a comparison-freeze record before any replay run. No blocker was identified.

A separate CPU-only mechanism proof finished in 12.87 seconds, covering all three ranks and both declared initialization seeds. Engine parameters and global RNG are bit-identical for each same-seed comparison across ranks and the original nonlinear adapter. Nested codebooks match the prescribed construction. Two current-state neural writes, including a genuine revision, match independently differentiated half-sum-SSE SGD exactly (maximum absolute error 0). Reads leave slow/adaptive values unchanged; save/load/continuation is bit-identical. Live/compact adaptive backing storage is unchanged across ranks at 131,072/65,536 bytes per episode. This proves bounded interface/mechanism properties only; it is not a training or efficacy result. Evidence: `adapter_cpu_audit.json` and `audit_adapter_cpu.py`.
