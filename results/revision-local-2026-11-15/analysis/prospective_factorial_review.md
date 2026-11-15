# Prospective review of coordinator's 2x2 proposal

Date: 2026-11-15. This addendum follows the initial mechanism recommendation and precedes new 90823 outcomes. The coordinator owns the final declared design, runner, allocation and repository changes. This document is advice, not the final experiment contract.

## Proposed factorial and estimand

Coordinator proposal: `H' = clip(d*(H - e*P_x H) + eta*x*y^T)`, where `P_x H = [x/max(||x||^2,epsilon)](x^T H)`, `d=(1-eta)` or `1`, and `e=0` or `1`. All inactive modules remain protected. Four rules are original EMA, additive (neither bulk decay nor erasure), erase-EMA, and local-delta (erasure without active bulk decay). Same first write from zero H in all four cases.

Proposed frozen design: two Hebbian parents 90511/90515, content plus all five fixed random maps 90701..90705, four rules = 48 cases; fresh development 90823, 128 histories per cell. Primary declared candidate is local-delta versus original EMA; controls are for mechanism interpretation, with no quality-based rule selection. The factorial is justified: candidate changes erasure and bulk decay simultaneously, while controls separate their effects and possible interaction. A positive candidate contrast alone cannot establish which factor caused the benefit.

Coordinator proposes 3 percentage points as the maximum tolerated stable accuracy loss. This differs from the reviewer's initial 2-point suggestion but is acceptable if fixed prospectively as an engineering tolerance before new outcomes. Neither margin is a universal scientific constant; do not select the margin to fit observed intervals.

## Testing and admission clarification

- Strong joint repair claim: revised-query accuracy improves (paired lower bound >0), and K8D64 stable accuracy is noninferior (paired lower bound >-.03), plus prespecified acquisition/revision gates pass for the candidate. Primary family can be mean of all five fixed random maps with content as descriptive replication. Alternatively require both conditions in both route families as one joint success statement.
- The improvement/preservation conjunction is an intersection-union test. Testing each necessary component at one-sided .05 controls the joint success claim at .05 without Bonferroni, because overall failure is the union of component nulls. Do not reuse that argument to assert separately multiplicity-controlled discoveries for whichever component passes.
- For four individual familywise claims, Bonferroni allocates .05/4 to each lower tail: one-sided 98.75% lower bounds, equivalently the lower endpoints of two-sided 97.5% intervals. One-sided 97.5% bounds only correspond to Bonferroni for two individual claims.
- All fixed maps and parents should remain in aggregate analysis. Candidate qualification should preserve acquire2/revise2 >=.90, acquire4 >=.80, the blocked normal-minus-rolled test, and K-specific D0 >=.80. Baseline acquisition and K8D0 validity are needed for stable comparison. Baseline failure of revise2 is the phenomenon the repair targets; if full baseline revision admission is also demanded, that choice can mechanically block progression despite a successful candidate and must be explicit before outcomes. Report the baseline's full original gate result regardless.
- Bootstrap history IDs jointly across methods, fixed parents and maps. Use the full episode as a cluster, preserving two stable queries and their association with revisions. The resulting uncertainty is conditional on the fixed parents/maps/codebook, not an estimate of training-seed population uncertainty.
- Gate-only thresholds are operational qualification criteria, not confidence intervals or proof of near-perfect population ability. Per-parent/map deltas and failures remain visible.

## Conditional matched training

Coordinator proposes proceeding only if the declared frozen candidate passes the joint rule, then 24 matched training runs: candidate versus baseline, two parents, six routing tables, 1200 AdamW steps of batch32 using fresh training90802 and confirmation90923. No alternative rule may be chosen from the 90823 results. Fresh confirmation must remain untouched until all training recipes, final-step checkpoints, gates, comparisons and analysis are fixed; failures of 90823 are preserved as development evidence.

This is a substantial number of optimization steps (28,800 total). Feasibility within the coordinator's single GPU-hour ceiling is unverified here; this reviewer ran no performance probe. A runtime-only pilot must not inspect efficacy or change method selection, and a missed runtime bound means report incomplete evidence. Reserve time for artifacts/analysis; no unbounded retries, quota renewal or partial best-run selection. Preserve all failed and incomplete pairs. If paired equal-budget runs cannot all finish, do not call the subset a complete matched training comparison.

The paired runs need identical parent tensors, train arrays/order, routing table, optimizer, step count, batch size, loss, clipping and checkpoint rule. Both conditions may adapt slow weights under identical budgets, but their resulting values need not match. Start zero fast state for each training/evaluation episode. If the new rule's denominator or erasure induces numerical instability, record it as an implementation/protocol failure; do not silently introduce candidate-only stabilization or tune on confirmation.

## Coordinator's final refinement, still before outcomes

The preceding conditional-training proposal was superseded before new outcomes. Stage2 now uses the ORIGINAL UNTRAINED Hebbian initial checkpoints for 90511/90515 (payload training.step=0), rather than the dense-trained parents of stage1. This distinction is essential for a fair trained-from-identical-start comparison. Both initial and final models are saved and evaluated on 90923. All 24 stage2 conditions are intended regardless of stage1 quality; feasibility, not frozen efficacy, determines execution. Candidate remains local-delta, with no winner selection. Proposed training is 1200 steps, batch32, AdamW lr=.001/wd=0, gradient clip1, shared training90802.

The final success rule requests four individually adjusted inequalities: content and mean5random, each with revised-short superiority and stable K8D64 noninferiority at 3 points. Use the lower .0125 bootstrap quantile (lower endpoint of two-sided97.5% intervals), and require all four for global primary success plus candidate short/K8D0 admission. Baseline revision gate is excluded only for the repair contrast, while all other baseline gates remain visible. If both matched trained rules reach revision ceiling, unique revision superiority can fail even if both become qualified baselines; preserve that negative result.

Runtime projection must cover all24*(training1200+initial-evaluation+final-evaluation) within the coordinator's approximately2700-second continuation threshold, with maximum1 allocatedGPUhour and no renewal. Frozen diagnostics run in a separate CPU allocation. If infeasible, preserve only clearly labeled runtime-preflight evidence; do not choose a successful partial subset.

## Old failure raw check completed

`audit_old_revision.py` independently parsed all 48 old cases' saved revise2-online predictions and recomputed correct/obsolete flags. Across 6144 revised queries, 143 errors occurred; every revised error was the obsolete value, and all corresponding stable queries were correct. Output: `old_revision_raw_audit.json`. This verifies saved-prediction arithmetic, not inference reproduction or the proposed mechanism's causal sufficiency.
