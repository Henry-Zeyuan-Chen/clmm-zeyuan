# Independent local revision review, before new outcomes

Date: 2026-11-15. Input checkout: `clmm-replay`, branch `routing-decay-controls`, initial HEAD reported by coordinator `a98c849821b16ff48108e51a309e4676770512fb`. Scope: mechanism/protocol review and CPU analysis of saved outputs. No training, accelerator execution, performance probes, repository changes, commits, or external communications. This reviewer saw the earlier failure claim in the assignment; independence applies to formula and raw-evidence checking, not an outcome-blinded review of that older claim.

## Observed mechanism

`clmm/hebbian/adapter.py::write` forms module-specific `x=tanh(key @ key_encoder)` and `y=tanh(value @ value_encoder)`. The Hebbian active update is `H' = clip((1-eta) H + eta x y^T)`. `RoutingDecayAdapter` selects one fixed, identical write/read module, and leaves inactive H unchanged under protection. The read uses `W0 + alpha*H` and the learned decoder. Eta and all slow tensors are fixed for the proposed comparison.

For repeated identical x, old responses remain in H until subsequent same-module writes decay them. A conflicting value contributes another outer product instead of removing the old one. The pre-existing short task is exactly three observations: key A/old, key B/stable, key A/new, then a query for each. This supports examining old-association retention as a local failure mechanism; returning the old label alone does not establish that it is the only mechanism.

## Smallest proposed causal intervention

Use parameter-free projection replacement in the active trace only:

```
r = x^T H
s = max(sum_i x_i^2, epsilon)
H' = clip((1-eta) * [H - (x/s) r^T] + eta x y^T)
```

Retain existing inactive protection, clipping, readout, router, eta, and slow tensors. A fixed numerical epsilon is a denominator guard, not a tuned hyperparameter. For nonzero x where the guard/clamp is inactive, `x^T H' = eta ||x||^2 y`, eliminating the old response in the currently observed key direction. For any z orthogonal to x, `z^T H' = (1-eta) z^T H`. A first write from zero H is exactly the original Hebbian write. Apply the same rule to every observation, including nonrevisions; no old-value table, revision flag, query/target labels, or extra persistent state is needed.

This is a candidate associative local rule, not a biological or methodological novelty claim, and not a complete Titans implementation. Because alpha is elementwise and W0 is nonzero, erasing the raw fast response does not algebraically guarantee erasing the final decoded obsolete score. The experiment tests the actual end-to-end readout. Correlated key directions can make replacing one association damage other stable keys; this is the central competing explanation to measure.

Avoid changing eta, target magnitude, or effective-weight scaling simultaneously. A conventional delta update scaled by eta would also alter the old residual, but would not remove its key-direction response exactly in one write. Whole-matrix resets, label-aware conflict detection, altered value routing, and separate per-key histories would add confounds or state.

## Prospective protocol recommendations

1. Compare original protected EMA versus protected projection replacement for both existing Hebbian parents and content plus all five preselected random maps. Keep old global EMA as an explicitly secondary anchor if budget allows; do not replace protected EMA with it as the sole control. The mathematical intervention is the rule within the same parent/map/decay.
2. Use fresh whole-history seed(s), exclude training and consulted old development histories, and preserve exact shared episodes/values/order across all conditions. Existing 90723 outcomes informed this rule choice and are development evidence. No eta/map/checkpoint/hyperparameter selection on new evaluation outcomes. Keep every fixed map and both parents, including gate failures. Preserve the complete condition manifest before outcomes.
3. Keep one retained association update, one effective read module, the same 64 KiB traces, 512-byte routing table, and zero extra online storage per stream. Correction requires a local trace contraction and an erasing outer product: equal logical budget does not mean equal FLOPs. Report extra arithmetic and measured elapsed time without changing compute allocation or tuning based on quality.
4. Preserve original per-condition short gates: acquire2 and revise2 episode-all-correct >=.90, acquire4 >=.80, blocked paired normal-minus-rolled acquire2 lower95 >0; reset stays diagnostic. Preserve K-specific D0 four-original-target all-correct >=.80 in each parent. A failing revised method is a diagnostic, regardless of aggregate gains. Do not retroactively relax gates.
5. Primary revision measurement: paired per-history revised-query accuracy and obsolete rate in revise2; report both-correct to preserve its existing gate. Pressure revisions at D0/D16/D64 are secondary, as are K16 and allocation effects. Short-task stable-query accuracy checks collateral damage locally, while K8D64 stable accuracy measures the previously protected benefit.
6. For stable recall preservation, predeclare a task-level absolute margin and use a paired noninferiority bound rather than 'not significantly different'. Recommendation: delta=2 percentage points, explicitly a maximum tolerated engineering loss rather than a universal scientific constant. For each route family (content and mean of the five fixed random maps), require lower one-sided95 paired-history bound for new-minus-old K8D64 stable accuracy > -0.02, and disclose every per-parent/map delta. Larger observed regressions in an individual map/parent should prevent a blanket robustness claim even if the aggregate passes. If the interval is too wide, report inconclusive preservation; do not enlarge the margin after outcomes. A zero-margin request would require stronger evidence and should be stated explicitly.
7. A defensible success conjunction is: all required gates pass, revised-query improvement is positive with its paired lower bound >0, and stable noninferiority passes in both allocation families. Because this is a conjunction of improvement/preservation, do not convert a failure of one component into overall success. If separately claiming benefits across multiple families, use adjusted intervals (e.g., one-sided97.5 per family) or state the nominal exploratory scope. New-rule versus old-rule differences conditional on frozen parents/maps are the estimand; histories are the resampling unit, not queries, maps, or numeric seed labels.
8. Resample history IDs jointly across methods, parents and maps, preserving within-history queries. Averaging fixed parents/maps first estimates their conditional average; no population claim across training seeds or router draws follows from two parents/five maps. Delay cells share prefixes, so do not treat them as independent replicates. Rolled states create 16-history dependencies; retain its separate blocked bootstrap.

## Required implementation and artifact gates

- Independently verify the equation on synthetic tensors, exact zero-state equivalence, the unclamped projection identity, orthogonal-subspace behavior, masked module preservation, denominator behavior for zero/small keys, clipping, and unchanged write counter.
- Keep original adapter/routing-decay checkpoint/state schemas unchanged. New rule must be in a separate versioned config/fingerprint/state schema; reject cross-rule imports even when shapes match. Save exact rule/epsilon/map/parent SHA and slow tensor identity.
- Verify fixed routing construction/cardinality/hash and identical read/write masks; global RNG and parameters remain unchanged online. No optimizer or fast-state inheritance; every session begins H=0. Parameter counts and persistent physical fast bytes must match.
- Independently reconstruct saved prediction argmax/correct/obsolete/MSE, summaries, gates, plans, resource counts and query exposures from raw outputs. Check shared episodes/exclusions, model sources and checkpoint tensors. Preserve failures and amendments rather than overwriting histories.

## Possible conclusions

Revision improves and stable preservation passes: the local raw-trace replacement resolves this observed failure under the fixed component/protocol, with resource qualification. Revision improves but stable preservation fails: a quantified stability/revision tradeoff, not a successful repair. No revision improvement: raw-trace interference, elementwise effective readout or training compatibility may invalidate this candidate. Passing only one parent/map or a postselected subset is insufficient for a family-wide claim.
