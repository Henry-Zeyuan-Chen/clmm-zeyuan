# Prospective adapter-integration review

Review stage: pre-execution protocol review, November 15, 2026. Inputs are the coordinator's explicit task packet, the existing shared state/scheduler implementation, and the supplied final proposal. The team repository currently has no uploaded sleep/Hebbian implementation beyond empty directories. New code is therefore a **reference component implementation**, not an integration claim about unavailable teammate code or a complete Titans sequence architecture.

Parent branch: `adapter-integration`, based on `e3b744c`. Existing qualified correlated-key model source is `b5f27ff`; its literal-codebook compatibility loader is separate. The coordinator owns all model runs, resources, registry/checkpoints and the final handoff. This reviewer writes only this analysis directory and performs no accelerator work, retraining or delegation.

## Preliminary decision

The proposed common interface and bounded pilot can answer whether both reference tracks execute under the same observed streams, state/resume semantics and acquisition checks. The four-way associative comparison can compare trained component systems at equal declared offline updates. It cannot by itself attribute a difference solely to a biological rule, because independently trained encoders, decoders, base weights, plastic coefficients and learning rates can diverge. The existing nonlinear Titans-derived model is a descriptive common-data anchor, not a parameter- or architecture-matched control for the associative modules.

## Changes to freeze before a main run

1. **Selective access is a combined intervention.** Dense versus top-one changes both writing and reading and uses four versus one active module per observation. Report actual active-module updates and read access. Do not call this a matched-compute or isolated write-routing effect. A future write-only/read-only ablation is separate work.
2. **Separate trained-system and frozen-transfer comparisons.** Equal 1,200-update training budgets compare the resulting systems. Same-slow-weight forks can isolate the online rule/access change, but their Hebbian variants are transfer diagnostics rather than independently optimized Hebbian models. Their labels and checkpoint lineage must remain distinct.
3. **Predeclare the small fork diagnostic.** Replace an unspecified “if time” choice with either a fixed small diagnostic or a deterministic remaining-time criterion recorded before outcomes. Never run it only because a result is favorable or unfavorable.
4. **Reapply acquisition gates to every new adapter.** The prior nonlinear model's K8/K16 qualification does not transfer to a new architecture. Short acquisition, genuine revision, state-dependent retrieval and per-K D0 checks must precede a retention interpretation. Retain failures as diagnostics.
5. **Expose learned dynamics.** The EMA rule couples learning and forgetting through eta. Save its learned value, trace saturation and router occupancy alongside parameter/state sizes. A routing benefit can arise because inactive traces avoid decay; do not present it as evidence of an unmeasured mechanism.
6. **Keep the resource bound operational.** Eight maximum training invocations are acceptable only under the aggregate one-allocation/one-hour and pipeline timeout, with initial throughput checked. Preserve partial/interrupted runs and complete paired groups if the bound is hit. No best-seed or acquisition-dependent deletion of weak results.

## Mathematical contract to verify in source

For a module's encoded observed key `p`, encoded observed value `q`, fast trace `H`, and frozen-at-evaluation slow parameters:

`E = W0 + alpha * H`, `prediction = p @ E`.

The Hebbian reference proposes `H_next = clip((1-eta)*H + eta*outer(p,q), -1,1)`.

The matched associative-gradient reference proposes `H_next = clip((1-eta)*H - eta*alpha*outer(p,prediction-q), -1,1)`. The outer product here is exactly the gradient of half the **sum** squared error with respect to `H`; it is not a mean-over-feature loss. The inactive-module mask must return the original trace bit-for-bit, including no decay. Query routing receives only the query key, and read must not modify any trace, routing state, slow parameter or RNG.

The effective-weight and running-average trace formulation is related to Miconi et al., [Differentiable plasticity](https://proceedings.mlr.press/v80/miconi18a/miconi18a.pdf), equations 1–2. The proposed observation-clamped value activity, fixed content router, multiple feedforward associative modules and explicit clip are project choices. This is not a reproduction claim for that paper, and its learned eta/alpha do not make online updates gradient descent. Offline differentiation through the trace remains permitted by the proposal.

## Persistence and causal sleep checks

The sleep wrapper should reuse the existing literal-record FIFO/snapshot scheduler. It may retain eight working records and four frozen selections per stream; no history or target answers are read inputs. Periodic and continuous variants must consume the same records in the same order with identical extra write opportunities, differing only in placement. All returned writes must be applied before serializing model and policy state.

Session resume requires the matching slow checkpoint, adaptive state and policy JSON. Config/method identity alone does not identify a slow checkpoint; the run registry or session bundle must bind its checkpoint hash. Partial-block pause uses export/import; terminal finish may flush only arrived periodic selections and closes the stream. Model and policy counters, phase and batch must agree on import. Boundary/reset controls have different effective-state work and must stay separate from equal-budget ordinary arms.

## Source review and independent pre-outcome proof

The first source review identified a no-op mutation inside codebook validation, type-alias acceptance in resume configuration, and a wrapper commit occurring before candidate coherence validation. The implementers corrected these before the independent proof. The wrapper now validates temporary candidate state/policies/counters before committing and explicitly requires functional backend writes. This reviewer made no repository edits.

`adapter_cpu_audit.json` and `audit_adapter_cpu.py` record the subsequent independent bounded CPU proof. It completed in 0.56 seconds after imports, using PyTorch 2.15.0+cpu and the actual saved rank32 checkpoint. Both initializer seeds × all four update/access configurations have identical initial slow tensors across variants. Gradient-rule writes agree with a separately differentiated current-state half-sum squared-error objective; Hebbian writes agree with the declared EMA. Inactive traces remain identical, online writes make no `torch.autograd.grad` call, reads/slow parameters/RNG remain unchanged, and save/load continuation is exact. Each associative stream has 65,536 bytes of adaptive trace backing.

The actual nonlinear shared checkpoint also passes the batched sleep-session proof for online, repeat, continuous and periodic modes with a terminal partial delay of three. Emitted events match the frozen causal plans exactly, including mid-block model-plus-policy resume. This demonstrates execution/state compatibility, not acquisition or efficacy of new trained methods.

Source hashes at this proof are `e52c78207bd308e98815e55b7140dc36016c87a40614d628995e6cc25b8409f5` for the associative adapter and `9b7e0b089905ec8027814644991d1135de6a93858c6b15dee78782b3bb87652d` for the sleep adapter. The machine-readable record lists scheduler/portable-loader hashes and the exact checkpoint hash. Any later source change must be reconciled before reusing this check as evidence.

## Accepted prospective clarification and remaining scope

The coordinator accepted the comparison framing above. A fixed eight-fork diagnostic (two gradient/dense final parents × four target configurations) will use identical slow tensors, reset traces and only short admission plus online K8 D0/D64. It follows completion of all eight paired training runs if the declared aggregate runtime permits; otherwise it remains explicitly unexecuted. It is never selected by efficacy outcomes. Its purpose is frozen-transfer mechanism diagnosis, not independently optimized Hebbian efficacy. All trained systems retain separate short/per-K D0 gates and learned eta/alpha/occupancy/saturation reporting.

The completed runner, registry, design JSON and launcher received a final prospective read at source `a41bb09`. `prospective_freeze_review.json` records exact design and launcher hashes before main outcomes. The primary is now explicitly the ONLINE K8,D64 stable-query top-one-minus-dense contrast within independently trained Hebbian systems, a paired development pilot rather than confirmatory efficacy. Its conditional interpretation still requires both systems' own admission gates. Secondary/fork results remain labeled separately.

The review found one runtime-estimator issue: pressure-two preflight rows had actual batch size two while main rows use batch size sixteen. The coordinator corrected the preflight to pressure-sixteen, measuring actual B16 batches before extrapolating batch counts. The preflight gate uses only elapsed-time rows, 1.25 padding and a declared conservative NNM multiplier; it does not choose configurations or seeds by score. The eight fixed fork runs are contingent only on all main training completing and the remaining hard runtime reserve.

No blocking runner, registry, gradient/state, data-isolation or declared-budget issue was identified in this reviewed scope. No main model outcome is available at this freeze review, so it does not claim actual training admission, GPU resource consumption or efficacy. Saved-artifact audits and result interpretation follow actual completed runs.
