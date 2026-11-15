# Prospective independent design review

Review date: 2026-11-14. Reviewer: delegated independent audit agent. This note reviews the supplied plan and existing source at `ea967d7`; it is not an execution result. The implementer and experiment coordinator own the source and runs. This reviewer may write only within this `analysis` directory and does not allocate an accelerator.

## Scope and admissible claims

The common controlled representation is 64 orthogonal one-hot keys in dimension 64 and 16 one-hot values in the first 16 coordinates. Reads decode those first 16 coordinates. A model consumes the current observed key/value pair during writing and only a query key plus its adaptive state during reading. Saved fast state must permit continuation across a session boundary without consulting raw observation history.

The exact linear Delta control is an interface positive control. For unit orthogonal keys, the half-squared-error gradient update with step 1 changes only the queried key's row/column and replaces its response exactly. Distinct keys do not interfere. Consequently correct acquisition, revision, and retention are expected by construction, including with many distractor writes. A perfect Delta result does not establish learned generalization, a sleep effect, a Hebbian advantage, or full Titans performance.

The separate neural model is an explicitly simplified adapter around the vendored `NeuralMemory`, with a nonlinear 64→128→64 MemoryMLP, fixed identity Q/K/V, no norms, no momentum, half-sum squared error, fixed step 0.5, and effectively zero decay through a finite −80 logit. Only the initial memory-model parameters receive offline updates. This is a learned closed-vocabulary mechanism experiment; it is not a full Transformer or a paper-faithful pretrained Titans reproduction.

## Validity checks to resolve before launch

1. Define held-out assignments as fresh episode-level key/value mappings or stream compositions. With all 64 keys, 16 values, and 38,400 training episodes, individual key/value pairs are almost certain to recur. Fresh compositions must not be described as unseen individual pairs, keys, wording, or language generalization.
2. Use genuinely changed revision values. Report obsolete-answer rate only for revised queries. Include a strict distinct-answer K2 acquisition measure and rank-specific accuracy; otherwise an always-one-value rule can obtain both-correct on the all-equal subset.
3. Declare the neural acquisition/admission criterion and seed aggregation rule before evaluation. Retain both initialization seeds and all failed/interrupted runs. Fixed checkpoints evaluated on many episodes are not many independent training replications.
4. Serialize every field required for continuation, and compare uninterrupted execution with a save/load/continue execution. This includes any sequence index, cache, or past optimizer state actually required by the adapter. Reads must be side-effect free. Reset and rolled-state conditions must change adaptive state without changing the evaluation data or the slow parameters. Save the rolling permutation.
5. Freeze the complete slow parameter set during evaluation and verify its fingerprint afterward. The inner update remains an allowed adaptive-state computation. Do not confuse frozen outer parameters with absence of inner gradients.
6. Replay source events and labels must both predate the frozen block boundary. A snapshot must retain values known at that boundary and must not retroactively use future revision labels. Specify FIFO age behavior on revision and whether distractors come from other initial keys or introduce new keys.
7. Continuous and periodic replay must share the same causal block snapshot and the same number of auxiliary writes; repeat must match that auxiliary write count. Use identical step/decay rules for observations and replay. The online arm is an explicitly unmatched lower-compute anchor.
8. Count adaptive tensor storage, working FIFO8, and the active frozen block snapshot. Distinguish logical model-accessible memory from precomputed evaluator traces, checkpoints, serialization overhead, and slow model parameters. When tensors alias, report a defensible unique-storage count or state that a logical tensor count is used.

## Planned interpretation

An exact-control failure is a harness or numerical-contract issue to resolve before interpreting learned-model scores. A neural acquisition or genuine-revision failure leaves that checkpoint unqualified for a clean retention claim; delay scores may still be reported as diagnostics with that limitation. Passing acquisition validates only this fixed codebook and these sampled assignments. A periodic-versus-continuous difference is evidence about the specified schedule under matched updates, not a general sleep mechanism, and a two-seed result cannot establish robustness of a training procedure. This work does not implement the teammates' sleep module or selective Hebbian rule.

## Source inspected

- `scripts/research_memory.py`: earlier learned symbolic reader, including learned key/value embeddings and output head.
- `scripts/memory_pressure_probe.py`: earlier causal matched replay and evaluator trace conventions.
- `titans_pytorch/neural_memory.py`: online gradient, initialization, decay, momentum, and continuation semantics.
- `titans_pytorch/memory_models.py`: nonlinear MemoryMLP implementation.
- `docs/MEMBER_C_STATUS.md` and `docs/PAIRED_ACQUISITION.md`: historical qualifications and existing acquisition gates.

The final code, traces, prediction rows, checkpoints, resource record, and reproducible aggregate analysis remain to be inspected after the coordinator supplies their schema.

## Coordinator protocol clarification before results

The coordinator subsequently fixed the claim as fresh independently sampled whole histories/compositions within the same 64-key, 16-value vocabulary. Exact ordered key/value training histories will be excluded from evaluation; atomic bindings and K2 subsets of larger training episodes may overlap and will be disclosed. Acquisition uses distinct current values, and revisions exclude both the old value and other current values.

The two fixed initialization seeds are 90301 and 90305; training uses seed 90302, 1,200 steps, and batch size 32. The first candidate has inner step 0.5. A single conditional alternative at inner step 1.0 is allowed only if the first candidate fails admission, with both original seeds receiving the same budget. No best-seed selection is allowed. Admission requires both seeds to reach K2 all-correct ≥0.9, revised-K2 both-correct ≥0.9, K4 all-correct ≥0.8, and a positive lower 95% paired-episode interval for normal-minus-rolled K2. The first candidate satisfying all requirements is evaluated on a separate confirmation seed 90313; neural pressure evaluation occurs only if both seeds confirm. The exact Delta control always receives the full suite.

Pressure evaluation fixes K∈{8,16,32}, D∈{0,16,64}, and 32 episodes per cell. The same K initial observations and revisions of keys 2/3 are the common online-only prefill for every arm. Adaptive state is serialized and restored at that session boundary. Delay distractors use keys outside the initial K and fixed per-key values; queried initial targets never recur as observations after the boundary. The FIFO holds the latest pair for each of at most eight keys, and revisions refresh FIFO age. Each four-observation delay block freezes four pairs selected from the FIFO at its start. Continuous replay interleaves these same four pairs with observations; periodic replay emits them at block end; repeat adds one current-pair write per observation. All rules use the same gates/decay. Pressure queries are stable keys 0/1, revised keys 2/3, and the latest distractor when D>0.

This clarification resolves the principal design ambiguities. One remaining interpretation caveat is structural: revised targets are more recent in fast state and refreshed in FIFO, while stable targets are older and absent from FIFO at the K16/K32 boundary. Compare schedules within each target stratum. A stable-versus-revised accuracy difference is not an isolated revision effect because age and residency differ as well.

## Frozen execution clarification

The frozen implementation is `ff62df0f85ff7cbf62f3209755ea066959aa7085`. Offline training minimizes mean squared error between the 16 raw read coordinates and the current one-hot target, using AdamW learning rate 0.001, weight decay 0, and gradient clipping at 1. Both initial and final checkpoints receive short acquisition evaluation. The gate's interval was clarified before launch as a bootstrap of the eight cyclic-roll blocks of 16 episodes, retaining state-donor dependencies, rather than treating all cyclically paired episodes as independent.

For pressure analysis, the four initial targets form a fixed aggregate. The recent distractor is reported separately because it is present only at positive delays. A K2/K4-qualified model still needs its K8/K16/K32 D0 scores inspected; low D0 performance at a larger K cannot be presented as a clean delay-retention result.

The first launcher attempt stopped after a completed Delta full suite and first completed neural seed because a NumPy boolean was passed to JSON serialization in the selection script. The coordinator preserved the failed launcher artifacts and reused completed successful model invocations after casting the gate result to a plain boolean. The model source/protocol remained frozen. This is an operational launcher failure, not a failed or excluded model result; final artifact checks must verify the resumed record and command/source identity.
