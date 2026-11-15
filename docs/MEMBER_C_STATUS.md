# Member C: shared training and evaluation

Updated November 15, 2026. Zeyuan owns pretraining feasibility, reusable baseline weights and common evaluation. Member A owns sleep consolidation; Member B owns selective Hebbian/neural-memory updates. The proposal asks whether those interventions improve new-association acquisition, unchanged-fact retention and genuine revisions. Sleep timing needs matched replay budgets. The Hebbian track must distinguish its update rule from selective routing.

## Current deliverable

The [key/value control](KV_CONTROL.md) now supplies two acquisition-qualified **memory-component** initializers, fixed common encoding/decoding, a write/read/state API, saved boundary states and a live bounded replay scheduler. Both nonlinear seeds score 128/128 all-correct on K2 acquisition, K4 acquisition and real K2 revision, including fresh confirmation histories. All keys and atomic bindings were familiar during training; transfer concerns new whole histories/compositions.

Initial checkpoints already pass those short accuracy gates. The two offline training runs lower read-coordinate MSE, without an observed short-accuracy improvement. Preserve untrained anchors in the next experiments rather than calling outer training essential. Each final checkpoint has explicit provenance and capability limits in `results/kv-control-2026-11-14/checkpoint_admission.json`.

The orthogonal pressure grid is 100% in all ordinary arms, so it cannot establish a replay advantage. A separately declared correlated-key linear control produces interference with adequate initial K8 acquisition, but its primary periodic-minus-continuous interval includes zero. It is one fixed untrained instrument with different geometry and state size, not a trained neural or Hebbian result. This identifies a direction for calibrating pressure rather than a project-method conclusion.

The common persistent interface is now implemented and tested at the component level. Slow weights are frozen during evaluation; queries receive only keys and fast state. Save/load/continue preserves answers, and both fast state and replay-scheduler state can resume within a block. The live policy stores ≤8 FIFO pairs and ≤4 frozen selections. It matches every saved experiment plan; both learned initializers pass the all-arm integration demo. Full language-model cross-session integration remains a later milestone.

## Earlier evidence and its limits

The [shared byte-language pilot](SHARED_PRETRAIN.md) provides two complete compact model checkpoints and a modest language-context memory effect. The [fact/revision evaluation](FACT_REVISION.md) supplies causal matched schedules and exposes failed acquisition. The [paired-query diagnostic](PAIRED_ACQUISITION.md) improves single-fact recall and revision, but fails two-key selection even when attention can see the facts.

The [fixed-set fitting and assignment-transfer diagnostic](FIXED_SET_GENERALIZATION.md) establishes finite-set trainability: all four models fit 32/32 contexts. Transfer still fails. Fresh-assignment follow-ups score only 1/128 and 3/128 both-correct on training-disjoint histories. These compact lucidrains-based pilots do not establish paper-scale Titans pretraining, architecture impossibility or sleep/Hebbian efficacy. The typed component success does not promote those byte checkpoints into retention baselines.

## Shared contract for the two method tracks

1. **Weights and provenance:** retain initial/final checkpoints, architecture, actual source/data identity, optimizer and training budget. Memory states require the corresponding slow weights. Byte weights, typed-memory weights and new Hebbian architectures need explicit loaders/adapters; matching dimensions alone is not compatibility.
2. **Evaluation:** use the same observed streams, key-only reads, fixed decoder, genuine revisions and separate stable/revised/recent metrics. Gate on short acquisition and per-K D0 performance before interpreting delay. Keep slow parameters frozen and verify read purity, reset/rolled controls and persistent state behavior. The fixed four initial targets form the delay-curve denominator.
3. **Replay:** continuous and periodic must see the same eight-pair FIFO, same frozen samples and same number/order of extra writes, differing only in timing. Report repeat-current and lower-budget online anchors. Record actual adaptive tensor storage, slow parameters, replay pairs and update opportunities separately; equal writes are not necessarily equal compute or nonzero gradients.
4. **Hebbian comparisons:** implement the update rule behind the same API first, then add selective routing as an ablation. Expose extra slow training, state tensors or representations rather than calling unlike resources matched.

## Next Member C milestone

Implement a small declared correlated/capacity-limited setting in the nonlinear adapter, retaining the untrained initializer and D0 acquisition checks. Freeze one setting with real pressure headroom before comparing sleep and Hebbian variants. The dense calibration is a feasibility instrument, not the selected nonlinear setting. Teammates can integrate adapters with the current state/scheduler contract while this is calibrated.

The team should record whether this typed suite is the common mechanism test, the chosen pressure capacity/representation, and which resources define the main matched comparison. Keep the complete sequence-model baseline separate until a stronger encoder/association curriculum passes fresh two-key selection. Future subword training must save a compatible tokenizer/embeddings/output head, measure throughput and context coverage, and use a finite compute/configuration budget. Byte-pilot costs do not predict paper-scale FineWeb-Edu training costs.

For the interim report, there is now concrete implementation, acquisition, persistence and interference evidence to report, together with the unresolved full-model and method-efficacy gaps. Current code/results are local on `kv-representation-control`; no shared document or GitHub publication was changed. [HANDOFF.md](HANDOFF.md) records artifact recovery and closed compute status.
