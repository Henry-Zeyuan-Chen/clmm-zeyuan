# Member C: shared training and evaluation

Updated November 15, 2026. Zeyuan owns pretraining feasibility, reusable baseline weights and common evaluation. Member A owns sleep consolidation; Member B owns selective Hebbian/neural-memory updates. The proposal asks whether those interventions improve new-association acquisition, unchanged-fact retention and genuine revisions. Sleep timing needs matched replay budgets. The Hebbian track must distinguish its update rule from selective routing.

## Latest result: working reference adapters and a shared pilot

The [adapter integration](ADAPTER_INTEGRATION.md) supplies a generic sleep session, local Hebbian and matched gradient components, dense/top-one access, checkpoint registry and one evaluator. These are references because the refreshed team repo had only empty method directories. A/B can replace the backend/controller without changing the observed streams, key-only reads, state checks or metric contract.

All four trained systems qualify at K8; top-one versions also qualify at K16, where dense versions fail acquisition. At K8D64 online stable recall, Hebbian top-one21.48% exceeds dense5.47% by16.02points[8.98,23.44], but gradient top-one reaches36.72%. The positive result concerns access, with inactive decay protection and learned-parameter differences explicit, not a Hebbian-rule superiority claim. Fixed-slow-weight tests support a conditional access effect and leave the top-one rule difference uncertain. Sleep timing comparisons remain inconclusive.

The exact initializers remain important: no late-training gain is established for Hebbian top-one; gradient top-one improves from24.61% to36.72%. Longer-horizon training is a future controlled study rather than an unplanned remedy on these histories. Fully typed component success does not qualify the complete byte model for fresh two-key selection.

62 new integration tests, full artifact coverage, exact GPU and close CPU saved-score checks pass. All artifacts/weights are local and the new allocation is released. The old unverified portable GPU check was retrieved and passed; old job closure is also verified. [HANDOFF.md](HANDOFF.md) records the recoverable source, budgets, closed jobs and next decisions.

## Earlier evidence and its limits

The [shared byte-language pilot](SHARED_PRETRAIN.md) provides two complete compact model checkpoints and a modest language-context memory effect. The [fact/revision evaluation](FACT_REVISION.md) supplies causal matched schedules and exposes failed acquisition. The [paired-query diagnostic](PAIRED_ACQUISITION.md) improves single-fact recall and revision, but fails two-key selection even when attention can see the facts.

The [fixed-set fitting and assignment-transfer diagnostic](FIXED_SET_GENERALIZATION.md) establishes finite-set trainability: all four models fit 32/32 contexts. Transfer still fails. Fresh-assignment follow-ups score only 1/128 and 3/128 both-correct on training-disjoint histories. These compact lucidrains-based pilots do not establish paper-scale Titans pretraining, architecture impossibility or sleep/Hebbian efficacy. The typed component success does not promote those byte checkpoints into retention baselines.

## Shared contract for the two method tracks

1. **Weights and provenance:** retain initial/final checkpoints, architecture, actual source/data identity, optimizer and training budget. Memory states require the corresponding slow weights. Byte weights, typed-memory weights and new Hebbian architectures need explicit loaders/adapters; matching dimensions alone is not compatibility.
2. **Evaluation:** use the same observed streams, key-only reads, fixed decoder, genuine revisions and separate stable/revised/recent metrics. Gate on short acquisition and per-K D0 performance before interpreting delay. Keep slow parameters frozen and verify read purity, reset/rolled controls and persistent state behavior. The fixed four initial targets form the delay-curve denominator.
3. **Replay:** continuous and periodic must see the same eight-pair FIFO, same frozen samples and same number/order of extra writes, differing only in timing. Report repeat-current and lower-budget online anchors. Record actual adaptive tensor storage, slow parameters, replay pairs and update opportunities separately; equal writes are not necessarily equal compute or nonzero gradients.
4. **Hebbian comparisons:** implement the update rule behind the same API first, then add selective routing as an ablation. Expose extra slow training, state tensors or representations rather than calling unlike resources matched.

## Next Member C milestone

Maintain the shared checkpoint/state/evaluation contract while A/B replace the explicitly reference methods with their intended mechanisms. Before a biological claim, separate update rule from access/decay exposure and independently trained representations. Before broader language claims, repair the complete sequence model's acquisition/generalization gap.

Design any longer-horizon curriculum with a fresh evaluation split and declared total observations, answers and optimizer opportunities; changing context length does not preserve every training budget automatically. Keep exact untrained anchors and own admission gates. The interim report can now cite a working integration, controlled ablations and actual limitations without claiming final project efficacy or full Titans reproduction.
