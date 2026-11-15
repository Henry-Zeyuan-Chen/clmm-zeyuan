# Member C: shared training and evaluation

Updated November 15, 2026. Zeyuan owns pretraining feasibility, reusable baseline weights and common evaluation. Member A owns sleep consolidation; Member B owns selective Hebbian/neural-memory updates. The proposal asks whether those interventions improve new-association acquisition, unchanged-fact retention and genuine revisions. Sleep timing needs matched replay budgets. The Hebbian track must distinguish its update rule from selective routing.

## Latest result: qualified nonlinear interference regime

The [nonlinear correlated-key experiment](NONLINEAR_CORRELATED.md) now supplies rank32 trained component initializers with short/fresh acquisition and revision checks, plus meaningful delay interference. K8 and K16 pass actual D0 admission in both checkpoints; K32 remains diagnostic. The finite rank sweep stopped at its first candidate, before any replay outcomes.

At K8,D64 stable accuracy, the trained mean is39.06% online,35.35% continuous and34.77% periodic. Periodic-minus-continuous is−0.59 percentage points,95% paired interval[−2.15,+0.98], so there is no established timing advantage. Exact untrained anchors average29.10% online; outer training improves delayed classification despite already-perfect short accuracy, while late coordinateMSE worsens. These are128 paired histories for two fixed checkpoints/one codebook, not population training uncertainty.

Use the portable correlated loader to preserve saved literal codebook vectors across framework environments. Strict GPU inference and portable CPU streaming/state-resume checks pass. Handoff/log record the unverified optional GPU check and final scheduler status after SSH timeout; the main results are fully local.

The next shared milestone is actual sleep/Hebbian adapter integration against this frozen K8/K16 regime. Keep update rule, routing, buffer content and timing ablations separate, and retain untrained anchors, acquisition gates and physical resource accounting. Complete byte-model transfer remains unresolved.

## Earlier evidence and its limits

The [shared byte-language pilot](SHARED_PRETRAIN.md) provides two complete compact model checkpoints and a modest language-context memory effect. The [fact/revision evaluation](FACT_REVISION.md) supplies causal matched schedules and exposes failed acquisition. The [paired-query diagnostic](PAIRED_ACQUISITION.md) improves single-fact recall and revision, but fails two-key selection even when attention can see the facts.

The [fixed-set fitting and assignment-transfer diagnostic](FIXED_SET_GENERALIZATION.md) establishes finite-set trainability: all four models fit 32/32 contexts. Transfer still fails. Fresh-assignment follow-ups score only 1/128 and 3/128 both-correct on training-disjoint histories. These compact lucidrains-based pilots do not establish paper-scale Titans pretraining, architecture impossibility or sleep/Hebbian efficacy. The typed component success does not promote those byte checkpoints into retention baselines.

## Shared contract for the two method tracks

1. **Weights and provenance:** retain initial/final checkpoints, architecture, actual source/data identity, optimizer and training budget. Memory states require the corresponding slow weights. Byte weights, typed-memory weights and new Hebbian architectures need explicit loaders/adapters; matching dimensions alone is not compatibility.
2. **Evaluation:** use the same observed streams, key-only reads, fixed decoder, genuine revisions and separate stable/revised/recent metrics. Gate on short acquisition and per-K D0 performance before interpreting delay. Keep slow parameters frozen and verify read purity, reset/rolled controls and persistent state behavior. The fixed four initial targets form the delay-curve denominator.
3. **Replay:** continuous and periodic must see the same eight-pair FIFO, same frozen samples and same number/order of extra writes, differing only in timing. Report repeat-current and lower-budget online anchors. Record actual adaptive tensor storage, slow parameters, replay pairs and update opportunities separately; equal writes are not necessarily equal compute or nonzero gradients.
4. **Hebbian comparisons:** implement the update rule behind the same API first, then add selective routing as an ablation. Expose extra slow training, state tensors or representations rather than calling unlike resources matched.

## Next Member C milestone

Maintain the qualified correlated component baseline and integrate teammate adapters with the common write/read/state contract. Freeze new changes before fresh evaluation; do not reselect on the current replay histories. A later buffer-coverage/content ablation can test why recent FIFO replay harms older stable facts under this rule. Keep complete sequence-model encoding/pretraining as a separate, acquisition-gated milestone.

The team still needs to state which resources define its main matched comparison: writes, adaptive bytes or measured compute. The interim report can now cite actual calibration, untrained controls, persistent-state checks and an informative non-ceiling timing result, with the full-model and method-efficacy gaps explicit. See [HANDOFF.md](HANDOFF.md) for recovery and the outstanding cluster-status check.
