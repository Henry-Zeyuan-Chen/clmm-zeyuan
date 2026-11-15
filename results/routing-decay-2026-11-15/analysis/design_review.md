# Prospective independent design review

Written 2026-11-15 18:17 UTC before the new evaluator or outcome artifacts existed in the output root. Repository parent inspected: `65609b3648a8e36b3ad0bee02202f84522a9b550`. Reviewer scope is read-only source review and output-root analysis; no model training, GPU work, source edits, routing-map selection, or additional data selection.

## Decision and estimands

The frozen factorial can distinguish (i) this fixed cosine-key allocation from five prespecified nonsemantic fixed key allocations under protected inactive traces, and (ii) protecting inactive traces from globally applying EMA decay under the content allocation. It directly addresses the old dense/top-one comparison's unequal inactive-decay exposure. It does not separately identify read routing, write routing, content geometry, or collision allocation.

The two co-primary estimands are mean episode-level stable-query accuracy at K8,D64 within the Hebbian dense-parent family, averaged equally over parents 90511/90515:

1. content/protected minus the equal-weight mean of random90701..90705/protected;
2. content/protected minus content/global decay.

Gradient dense-parent replication, routing-by-decay interaction, random-map decay penalties, revised/recent queries, K16, and other delays are secondary. All maps and parents remain visible regardless of gates or favorable effects. A failed own gate makes a contrast an acquisition/interference diagnostic; it does not authorize map replacement or dropping a case.

## What must remain fixed

- Four existing dense-trained parent checkpoints: gradient and Hebbian, seeds 90511/90515. Preserve every learned encoder, decoder, W0, alpha, and raw_eta tensor bit for bit; freeze gradients; begin every history with new zero H. Keep each parent's update rule throughout its 12 conditions.
- One content table and five PCG64 private-key assignment tables from seeds 90701..90705. Construct each random table by permuting the 64 content-table entries, thereby preserving the exact per-module cardinality vector. Use the same fixed table on writing and reading, with one module selected per key and no per-token randomness, value/label access, or history-dependent table change.
- Active update formula, eta, bounds, read aggregation and codebook. Only inactive behavior changes: protected leaves H unchanged, global replaces inactive H with `(1-eta)*H`. The active result should be exactly the old top-one active result on every step, for both analytic-gradient and Hebbian parents.
- All four 64x64 float32 traces remain allocated (65,536 bytes per stream). Association updates are one active module per observed fact. EMA multiplications are one versus four modules by intervention. Total state updates or FLOPs are therefore not matched; existing code also vectorizes inactive candidate calculations, so an active-count claim is not a measured speedup.
- Fresh development histories from 90723, identical across all 48 cases. Three short cells with normal/reset/rolled controls, plus K8/K16 at D0/16/64 with 128 histories per cell. No replay or timing factor. Exclude exact signatures of the planned train-seed-90502 1,200-by-32 histories, and verify the regenerated exclusion set against saved training_data.npz. Familiar atomic keys and bindings do not constitute novel-entity generalization.

## Source checks already completed

`clmm/hebbian/adapter.py` writes all module candidates before selecting active results. Its original top-one route is cosine similarity to literal key prototypes 0,16,32,48 and leaves inactive H untouched. The inherited active update is `clamp((1-eta)*H + eta*change)`, where Hebbian change is `pre outer post` and the gradient change is the negative exact reconstruction gradient in H including alpha. Reads access the same route and average active module outputs. `raw_eta.sigmoid()` is one frozen scalar per parent.

`clmm/pretrain/kv_streams.py` creates a closed 64-key/16-value setting. Within each K, D0/16/64 share identical initial/revision prefixes; distractors cannot revisit initial target keys but can repeat other keys with fixed values. The first two pressure targets are stable and the next two revised. Thus stable loss at D64 reflects interference/decay without direct target overwriting. Pressure K8/K16 are different history populations, so compare conditions within each K rather than treating K8/K16 episodes as paired.

The source supports unchanged-state reads, bounded functional fast state, state method/config binding and saved literal codebooks. The new adapter must extend state/config binding to include table and decay policy, or explicitly refuse cross-policy state import. Parent tensor equality needs a separate learned-tensor/codebook comparison because new routing metadata may add buffers.

Source SHA256 at this review: HANDOFF `5dacaaf5da48d649d05e26344f773dc1e96e637fd34634177db12cae00f759a6`; ADAPTER_INTEGRATION `b492be1ff9dff0b826d9bd02bb08f5a87d36674d0d03464f4672ae4bf3e4a475`; adapter.py `e52c78207bd308e98815e55b7140dc36016c87a40614d628995e6cc25b8409f5`; kv_streams.py `1a62fd8d5d58fab9dcb4ba7e49f1cee8506c5564ce4ebc64f0b6d6f5a2537b9a`.

## Remaining confounds and reporting requirements

1. Global key counts do not equal per-stream module loads. Content routing can change collisions, burstiness, numbers of later writes to a target's module, and exposure to decays. Log each history's module occupancy and, preferably, target-specific same-module versus other-module writes after its final observed target write. These are mechanism descriptors and should not be used to reweight/select primary histories.
2. Dense-trained features avoid selecting content-specialized top-one checkpoints, but the result is still conditional frozen transfer from four fixed dense parents. Module-specific encoders/decoders can differ after dense training. Two fixed parents do not estimate training-seed uncertainty, and five fixed random maps do not estimate the population distribution over routing maps. Show every map and parent as well as their fixed averages.
3. The cosine router is content-based relative to one synthetic correlated codebook. A positive content-minus-random contrast supports geometry-informed allocation in this setting, not natural-language semantic routing, learned routing, a general Hebbian advantage, biological plausibility, novelty, or full Titans integration.
4. Decay protection and content allocation may interact. The content/protected minus content/global contrast is the effect of the inactive-decay intervention at that allocation, not a universal fraction of the old top-one benefit. There is no new dense baseline in this factorial, so do not divide either contrast by the old dense/top-one gap and label it a causal percentage explained. Report the 2x2 routing-by-decay interaction and random-map penalties as secondary.
5. Exact signature exclusion prevents repeated complete training histories; it does not make all subsequences or atomic associations unseen. Preserve the old development results as historical context without combining their samples with the fresh pilot.

## Gates and uncertainty

Freeze gate metrics as episode-level all-correct: acquire2 and revise2 >=0.90; acquire4 >=0.80. The normal-minus-rolled short contrast should use the same paired episode metric and require a lower 95% bound above zero. Report reset separately unless explicitly declared an additional gate before outcomes. For each K independently, require D0 all-four-target accuracy >=0.80 in both parent seeds for a rule/table/decay family. No gate is inherited from the original architecture or from another routing condition.

For the co-primary K8,D64 stable accuracy contrasts, aggregate the two stable queries within each history first; average both fixed parents and all applicable fixed maps within the same history ID; then bootstrap 128 history IDs with replacement, applying each draw jointly to both parents and all maps. Use 5,000 draws with private seed 90704. Report percentile 97.5% intervals (1.25th/98.75th percentiles) for the two co-primary contrasts, giving Bonferroni familywise 95% nominal coverage; 95% secondary intervals are descriptive and unadjusted. Do not bootstrap individual queries, independently resample each condition, or use the five maps/two parents as 10 independent history samples. Report observed point effects in percentage points and absolute accuracies.

The rollout must retain all 48 conditions even if admission fails, preserve partial output when the 3,300-second cap is reached, and make missing cases explicit rather than analyze an outcome-selected complete subset. One fixed allocation/hour is a cap, not permission to renew or retrain.

## Checks required before accepting outcomes

- Exact learned-tensor equality to each parent, table cardinalities/hashes, one-hot read/write identity, and distinct policy-bound state fingerprints.
- Protected/content identity with the old top-one adapter on deterministic traces; analytic active equality and exact expected inactive decay for both rules; no input-state mutation or slow-weight/RNG change; serialization resume.
- Regenerated training arrays versus saved arrays, all64 atomic IDs/bindings familiar claim, fresh signature exclusion, identical history IDs/targets across all cases, intended prefix pairing across delay.
- Raw score/prediction alignment, every case/cell/arm/episode/query, finite values, strict target/category/order, correct causal write counts and per-stream bytes, declared one/four decay exposure, target-state/rolled source mapping.
- Independent recomputation of gates, per-map/parent summaries, co-primary bootstrap and secondary interaction from raw rows, with source hashes and runtime/partial status recorded.

Prospective judgment: proceed as a bounded frozen-transfer mechanism diagnostic after these implementation checks. No protocol-level defect requires choosing a new router, seed, checkpoint, or evaluation history.

## Implementation validation addendum, before main outcomes

The new source was frozen at `61212289d6029da65eeb6a3735ce5ec3894866a9`. Independent CPU checks of all48 combinations pass: exact parent tensors, fixed private tables/cardinalities, zero64KiB H, exact active protected/global equality, expected inactive traces, independent modulewise analytic formulas (maximum coordinate difference1.49e-8), unchanged inputs/slow weights/RNG, pure reads, exact state serialization and cross-policy import rejection. Content/protected matches old top-one traces and scores exactly for12 writes per parent. These are CPU component checks, not GPU or training reproduction.

The evaluator's added exposures are computed after the model updates. Online-only plans preserve causal observed facts, with no query labels passed into writes. Formal gates are now explicitly episode-all-correct, with acquire2 normal-minus-rolled bootstrapped over8 independent16-history rolling cycles. Reset is diagnostic.

The implementation computes association and read candidates for all four modules before masking. The global intervention also constructs a full decay candidate before selecting inactive entries. Therefore the reported one/four decay applications and one active association/read are **retained state/read exposure counts**, not counts of arithmetic multiplications, kernel work or FLOPs. No computational-saving claim follows.

The saved original training arrays agree exactly with regeneration for all four parents:38,400 samples contain38,393 distinct histories. Every parent saw all64 keys,16 values and1,024 key/value bindings. Fresh90723 histories have zero complete-history signature overlap. The content key-cardinality vector is [14,20,9,21], preserved by every random table. See `parent_input_audit.json` and `adapter_independent_audit.json` for machine-readable evidence.
