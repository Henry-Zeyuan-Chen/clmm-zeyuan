# Shared fact/revision experiments

Read [research notes](../../docs/FACT_REVISION.md), [primary decisions](analysis/final/decisions.json), and [attention-control decisions](analysis/control-final/decisions.json). Both main models fail the acquisition gate, so the replay comparison is inconclusive. Neither attention control solves two-key recall. Execution success and artifact verification do not imply successful fact acquisition or a replay benefit.

Raw logits, reconstructed input streams, training arrays, checkpoints, optimizer states, source snapshots, preserved failures and the full Git bundle are in `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-fact-revision-2026-11-14` and its adjacent `CLMM_Fact_Revision_2026-11-14.zip`. Weights are outside Git. Main source is 646c670; the supplemental control uses ed2a8dc. Both fixed main seeds remain in the evidence.
