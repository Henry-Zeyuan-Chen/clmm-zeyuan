# Independent prospective review

Scope: complete 839,248-parameter byte transformer plus bounded raw-record sequence replay. Reviewer reads source and saved evidence; no training, inference, accelerator access, core edits, documentation edits, or commit. Review started 2026-11-15 before any new efficacy output was supplied.

## Source checks completed

- Read ml-research-workflow and its evidence/multi-agent contracts.
- Read docs/SHARED_PRETRAIN.md, FIXED_SET_GENERALIZATION.md, FACT_REVISION.md, and BOUNDED_FRESHNESS.md from clmm-replay.
- Inspected clmm/pretrain/fact_streams.py, kv_scheduler.py and scripts/fixed_set_generalization.py.
- Existing original language parents have 839,248 parameters; 128-versus-32 geometry can preserve all learned tensor shapes. Geometry preservation must still be verified for new artifacts.
- Existing fixed-bank fitting succeeds but novel-assignment and heldout-template acquisition fail; no current sequence checkpoint is an admitted retention baseline.
- Existing bounded component result repairs stale-revision damage, while not beating repeat-current. It does not establish efficacy of full-transformer replay.

## Conditions required before interpreting new results

1. Freeze explicit prefill/prime semantics. Old fact_streams.schedule_episode gives the first empty block no extras. StreamingReplay instead requires a nonempty prefill FIFO and draws four samples before the first step. A new wrapper may use either a declared integration protocol; these are not identical schedules.
2. Same parents, initial learned tensors, real histories, ordered sampled keys/RNG, target opportunities, extra-record counts, and input lengths must be paired across literal/bounded training. Values/templates at application may differ by the intended freshness intervention. Repeated-current has equal tokens; online has fewer.
3. Retain a fixed balanced acquisition component throughout curriculum. Fresh assignments over the 600-key training namespace and separate heldout dev-key/template episodes address the old narrow finite bank, but exact raw/semantic context overlap should still be audited.
4. Require full-256 both-correct accuracy on complete two-key episodes and report first/second query accuracy. A prediction changing with the key is insufficient; both answers must be correct. For query-switch probes, render the same wording apart from the key and condition on distinct target values.
5. Pair the value-relabeling counterfactual with the original episode. Target changes must be derived from observed real records, while query wording and keys remain identical. Count jointly correct original/counterfactual answers rather than treating arbitrary prediction changes as acquisition.
6. Keep heldout wording distinct from training templates, and label reused dev keys/templates as development evidence. Reserved test namespace should remain unqueried. Reject exact context collisions before consulting scores, preserving rejected examples.
7. Training at segment128 and evaluating segment32 changes computation geometry. Segment32 scores are geometry-transfer diagnostics; failure there does not isolate memory capacity, replay freshness, or learned-rule validity.
8. Replay payloads must preserve exact source bytes and only parse observed key/value fields. The controller may retain at most FIFO8 plus snapshot4 raw records, no query or full-history lookup. Producer audit history is permitted only outside the controller.
9. Full-transformer raw replay includes attention/embedding/feedforward computation and extra context. Label it a reference intervention; it is neither memory-only sleep nor a faithful paper MAC implementation.
10. Report fixed initial/final endpoints for all four runs. Failed acquisition, timing preflight failures and interrupted jobs remain evidence. If acquisition gate fails, longer histories cannot support retention/replay efficacy claims.
11. Bootstrap whole paired episodes per fixed checkpoint. Two parent initializations do not estimate general training-procedure variance. Report geometry/training-arm and template scopes explicitly; avoid selecting a favorable parent, arm or endpoint.
12. One nonrenewable allocation <=90 minutes requires a timed preflight covering training and planned evaluation, with time reserved for artifacts and release. Byte counts, supervised-answer counts, actual sequence lengths, memory-update opportunities and observed wall time should all be retained.

## Planned independent evidence work

After packet freeze: inspect actual data generation and wrapper source; reconstruct semantic labels and replay chronology independently from raw records; verify train/eval namespaces, templates, fingerprints and pairing; recompute all raw logits' scores, episode aggregates, gates and paired bootstrap intervals; regenerate source tables and figures from audited output. No model execution is included.

Current status: prospective concerns only. Final wrapper/runner/packet and all new efficacy outputs remain unreviewed.

## Prospective refinement received before results

Coordinator now proposes eight independent native-geometry adaptations: 2 original parents x segment32/128 x literal/bounded periodic, all 839,248 parameters, 2,000 updates, 8 contexts/16 answer forks. This supersedes the initial geometry-transfer proposal; each model is evaluated only in its own trained geometry. Fixed curriculum: 256 K1 updates, 1,024 K2 updates, then 720 updates comprising three K2 acquisition batches per one two-revision-plus-two-distractor batch. First two real records are primed; later real arrivals each receive one extra token record in matched arms. K1 supervises the one query twice, making 16 answer forks from 8 contexts.

Expected per-case token budget is 3,646,208 input bytes and 32,000 answer forks; across eight cases it is 29,169,664 bytes and 256,000 answer forks. K2 acquisition is 95 bytes and replay-ineligible; revision is 351 bytes (6 real + 4 extra records + 31-byte query). Acquisition contrasts therefore estimate effects of the training intervention, not a direct test-time freshness change.

Current data-source inspection confirms same wording within query pairs, fresh 600-key assignments, original-plus-swapped exclusion before acceptance, explicit prime2 scheduling and exact raw source retention. Distinct current values are required after every target update, a constraint stronger than only initial/final distinctness. Evaluation arrays must slice by actual lengths; their 351-wide zero padding is storage only.

Prospective co-primary outcomes are bounded-minus-literal both-correct acquisition on fresh training keys/seen wording and dev keys/heldout wording. Average four fixed parent-by-geometry outcomes within episode before whole-episode bootstrap; 97.5% marginal intervals implement two-endpoint Bonferroni adjustment. Keep all eight final-case acquisition gates and separate native-geometry tables. No training-procedure uncertainty claim follows from these four fixed cases.

The new data and runner are not yet frozen; these observations describe files read during implementation. All final source hashes and packet checks remain pending.
