# Supplemental attention-control audit

This separate auditor does not modify the main v1 analyzer or its protocol. It imports only its independent low-level artifact/statistics helpers and verifies that the main helper file retains SHA256 `3718b7b68a3f766c350c241599c9a88363b00d85d0b6171598fef08274bbf025`.

```text
python analyze_attention_control.py --root ROOT --out ANALYSIS/NEW --expected-commit ed2a8dce3ab1eded7165a32034f7df311a594df9
```

The output must be a new directory below the existing analysis directory. `--parent PATH` optionally overrides the original shared seed-68101 checkpoint path. Default: `ROOT/../clmm-shared-pretrain-2026-11-14/lm-68101/checkpoint.pt`.

Both `ROOT/attention32` and `ROOT/attention128` must have succeeded with the exact supplemental design: original fixed parent; 1,000 steps; batch 16; training seed 78122; development seed 78123; 64 episodes per short cell; attention geometry 32 versus 128; unchanged remaining configuration. The auditor verifies:

- Every saved initial tensor against the original parent and across arms, including finite values, exact equality, shape and dtype.
- All source snapshots and hashes; final checkpoint lineage/configuration; optimizer state at step 1,000.
- Actual `(1000,16,95)` input bytes, latest-online targets, namespaces, seed/query plans, complete trace and budget; exact training-data equality across arms.
- All 1,536 gate-evaluation rows per arm, by semantic sample identity; raw prefixes, labels, counterfactual/template pairing, full-vocabulary and forced-choice metrics, legal-answer mass and heuristic baselines.
- Exact equality of online/reset logits for every attention128 short query, since all short inputs fit within a single attention segment.
- 5,000-draw episode-bootstrap intervals using seed 78124; the two K2 outcomes remain clustered, and the principal contrast is attention128 minus attention32 on K2/seen/online.

Outputs: `verification.json`, `source-manifest.json`, `estimates.csv`, `contrasts.csv`, and `decisions.json`. Decisions use the 80% accuracy target descriptively, never the main v1 normal-minus-reset gate. They explicitly state that this one-parent exploratory control does not rescue a failed main replay comparison. Neither training nor inference is rerun by the audit.

```text
python analyze_attention_control.py --selfcheck --out ANALYSIS/NEW_SELFCHECK
```

Selfchecks use synthetic fixtures only and cannot certify control results. Failed or missing artifacts produce a nonzero exit and a noncertifying decision file. No figures or empirical control conclusions are generated before the two complete control datasets are available.
