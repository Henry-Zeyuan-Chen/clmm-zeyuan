# Independent result and claim review — 2026-11-16

The frozen raw-output audit passes. Mixed-format training improves the
prespecified average joint4 score on all three new-wording endpoints, with all
three adjusted lower confidence bounds above zero. This meets the registered
three-shift average-effect criterion. The effect is heterogeneous, familiar
performance falls, and **none of the eight terminals passes full acquisition
qualification**. The supported result is conditional transfer improvement for
these four fixed paired cases and these wordings; a generally robust or qualified
ordinary-value baseline has not been established.

## Integrity and numerical evidence

`independent_raw_audit.json` verifies every one of the 350,208 saved full256
confirmation score vectors, all 12 capabilities, all 21,888 actual 16-query
batches, all packet joins/targets/full and legal argmax/correctness, and all
same-input collisions. All 4,608 cue-erased fork pairs have identical logits and
distinct labels. Both engineering evaluations are also rechecked. File/state
hash lineage, all-eight-terminal lock timing, shared packet identity, resource
decision and all eight main training CSV budgets pass. All 48,000 main steps
have finite reported gradients and positive reported memory-gradient norms;
there are 768,000 supervised answer forks and 72,960,000 input bytes.

The maximum independently reconstructed full CE error is 3.866310422040442e-6,
forced CE error 1.1249188638373653e-6, and legal-mass error
5.081652240113144e-7. These are below the unchanged frozen 5e-5 / 5e-6 bounds.
The audit uses NumPy float64 math from the saved float32 logits. It does not
execute a neural model or inspect optimizer/tensor contents. Actual tensor and
selected numerical reload checks belong to the coordinator and are separate
evidence gates.

The initial complete data audit and resource-only rebind establish exact source,
data, wording, initializer and semantic-orbit lineage. The resource amendment
changed deadline handling before main training; all data/recipe hashes were
unchanged. No checkpoint, case, threshold or wording was selected after seeing
current confirmation outcomes. This reviewer did not alter the frozen main
auditor or numerical policy when inspecting these outputs.

## Three registered primary contrasts

Values are percentages of worlds for which both requested answers are correct
in both the original and reversed record views. Means equally weight all four
fixed parent-by-encoding cases on the same 256 development worlds.

| New wording endpoint | Familiar arm | Mixed arm | Mixed−familiar, pp | Adjusted 98.3333% CI, pp |
| --- | ---: | ---: | ---: | ---: |
| Facts only, F2Q0 | 29.88% | 45.90% | +16.02 | [13.38, 18.65] |
| Query only, F0Q2 | 28.52% | 45.12% | +16.60 | [12.89, 20.12] |
| Facts and query, F2Q2 | 20.70% | 41.02% | +20.31 | [17.25, 23.34] |

Each adjusted lower bound is positive. These intervals use one shared direct
PCG64/92004 5000×256 world-index matrix, after averaging the four paired case
differences; exact .05/6 and 1−.05/6 linear quantiles implement the prespecified
three-comparison adjustment. Exact values, every per-world difference and the
saved index matrices/hashes are in `independent_raw_statistics.json`.

The mixed arm's absolute mean success is still 41–46% on these endpoints.
Statistical improvement must not be described as high absolute reliability.

## Keep the four cases visible

Each cell below is familiar→mixed **joint4 counts out of 256**. These are fixed
components of the primary average, not four independent training replications.

| Parent / encoding | Familiar F0Q0 | Fact-only F2Q0 | Query-only F0Q2 | Both F2Q2 |
| --- | ---: | ---: | ---: | ---: |
| 68101 / digits | 254→231 | 0→211 | 32→238 | 2→216 |
| 68105 / digits | 239→247 | 230→247 | 177→219 | 174→204 |
| 68101 / atomic | 0→0 | 0→0 | 0→0 | 0→0 |
| 68105 / atomic | 173→78 | 76→12 | 83→5 | 36→0 |

Both digit cases improve on all three new-wording primaries. The 68101 digit
case contributes most of each aggregate gain. The 68105 atomic case worsens on
all three primaries as well as on familiar wording. The 68101 atomic case stays
at zero joint4 across every cell in both key groups and both arms. That case
has no transfer equivalence evidence; it remains a fitting failure. Retaining it
in the mean is required and correct. No uniform benefit across encodings or
parents is established.

All four exact initial S checkpoints, evaluated as V, have zero joint4 and
joint8 in all 72 case/group/cell summaries. They are selector-adapted controls,
not random networks. Their floor supports the need for V adaptation in these
cases, but this design cannot attribute V performance to selector pretraining
without an equally budgeted V-from-language-parent comparison.

## Familiar cost, counterfactual views and controls

On development F0Q0, the fixed-case mean falls from **65.04% to 54.30%**:
mixed−familiar **−10.74 pp**, descriptive 95% interval **[−12.99, −8.59] pp**.
On train-key F0Q0 it falls from 71.88% to 62.11%: −9.77 pp, descriptive
95% interval [−12.30, −7.23] pp. The per-case development changes are −8.98,
+3.13, 0.00 and −37.11 pp in the table's row order. This is a visible familiar
performance cost; there is no basis for a non-degradation claim.

Joint8 requires all answers in all four order/value views. Its development
means show the same broad transfer/cost pattern descriptively:

| Cell | Familiar joint8 | Mixed joint8 |
| --- | ---: | ---: |
| F0Q0 | 64.16% | 51.95% |
| F2Q0 | 28.13% | 44.34% |
| F0Q2 | 25.00% | 43.07% |
| F2Q2 | 17.29% | 39.45% |

These are descriptive secondary point estimates; no primary adjustment is
implied. All per-view BOTH counts and swapped joint4 remain in
`independent_raw_case_cells.csv`. The cue-erased BOTH count is structurally zero
because its forks have identical input and different labels; that zero is a
parity/leakage check, not evidence of a neural mechanism. Across all capabilities
and groups, no-facts controls have eight BOTH successes and cue-erased controls
have zero; control denominators are preserved in the control CSV.

## Qualification is not waived

Each terminal has 90 required cells: four per-view BOTH gates plus joint4 for
each of nine cells in each of two key groups. Thresholds remain 205/256 dev and
103/128 train-key worlds. Joint8 remains descriptive.

| Parent / encoding | Familiar gates passed | Mixed gates passed |
| --- | ---: | ---: |
| 68101 / digits | 10/90 | 89/90 |
| 68105 / digits | 65/90 | 86/90 |
| 68101 / atomic | 0/90 | 0/90 |
| 68105 / atomic | 7/90 | 0/90 |

The 68101 digit mixed terminal misses train-key F2Q0 joint4 at **102/128**, below
103. The 68105 digit mixed terminal misses train-key joint4 at F0Q2 (99/128),
F1Q2 (97/128), F2Q2 (95/128), and development F2Q2 (204/256). These are genuine
registered failures despite some being one world short. Rounding, dropping
atomic cases, selecting a terminal, or relaxing the threshold after this result
would not establish qualification under this protocol. No terminal should be
handed off as an acquisition-qualified ordinary baseline.

## Claims the evidence does and does not support

Supported: with the unchanged 839,248-parameter model, this matched 6,000-update
mixed-format V recipe improves the average fresh-world joint4 score on the
three prospectively frozen new-wording endpoints, conditional on the four
fixed initializer/encoding pairs and shared training stream. It also shows a
familiar-format cost and strong encoding/parent heterogeneity.

Unsupported: uniform format robustness; familiar non-degradation; an admitted
ordinary-value baseline; equivalence in floor cases; uncertainty over the
training procedure; model-size sufficiency or insufficiency; an identified
internal selection mechanism; and retention, revision, corruption recovery,
native32 or sleep/Hebbian efficacy after context/buffer exit.

The sampling unit is the semantic world. Repeated forks, views and models are
not independent replicates. Intervals condition on fixed checkpoints, a fixed
training stream, fixed codec/key pools and two new fact plus two new query
wordings; they do not generalize across random training seeds or an unrestricted
natural-language template population. Fresh-history isolation does not imply
that every individual fact or masked input is novel. Formatting changes bundle
words, padding and key position; the mixed arm also allocates only one quarter
of its contexts to familiar F0Q0. Those facts limit mechanism attribution.

No additional training, threshold changes, checkpoint selection or new wording
is justified inside this completed confirmation. The remaining handoff work is
the coordinator's tensor/reload/closure verification and an integrated record
that preserves both the positive average transfer result and failed full
qualification.
