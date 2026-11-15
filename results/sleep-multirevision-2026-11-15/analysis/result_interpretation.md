# Fixed-checkpoint replay timing under genuine revisions

Periodic replay performed worse than continuous replay on both prespecified D128 endpoints in this bounded reference experiment. Both replay arms already fail their own D0 six-target gate, so the result is a revision/interference diagnostic. It is not an admitted-retention comparison or evidence for a biological sleep benefit.

All twelve prescribed terminal local-delta checkpoints were retained: two initializers, content routing and all five fixed random maps. No training, checkpoint selection or additional optimizer step occurred. All arms received exactly the same real stream within each paired history. Continuous and periodic replay applied the identical extra source/key/value sequence in the identical selected order, with only its placement among real arrivals changed.

## Prespecified primary results

The estimand weights content and the mean of all five random maps equally, averages two fixed learned initializers and both phases, and uses 128 paired base histories. Values below are accuracy percentages and percentage-point differences. The two primary intervals are two-sided 97.5% paired-history percentile-bootstrap intervals, 5,000 PCG64 draws, seed 91104; they implement the declared Bonferroni adjustment for two endpoints.

| D128 endpoint | Continuous | Periodic | Periodic minus continuous | 97.5% interval |
| --- | ---: | ---: | ---: | ---: |
| Four stable targets | 83.31% | 83.08% | −0.23 pp | [−0.47, −0.02] pp |
| Two revised targets | 84.22% | 81.76% | −2.46 pp | [−4.05, −1.04] pp |

The stable effect is small in absolute size despite excluding zero in the declared interval. The revised effect is larger. These are conditional effects over sampled synthetic histories for the fixed models and maps; the intervals do not describe new training initializers or new random maps. No joint superiority criterion was added after seeing results. Exact values and complete secondary contrasts are in `summary.json` and `contrasts.csv`.

## Qualification and treatment damage

Every checkpoint passes the unchanged short gates and the common neutral K16 gate. All six neutral targets are correct in 100% of histories for every checkpoint. The neutral gate places all three genuine changes per revised key in the common online-only prefill.

Every replay arm's own D0 gate is still reported, with no removal from analysis. D0 means no final distractor tail after the three revision windows and two intervening distractor gaps; it includes 56 post-prefill observations.

| Arm | Own D0 gate passes / checkpoint-phase cases | Range of six-target all-correct |
| --- | ---: | ---: |
| Online | 24/24 | 89.06–97.66% |
| Repeat current | 24/24 | 89.06–97.66% |
| Continuous | 0/24 | 32.03–58.59% |
| Periodic | 0/24 | 28.13–32.03% |

Thus the checkpoints can acquire and replace the facts under the common neutral protocol, while the replay treatments damage final revision correctness. Every revised-target error in the D0 continuous/periodic cells predicts the immediate penultimate value. The full `gates.csv` retains every initializer, map, arm and phase.

The revised-point accuracies can rise between D0 and D64: continuous 71.48% to 89.22%, periodic 60.94% to 86.23%, averaged across phases. The tail changes interference in the model state; it is not a pure elapsed-time forgetting clock. The complete curves remain visible.

## Controls and mechanism evidence

At D128 the lower-write online anchor and matched-write repeat-current control both have stable accuracy 84.88% and revised accuracy 93.89%. Their identical aggregate accuracies do not establish population equivalence. Both replay policies have lower point accuracies than this matched-write control in the declared grid. The repeat control changes replay content and stores no FIFO, so it is a control for write opportunities rather than an isolated timing comparison.

Across the 128 paired streams, the following counts are independent of the chosen checkpoint:

| Phase | Continuous stale extra writes | Periodic stale extra writes | Final-obsolete query extras, identical C/P |
| --- | ---: | ---: | ---: |
| 0 | 336 | 385 | 1,056 |
| 2 | 156 | 385 | 1,100 |

These stale counts are the same at D0, D64 and D128 because all actual world changes precede the final tail. At D128 each policy applies 23,552 extra writes across the 128 streams. A write is stale only if its literal value differs from the latest actual world value when applied. A final-obsolete extra instead refers to any older version of a queried key, even when that value was still current at the time of replay. Identical final-obsolete literal totals coexist with different application-time staleness because the schedules move the same records across revision arrivals. `replay_telemetry.csv` preserves these counts and denominators.

The secondary D128 revised P−C difference is −0.84 pp in phase 0 (nominal 95% interval [−2.21, +0.23]) and −4.08 pp in phase 2 [−6.64, −1.82]. The larger phase-2 gap is consistent with its larger difference in stale applications. Timing also changes the order of noncommuting overlap writes, so these measurements do not isolate a mediation effect of staleness from every other ordering effect.

Stable early targets at positions 0/1 never enter the replay FIFO after the sixteen-fact prefill. Initially eligible stable targets at positions 12/13 can be sampled while resident. At D128 their C/P accuracies are 79.91/79.65% (early) and 86.70/86.50% (eligible). Initial eligibility also differs in priming recency, so the group difference alone is not an eligibility-only causal comparison. Both groups remain in the stable primary endpoint.

Content/random checkpoints were previously trained separately and their slow tensors differ. Each within-checkpoint timing comparison fixes every slow tensor, but descriptive differences between routing families include those prior training-induced tensor differences. The results concern the fixed 64-key/16-value codebook and reference local-delta update with protected inactive modules. They do not establish full Titans/MAC performance, a map-only routing effect, or biological novelty.

## Independent verification and artifacts

The saved-output audit passes all 310,272 score vectors, 8,474,112 causal evaluation writes, 3,456 resource batches and 55,296 literal plans. The extra-write total is 3,317,760. All raw score argmaxes/MSEs, labels, causal schedules, observer annotations, source pair equality, module-update counts, zero bulk-decay counts, 64-KiB trace state records, 8+4 literal-record limits, reset/rolled source indices, admission tables and bootstrap inputs were checked.

The prepared-data audit independently regenerates all 1,280 literal histories exactly, reconstructs the complete 87,653-signature exclusion union and required old training/evaluation roles, checks regenerated exclusion arrays against the saved original arrays, authenticates all twelve checkpoint copies, and matches all 44 preserved source files to Git commit `06c0a35b543c25437659765128b81718e5bfa1c5`. The learned checkpoints originate from `c9f5c153f786f1b82ac383d01c1d18c6c595946b`.

This reviewer performed full saved-artifact arithmetic and source/data reconstruction, without model inference, retraining, GPU use, core-code edits or commits. The coordinator's separate actual CPU/GPU inference proof supplies its own scope and status; it should not be conflated with this full arithmetic audit.

Five numerical figures are exported as PNG/PDF/SVG from the saved source tables and visually reviewed: balanced accuracy, routing families, stable groups, co-primary contrasts, and revised errors/stale replay. The error-stack figure's C/P labels mean continuous/periodic. Curve bars are nominal 95%; the co-primary contrast bars are 97.5%. Source/output hashes are in `figures/manifest.json`, and exact reproduction commands are in `README.md`.
