# Independent key/value-binding result review

The atomic-key encoding did not repair reusable key/value binding under the fixed 2,000-step recipe. All four terminal checkpoints fail acquisition admission. This is a valid negative diagnostic, not a retention or sleep result.

## Prespecified outcomes

Every entry is both requested-key answers correct out of 256 worlds. Admission requires at least 205 in each of the six groups.

| Parent / encoding | Train | Train swap | Dev | Dev swap | Dev heldout | Heldout swap | Joint four: train / heldout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 68101 / digits | 2 | 0 | 0 | 1 | 1 | 2 | 0 / 0 |
| 68101 / atomic | 0 | 0 | 0 | 0 | 0 | 1 | 0 / 0 |
| 68105 / digits | 1 | 2 | 1 | 0 | 1 | 0 | 0 / 0 |
| 68105 / atomic | 1 | 0 | 0 | 2 | 0 | 0 | 0 / 0 |

Zero of 24 gate cells pass. All four models score 0/256 joint-four on both co-primary endpoints. Both atomic-minus-digits effects are 0.000 percentage points, with prespecified adjusted 97.5% percentile-bootstrap intervals [0.000, 0.000]. These intervals degenerate because every observed joint-four outcome is zero. They are not evidence of equivalence, general success-rate precision, or a capacity limit. Inference remains conditional on the two fixed parents and sampled worlds.

## What the saved outputs show

On fresh training-key worlds, every answer is one of the two observed values (512/512 per checkpoint), but 253–256 of 256 pairs predict the same byte for both requested keys. Individual-answer accuracy is 50.00–50.20%. Query-only accuracy is 6.45–7.42% for training keys and 5.47–8.98% for dev keys. The model uses context values while failing to select the requested association.

Zeroing retrieved outputs changes 55–136 of 512 training-key predictions per checkpoint while accuracy remains 49.61–50.20%. This demonstrates forward sensitivity, not useful key-conditioned recall. Normal/reset score vectors agree exactly; with 95-byte prefixes inside one 128-byte segment, that reset is a parity check and does not establish memory independence. Retrieval zero preserves within-prefix updates and changes later activations; it is not a separately trained attention-only baseline.

Descriptive training windows show 100% accuracy over the last 32 K1 updates in all cases. Over the final 100 K2 updates, accuracy is 49.81–50.06%, cross-entropy 0.731–0.743 nats, and requested-key prediction changes remain rare. These windows summarize retained traces and did not select checkpoints.

## Validity, scope and next decision

Independent audits pass all 81,920 main saved 256-way answer vectors, four initial and four terminal checkpoints, four recovery checkpoints, and eight optimizer states. Maximum metric recomputation error is 1.018e-6; predicted bytes agree exactly. Separate engineering validity covers 2,560 GPU calibration vectors plus 80 CPU smoke vectors, for 84,560 saved source vectors overall. Main training totals exactly 8,000 updates, 128,000 answer forks and 11,635,712 input bytes. Recovery checkpoints were not efficacy candidates.

The coordinator performed actual CPU inference on 2,560 selected queries across all eight models and all 20 group/control combinations, retaining each original GPU batch. An independent saved-artifact audit confirms batch membership, source rows, plan/script hashes and errors: maximum logit error 9.537e-6, probability error 2.283e-6, and zero argmax mismatches. Original tolerances and runtime weight/RNG purity checks pass. This establishes the selected exact-batch numerical check, not full training reproduction or portability of every unsampled score.

The encoding contrast bundles segmentation, key identities and prior byte exposure. Atomic dev identity bytes are task-untrained; decimal dev keys recombine trained digit bytes. The fixed codec JSON is 2,978 bytes of external input preprocessing state, separate from the unchanged 839,248 registered model parameters. Both current arms also share a smaller key pool and no-revision curriculum than the earlier study; old-versus-new differences cannot isolate encoding.

The initial one-GPU runtime gate failed before main work. Its outputs remain preserved. A documented execution-only amendment used one replacement two-GPU allocation with unchanged scientific packets and selection. The replacement runtime gate passed; all four terminal cases were retained. Cluster closure records both allocations cancelled, no active user jobs, and 0.551111 allocated GPU hours in total.

The future exclusion union contains 100,526 unique canonical ordered key/value histories, independently reconstructed from actual presented raw bytes plus the supplied 85,123-signature prior union. Original launch packets remain unchanged. The prior union is preserved at provenance/prior_exclusion_signatures.json in the package root.

The next bounded design can separate record selection from value decoding. These results do not identify a failing layer, prove insufficient parameters, qualify atomic replay integration, or establish retention, sleep, Hebbian updates, cross-session cache persistence or corruption recovery. No additional training was added in this session.
