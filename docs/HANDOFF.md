# CLMM working handoff

## Resume here

**Completed November 15, 2026: reference sleep/Hebbian adapter integration and bounded shared pilot.** Local branch `adapter-integration`, frozen model/runner source `a41bb097af3e7eed356d13e24e4a0d5bc6677b14`, parent `e3b744c`. Start with [ADAPTER_INTEGRATION.md](ADAPTER_INTEGRATION.md), [MEMBER_C_STATUS.md](MEMBER_C_STATUS.md) and `results/adapter-integration-2026-11-15`. No task process, GPU allocation, pending connection check or background monitoring remains.

Zeyuan is Member C: shared pretraining, weights and evaluation. A owns sleep, B owns Hebbian/routing methods. Fetched team main1e8de19 (only remote branch) still contained empty sleep/Hebbian directories. Current implementations are explicitly working REFERENCE component adapters, not uploaded teammate algorithms or full Titans/MAC integration. The complete byte-model two-key transfer failure remains unresolved.

## What exists now

- `clmm/sleep/adapter.py`: generic SleepSession over functional write/read memory and bounded literal scheduler; atomic combined model/policy handoff, midblock resume, no full history/read labels.
- `clmm/hebbian/adapter.py`: four associative modules with learned encoders/decoder,W0,alpha,eta; local bounded Hebbian EMA or matched analytic gradient; dense or top-one fixed key-prototype access. Inactive modules remain unchanged, including no decay.
- `clmm/pretrain/adapter_registry.py`: strict own checkpoint format and explicit slow-only forks. Shared neural slow weights are incompatible; only exact literal codebooks transfer to the new architecture.
- `scripts/adapter_integration_experiment.py`: common causal data, acquisition/revision/reset/rolled controls, replay budgets, combined state persistence and resource measurements.

## Scientific outcome

All8trainedsystems pass own short128/128 gates and own K8D0 admission; denseK16 fails, top-oneK16 passes63/64 both seeds. At K8D64 online stable recall: gradientdense6.25%,gradienttop-one36.72%,Hebbiandense5.47%,Hebbiantop-one21.48%.

Primary Hebbian access effect is+16.02percentage points [8.98,23.44], but access helps gradient even more(+30.47points). Independently trained gradienttop-one beats Hebbiantop-one by15.23points. Do not call this a Hebbian advantage. Top-one changes both reading and writing plus active update/decay exposure; independently trained encoders/eta/alpha also diverge.

Fixed-slow-weight forks hold gradientdense learned tensors identical, start newzeroH, then vary rule/access: gradientdense6.25%,gradienttop-one47.27%,Hebbiandense6.25%,Hebbiantop-one43.75%. Frozen top-one rule gap(Hb−gradient)−3.52points[−7.81,.78] is uncertain. Exactshamfork repeats7296savedscorevectors bitforbit. No frozenfork was promoted as a selected winner.

All fixed2seed K8D64 secondary P−C timing intervals include/touchzero, including the neural anchor. No sleep-timing advantage is established. Hebbiantop-one initial24.22%→trained21.48% has no established late gain; gradienttop-one24.61%→36.72% improves. Preserve initializers and short/late accuracy/MSE separately.

## Evidence and recovery

Output root: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-adapter-integration-2026-11-15`.
Work scripts: sibling `work/clmm-adapter-integration`.
Full recovery ZIP: sibling `outputs/CLMM_Adapter_Integration_2026-11-15.zip`; adjacent verificationJSON records observed CRC/every-file SHA and final handoff commit. Archive includes full branch bundle, unchanged baseline tag, supplied neural checkpoints, all8 initial/final models/optimizers, training data, observed plans/rawscores, fork outputs and source snapshots. Compact Git results omit weights/full raw arrays.

62 integrationtests pass; independent analytic/state proof passes. Main audit160768scorevectors/1107712causalwrites/2960serializationbatches. 19declared invocations include1throughput,8maintrain,2NNanchors,8forks; no skips. Actual main training1431552labels/2239744writes, not planned arrays saved again in evalfolders. Required-file coverage815checks and22pipeline sessionbundles includes2throughput bundles; optimizerpayloads9=8main+1throughput. Source/protocol pinned before main outcomes.

Selected rechecks cover all18mainmodels:3616savedpredictions+scoresexact onGPU; CPUallpredictionsmatch,maxcoordinateerror9.54e−7. Real SleepSession resume/pure reads/unchangedslowweights and RNG verified. These are inference/artifact checks, not retraining or complete-grid numerical reproduction. Three measured figures rendered and visually inspected.

## Compute closeout: resolved

Current job13918002,ins081,A6000,4CPU/16G: explicitlyCANCELLED after16m50s(0.281allocatedGPUh), main12m48spipeline andGPUrecheck completedexit0. Full artifact transfer extracted and SHA matched. Tar emitted `.: file changed as we read it` because outputwas inside root; archive stepExit1 is preserved, not treated as a failed model. File-level audit confirms expected artifacts. Raw current scheduler records and transfer proof are saved.

Prior13914782 is now observedTIMEOUT after1h3s, no active job; optional portableGPUcheck passed2432queriesexact and is retrieved. This supersedes e3b744c's unknown status, with addendum under prior outputs and current `prior-job-closeout.json`. No renewal, desktop helper selection change or unrelated cancellation occurred.

## Next work and team choices

Replace/extend reference adapters with A/B's actual modules using the shared functional API and exactcheckpoint binding. Agree fixed versus learned routing, top-k/modules, postsynaptic clamp/EMA versus another local rule, and full sequence-model bridge. Test another one-module allocation/decay control on fresh histories before attributing access benefit to content routing or biology. Keep rule/access and training effects separate.

Member C should next specify a longer-horizon training check with explicit writes/labels/optimizer budgets; short training does not guarantee late recall. Do not tune these same90623 histories or choose the favorable frozenfork as a final model. The earlier correlated control ([NONLINEAR_CORRELATED.md](NONLINEAR_CORRELATED.md)), orthogonal positive control ([KV_CONTROL.md](KV_CONTROL.md)), and byte fitting/transfer record ([FIXED_SET_GENERALIZATION.md](FIXED_SET_GENERALIZATION.md)) remain distinct milestones. Full paper-style roadmap and paper-baseline-v1 are not complete.

No Google Doc edits, teammate messages, push/PR/merge occurred. Preserve baseline-v0.6.0 at35a549f137548d48aecc26e33c6f8a34a6efc97c. Append substantive work to EXPERIMENT_LOG.jsonl; inspect real files/Git/scheduler rather than assuming this snapshot remains current.
