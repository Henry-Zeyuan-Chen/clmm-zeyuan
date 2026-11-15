# CLMM working handoff

## Resume here

**K16/multiple-revision comparison COMPLETE (November 15).** Branch `sleep-k16-multirevision`, frozen executable source `06c0a35b543c25437659765128b81718e5bfa1c5`. Read [SLEEP_MULTIREVISION.md](SLEEP_MULTIREVISION.md) and `results/sleep-multirevision-2026-11-15/analysis/result_interpretation.md`. All twelve existing terminal local-delta checkpoints were evaluated, both parents and all five random maps retained; no optimizer steps added. Main CPU time1121.71s, complete engineering/prepare/main pipeline1268.08s under3300s. No local inference remains.

At D128, stable/revised accuracy is84.88/93.89% online and equal-write repeat-current,83.31/84.22% continuous,83.08/81.76% periodic. Primary P−C stable−0.2295pp97.5%[−0.4736,−0.0244], revised−2.4609pp[−4.0527,−1.0352]. Both reference replay schedules trail repeat-current; no sleep benefit is established. The stable effect is small and its random-family secondary interval crosses zero.

All twelve common short/neutral gates pass; neutral K16 six-target all-correct100%. Every C/P ownD0 gate fails, compared with all online/repeat gates passing. D0 revised recall71.48% C/60.94% P versus100% controls; all revised D0 errors are penultimate versions. D0 is after56post-prefill arrivals, not neutral acquisition. Curves can recover byD64, so label this revision/interference rather than admitted retention. Do not drop damaged cases or call these checkpoints incapable of neutral acquisition.

Data91123 has128paired base worlds,16initial facts,3genuine changes per revised key, phases0/2, final tails0/64/128. Six original targets include2early stable,2initially replay-eligible stable,2revised. Early targets never enter FIFO8 after prefill. Query metadata is observer-only. All1280literal histories independently regenerated;87653training/prior signatures excluded. Main split is now consulted development data.

C/P have identical observations, exact same literal extra `(source,key,value)` sequence, fixed slow tensors and384writes/stream atD128, plus64KiBfast state and8+4bounded records. Online has200writes; repeat-current384withdifferentcontent. Phase0 stale applications across128worlds:336C/385P; phase2:156C/385P. Same final-obsolete sample totals, different application-time validity. This is mechanism evidence, not an isolated causal mediation estimate.

359tests pass. Full saved arithmetic310272vectors/8474112writes/3456resourcebatches/55296plans passes; all44source files match06cGitblobs. Coordinator independently rechecks primary bootstrap and totals from rawCSV, exact agreement. Separate CPU inference reproduces5184scores/all12checkpoints exactly, combined-session resume atprefill/firstarrival andindices21/22, pure reads and unchanged slow/RNG. FivePNG/PDF/SVG figures reviewed. This is selected actual inference plus full saved arithmetic, not full-grid numerical re-execution.

Output root: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-sleep-multirevision-2026-11-15`; launch/source checkout: sibling `work/clmm-sleep-multirevision`. `packet/models.json` authenticates all12weights with configs, tables, slow hashes and original training. `scripts/revision_local_experiment.load_checkpoint` and `clmm.sleep.SleepSession` supply the shared interface. Strict prepared source bytes require the isolated06c checkout for reruns, not a later documentation HEAD. Full recovery ZIP is adjacent `CLMM_Sleep_Multirevision_2026-11-15.zip`; adjacent verification binds final handoff commit and every-fileSHA/CRC. Compact Git results omit bulky raw data/weights.

Insomnia portability attempt CLOSED: job13921270 onins081, oneA6000/4CPU/16G. No proof output before bounded timeout; Python was blocked in kernelI/O (`cxiWaitEventWait`), GPUutilization0%. Underlying cause unidentified; this is not a measured numerical/model failure. Explicitly CANCELLED after7m36s (0.1267allocatedGPUh), final queue empty. GPU proof remains unverified. `cluster/metadata.json`, `scheduler-final.txt` and later `scheduler-confirmed.txt` preserve the failure/cleanup; the first snapshot briefly showedCOMPLETING, the later live check confirms nojobs. No renewal, unrelated cancellation or helperselection change. Main CPU experiment and exact selected CPU proof are complete.

Next team choice: keep literal frozen replay as the documented control, and test a causal freshness policy using only the current bounded FIFO (refresh a sampled key if still present; explicitly define the eviction fallback). Do not look up query answers or introduce an unaccounted full-history version map. A freshness intervention changes applied content even when writes/storage match; use a new declared split and report the timing×freshness factorial separately. A owns the intended sleep mechanism, B the biological update, C shared baselines/admission/evaluation. Full Titans/MAC and byte-model fresh-assignment transfer remain unqualified. No GoogleDoc edits, external teammate messages, pushes/PRs/merges; baseline tag remains35a549f.

## Historical revision-local study

**Reinforcement follow-up also COMPLETE:** all24 terminal weights evaluated on fresh91023, R1/4/16 ×D0/64, no new training/GPU.351.51s CPU under600s. Original trained joint primary remains false; the additional stress result is exploratory. No active local inference remains.

**Completed November 15, 2026: key-local revision and matched training.** Branch `hebbian-revision-local`; executable source `c9f5c153f786f1b82ac383d01c1d18c6c595946b`, parent handoff a98c849. Start with [REVISION_LOCAL.md](REVISION_LOCAL.md), [MEMBER_C_STATUS.md](MEMBER_C_STATUS.md), and `results/revision-local-2026-11-15`. No active task process, allocation or monitoring remains.

Zeyuan is Member C: shared pretraining, weights and evaluation. A owns sleep; B owns Hebbian/routing. Last fetched team main1e8de19 had empty method folders, so these are working REFERENCE components. Full Titans/MAC, paper-faithful roadmap and byte-model fresh-assignment transfer remain separate.

## Implementation and contract

`clmm/hebbian/revision.py` crosses active bulk EMA and key-local erasure. With x=pre, y=observed post, P_xH=(x/max(||x||²,1e-8)) outer (xᵀH), update is clamp(d*(H−e*P_xH)+eta*x outer y). EMA d=1−eta/e=0; additive1/0; eraseEMA1−eta/1; local_delta1/1. All inactive modules remain protected. Same first write from zero H. No revision flag, query labels, history lookup or extra buffer. Projection overwrites raw xᵀH when unclipped/epsilon inactive, without a general decoded-output guarantee.

New runner `scripts/revision_local_experiment.py` supplies dedicated `clmm.revision.local.v1` checkpoints/load_checkpoint; fast state is schema3, with the generic SleepSession API. Match fast states with exact slow checkpoint SHA; config fingerprints do not hash learned values. Architecture-specific loaders remain necessary.

Persistent traces64KiB/stream, table512bytes, parameters81921, one retained module update per observation. Projection adds arithmetic and four candidates are computed before masking. Matched writes/state/examples/labels/optimizer opportunities do not mean matched FLOPs. EMA preserves previous behavior exactly.

## Observed results

48 frozen cases: 2 dense-trained Hebbian parents90511/90515 ×6 tables(content+all5fixedrandom90701–90705) ×4rules. All nine original tensors identical, zero additional optimization. Fresh90823 excludes training/prior90723. Local delta revised100%, stable90.04% content/92.42% random versus EMA revised92.97/93.75%, stable34.96/41.25%. All four diagnostic inequalities and all12 candidate short/K8/K16 gates pass. Additive preserves stable scores but fails revision; eraseEMA repairs revision but retains bulk forgetting. Most compound stable gain is bulk-decay removal, not erasure alone.

24 matched training cases start from 2 ORIGINAL UNTRAINED step0 initializers ×6tables ×EMA/local_delta. Identical paired initial tensors; shared90802 data/order,1200AdamW steps,B32,lr.001,wd0,clip1, terminal checkpoint only. Initial/final90923 histories are paired and disjoint from all training/prior/frozen histories. No outcome selection.

Trained K8D64 stable: local95.51% content/97.07% random versus EMA24.02/28.52%. Differences+71.48pp97.5[65.23,77.73] and+68.55[65.66,71.41]. Both rules are100% short revision initially/finally. **The trained joint primary criterion is NOT MET:** revision superiority0[0,0] fails; stable constraints/gates pass. Do not relabel this primary success or infer population equality from a zero empirical interval.

All24initial and all24terminal conditions pass short/K8/K16D0 admission. Candidate terminal D0 four-target all-correct100% at both K values; EMA minima99.22% K8,96.09% K16. Local delta initial stable88.87/91.64%; training adds6.64pp95[3.71,9.96] content and5.43[3.87,6.99] random. EMA has no established late training benefit. Most candidate advantage predates training. The old dense-parent revision failure is not inherent in every Hebbian initializer.

Primary thresholds were fixed: revised lower>0 and stable lower>−.03 in both routing families. Two-sided97.5% lower.0125 implements four one-sided Bonferroni constraints.5000PCG64 draws90804 resample128 history IDs jointly across fixed parents/maps. These are conditional history intervals, not population seed/map uncertainty. Original short gates/per-KD0 remain unchanged; comparator revision exemption only for repair was declared before outcomes. All failed controls are retained.

## Checks and recovery

Additional stress: R16D64 revised local99.61%/98.87% content/random versus EMA.39%/2.03%; stable96.68%/97.58% versus18.36%/22.46%. EMA R4/R16D0 replacement already fails0%, all3072R16D0revisederrorsobsolete, whilelocal100%. Thus interpret exploratory revision/interference, not admitted EMA retention. Allshort gates pass; stronger D0allfour EMA gates fail and remain reported. AllR/Dcurves retained; no new recipe selection or primary rescue. Nominal95bootstrap91004focusbounds in analysis/stress.

Stress raw156672vectors/1078272writes/2880resourcebatches audited; coordinator reconstructs streams/latesttargets and focusCIs,168additionalcasefiles. SelectedactualCPUrecheck1920queries/all24existingtrainedmodels exact. Main+stress totals795648vectors/4408320writes; optimizerstepstotalunchanged28800. Driver snapshot/hash f7162bbf3d86be1da4dd663e52c2ca97d1434c1087e380ffc1876c35556a8ab3 lives in stress/driver_snapshot.py; core remainsc9. Fresh91023excludesallpriortraining/development. This is an exploratory post-outcome follow-up, not original primary success.

181 relevant tests pass. Independent formula/source audit covers48 constructions/155tracked source files. Complete raw audits total638976scorevectors,3330048causal evaluation writes,11520resource batches. All96checkpoint payloads and24optimizer states checked. Main optimization totals28800steps/6707712observations/4313088answers; engineering separate.792required case files and42executed sources per phase reconcile with c9 Gitblobs.

GPU/WindowsCPU selected checks each cover10752queries/all96weightsets, actual combined-session resume, pure reads, unchanged slow weights/RNG and allsavedpredictions matching. Max errors9.54e−7GPU/3.58e−7CPU. This is selected inference plus full saved arithmetic, not full-grid numerical reproduction or retraining. Five data figures exported PNG/PDF/SVG and visually inspected.

Full root: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-revision-local-2026-11-15`.
Work: sibling `work/clmm-revision-local`; `source-checkout` is isolated LF c9.
ZIP: sibling `outputs/CLMM_Revision_Local_2026-11-15.zip`; adjacent verification binds every-fileSHA/CRC and final handoff commit. Includes4original parents,48frozen weights,24initial/final pairs/optimizers, allraw data/sources/engineering/analyses and fullbranchbundle.

Use `checkpoint_admission.json`. Frozen payload training.step0 means no ADDITIONAL optimization; its parent trained1200steps. Trained-phase initial payloads are original untrained outer parameters with supplied structured initialization/codebook. Do not mix these meanings. Keep exact slow SHA with state. Analysis READMEs require full raw root; compact Git index omits weights/raw arrays. Clone bundle/check out c9 and relocate launch paths to reproduce. Main Torch2.11/Python3.11 uses GPUtraining/CPUeval; frozen/Windowsproof Torch2.15/Python3.13.

## Compute closed

CPUjob13919399 cancelled PENDING/Resources with zero elapsed/allocation after multi-hour estimate. Prospective hardware-only amendment moved frozen48 to local CPU (536.02s), same isolated source/data. GPUpreflight projected1189.77s below2700; mainmatchedphase1022.75s, fullpipeline1150.50s, all24complete. No quality-based changes.

GPUjob13919414(ins080,A6000,4CPU/16G) explicitlyCANCELLED after23m01s,0.3836allocatedGPUh includingCPUeval/checks/transfers. Tasksteps COMPLETEDexit0; wrappercancel intentional. Frozen upload/trained download hashes match; tar exits0. No active userjobs, helperselection change, renewal or unrelated cancellation. scheduler-final.txt/cluster-closeout.json authoritative. Previous13918985,13918002cancelled and13914782TIMEOUT remain resolved.

## Next Member C work / team choices

Use qualified local-delta weights and EMA comparators for intended A/B adapters, retaining both initializers and fixed random controls. Reinforcement diagnostic is now available; next test multiple genuine changes and admitted K16pressure, then continuous/periodic replay with identical weights/routing/decay and matched extra writes. Simple short revision is at ceiling. Do not tune consulted90823/90923/91023 or choose favorable maps.

Team should decide whether known delta-style erasure is a strong control or part of B's intended method, and which forgetting clock matches that method. It is not biological novelty. Full sequence/Titans and paper-baseline-v1 remain incomplete. No GoogleDoc edits, teammate messages, pushes/PRs/merges. Preserve baseline-v0.6.0 at35a549f137548d48aecc26e33c6f8a34a6efc97c. Append EXPERIMENT_LOG.jsonl and inspect real files/Git/scheduler on resume.

Prior evidence: [ROUTING_DECAY.md](ROUTING_DECAY.md), [ADAPTER_INTEGRATION.md](ADAPTER_INTEGRATION.md), [NONLINEAR_CORRELATED.md](NONLINEAR_CORRELATED.md), [KV_CONTROL.md](KV_CONTROL.md), [FIXED_SET_GENERALIZATION.md](FIXED_SET_GENERALIZATION.md).
