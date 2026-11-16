"""Finalize this session's prose while preserving historical/user documents."""
from pathlib import Path
REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
ROOT='C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-key-binding-2026-11-15'
p=REPO/'docs/KEY_BINDING.md';text=p.read_text(encoding='utf8')
text=text.replace('Status: implementation and prospective review complete; preparation/experiments pending, November 15, 2026.',
                  'Status: completed, November 15, 2026. All four terminal models fail acquisition; the atomic representation did not repair key binding under this recipe. Executed source `5e8c61829ff85ecea3c9314d4ccd8dfef3268b68`.')
text=text.replace('Compute bound: one 45-minute A6000 allocation','Original compute bound: one 45-minute A6000 allocation')
assert '## Completed results' not in text
text+='''

## Completed results

Each cell below is BOTH-correct worlds out of 256. Original/swapped counts are paired, and 205/256 is required in each of the six individual groups. None of 24 cells passes.

| Parent / encoding | Train original / swap | Dev original / swap | Heldout wording original / swap | Joint-four train / heldout |
| --- | ---: | ---: | ---: | ---: |
| 68101 / digits | 2 / 0 | 0 / 1 | 1 / 2 | 0 / 0 |
| 68101 / atomic | 0 / 0 | 0 / 0 | 0 / 1 | 0 / 0 |
| 68105 / digits | 1 / 2 | 1 / 0 | 1 / 0 | 0 / 0 |
| 68105 / atomic | 1 / 0 | 0 / 2 | 0 / 0 | 0 / 0 |

All four models score zero on both primary joint-four endpoints. Atomic-minus-digits point effects are therefore 0 percentage points. The prespecified percentile bootstrap mechanically gives [0,0] intervals at this empirical floor; these degenerate intervals are not equivalence evidence or proof of zero population effect. The substantive conclusion is failure to meet the acquisition requirement, not a precisely estimated null benefit.

The learning traces expose the same shortcut during training. Last 32 single-fact updates reach 100% answer accuracy in all four runs; last 100 two-fact updates average only 49.81–50.06%, with cross-entropy 0.731–0.743. On fresh training-key evaluation, every prediction is one of the two observed values (512/512 answers per model), but the model emits the same byte for both requested keys in 253–256 of 256 worlds. Answer accuracy is 50.00–50.20%. Query-only train/dev controls are near the 16-value chance level. These are observations under this recipe, not proof of an architecture impossibility or of a particular internal cause.

All normal/reset answer vectors agree exactly, as expected for the short within-segment inputs. Retrieval zero changes 55–136 individual training-key predictions per model, while answer accuracy remains 49.61–50.20%. Forward sensitivity alone does not demonstrate useful binding or online-memory dependence. This ablation preserves writes and changes later activations; no separately trained attention-only comparison is supplied.

All 109 scoped tests pass. Independent data reconstruction checks all three packets, 50 source files against frozen Git blobs, and original parent tensors. Full saved-output audits cover 81,920 main plus 2,640 engineering/smoke vectors, 84,560 total. The largest main score recomputation discrepancy is 1.01744e-6; predicted bytes agree exactly. Tensor-only inspection checks four initial, four terminal and four recovery checkpoints plus eight main/midpoint optimizer states. Eight registered template/frequency tensors remain parent-identical without Adam state as expected from the unchanged functional memory implementation; all 8,000 main updates report finite nonzero memory-parameter gradients.

Coordinator raw-CSV recomputation independently agrees on both primary effects/intervals and 8,000 updates, 128,000 forks, 11,635,712 input bytes. Actual CPU reload/inference checks 2,560 selected queries across all eight initial/terminal conditions, using the exact original GPU batches from every group/control. All predicted bytes agree, weights/RNG remain pure, maximum logit difference is 9.53674e-6 and probability difference 2.28259e-6. Both original thresholds pass without revision. This is selected numerical inference, not a complete training rerun or resolution of the preceding study's long-revision numerical caveat.

The first one-GPU allocation ended at the declared runtime gate before any main case. The separately reviewed execution amendment retained every scientific input and used one explicit two-GPU replacement. Its own gate passed at 2,273.09 s; complete pipeline time was 819.76 s and main time 715.87 s. Jobs 13926582/13926674 are both cancelled after completed work, respectively 148 s × one GPU and 918 s × two GPUs: 0.55111 allocated GPU hours total. Accounting and the final queue confirm both closed and no active user jobs. No automatic renewal, time-limit extension, unrelated cancellation or helper-selection change occurred.

`models.json` indexes all four diagnostic terminal checkpoints, hashes, actual configs, codec and six gate results. They are not admitted shared sequence-retention baselines. The unchanged learned model has 839,248 registered parameters; the supplied static codec is a separate 2,978-byte JSON preprocessing artifact. Its identity mapping is not a growing memory or a learned replacement module. The future-only exclusion union contains 100,526 canonical histories, including all actually presented main/engineering/smoke histories; use it for subsequent studies without changing launch packets.

## Next diagnostic and team boundary

Replacing a composed key with one token was not a sufficient repair here. Before more language pretraining, larger models or longer retention streams, separate selecting the matching record from decoding its value. A deliberately supervised record-selection positive control can test whether the model learns requested-key matching when value decoding is removed; it must use a new frozen protocol and fresh histories, and be labeled as a diagnostic with extra alignment supervision. Preserve the current full-value task as the comparator. No such extra training was run in this session.

Member C should keep the qualified typed-memory reference available for A/B while debugging this complete sequence path. The intended sleep/Hebbian methods, atomic-compatible replay parsing, native32/cross-session memory and corrupted-observation handling remain separate requirements. No memory-retention, sleep timing, biological novelty, paper-faithful Titans or universal byte-tokenization conclusion follows from these results. All current histories are consulted development data; no terminal run or failure was dropped.

Full root: C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-key-binding-2026-11-15. Main results are `gpu-main/result`; original preflight-only output is `gpu-output/result`. `validation.json`, `models.json`, `analysis/independent_results_review.md`, `analysis/claim_evidence_map.csv` and the updated handoff bind claims to evidence. Three numerical PNG/PDF/SVG figures were inspected. The compact Git index is `results/key-binding-2026-11-15`; the adjacent `CLMM_Key_Binding_2026-11-15.zip` and verification file preserve raw inputs, weights/optimizers, both resource attempts, source and branch bundles and SHA/CRC checks. The pre-existing SLEEP_MULTIREVISION.md user edit is preserved; no Google Doc change, teammate message or GitHub push occurred.
'''
p.write_text(text,encoding='utf8')

p=REPO/'docs/HANDOFF.md';text=p.read_text(encoding='utf8')
_,marker,history=text.partition('**COMPLETE: sequence integration, acquisition FAILED, November 15.**')
assert marker
prefix='''# CLMM working handoff

## Resume here

**COMPLETE: fixed-size key binding diagnostic; acquisition FAILED, November 15.** Branch `sequence-key-binding`; executed source `5e8c61829ff85ecea3c9314d4ccd8dfef3268b68`. Read [KEY_BINDING.md](KEY_BINDING.md). Four matched terminal cases: original language parents 68101/68105 × digit/one-token atomic keys, independently adapted at attention128 with 839,248 registered parameters unchanged. Per case:2,000 updates,32,000 forks,2,908,928 input bytes; all four task-unadapted initial conditions retained. Common96train/32dev keys,256K1+1744K2, no replay or revisions. Terminal2000only; midpoint1000recovery-only. All cases/failures retained, no extra outcome-based training.

All24 acquisition cells fail the205/256 threshold; each both-correct count is0–2/256. All four joint-four train-seen and dev-heldout endpoints are0/256, yielding0pp atomic-minus-digits effects and degenerate bootstrap[0,0]intervals. Do not call those equivalence evidence. Single-fact training reaches100%, but last100K2 updates average49.81–50.06%accuracy. Freshtrain answers are512/512oneof2present values permodel, yet253–256/256pairs useidentical predictions forbothkeys. Atomic encoding was not a sufficient fix underthisbudget; no capacity impossibility, tokenizer-only cause or retention qualification follows.

109scoped tests pass; all3packets/worlds/arrays/labels/exclusions and50sourceGitblobs independently reconstruct. Main81920+engineering/smoke2640=84560savedvectors pass rawvalidity, withmax mainmetricerror1.01744e-6 andexact predictedbytes. Fourinitial/fourterminal/fourrecoverycheckpoints andeightmainoptimizerstates pass tensor/provenance checks. All8000mainupdates havefinite nonzero memory gradients (plumbing only). Coordinator independently agrees on primarybootstrap/budgets. ThreePNG/PDF/SVG figures visually reviewed.

Actual CPU proof PASSES:2560selected queries/all8mainmodels/allgroups-controls, exact original GPU batch membership, pureweights/RNG, zeroargmaxmismatches. Maxlogit9.53674e-6/maxprob2.28259e-6 under original0.002/0.0002 tolerances. This short-input proof does not repair the previous study's long-revision numerical caveat. `validation/cpu-proof-plan.json` binds prospective script/selection; `cpu-recheck/verification.json` owns observed result.

Runtime-only first attempt13926582 onins093 stopped beforemain (projection3940.09s>1800), CANCELLED after148s ×1GPU. Original `gpu-output/result`/`cluster` preserved. Explicit execution-only amendment before main kept all frozen data/source/steps/endpoints, replacing resources once with job13926674 onins086,2A6000/8CPU/16GiB/45min. Parallel gate2273.09<=2300s passed; hardpipeline2400s. Actual pipeline819.76s/main715.87s; secondjob CANCELLED after918s ×2GPU. Total0.55111allocatedGPUh; final accounting/queue confirm bothclosed/nojobs. No renewal/time-limit extension/helperselectionchange/unrelated cancellation. `cluster-closure.json` is authoritative. No inference remains.

Root: `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-key-binding-2026-11-15`; source/launch work: sibling `work/clmm-key-binding`. Main `gpu-main/result` succeeded. Effective protocol is immutable `design.json` plus separately hash-bound `execution-amendment.json`; originalpacket source50hashes and mainmetadata SHA are unchanged. Frozen mainpacket has15055unique training histories/99502exclusions and1024newcanonical evaluation histories/20480inputrows (10240perencoding). Future-only union100526 is `analysis/future_exclusion_signatures.json`; supplied85123priorunion is archived at `provenance/prior_exclusion_signatures.json`. Do not rewrite launch packets. `models.json` supplies four diagnostic weights/configs/source/codec/gate statuses. Static codecJSON2978bytes is preprocessing state separate from839248learned parameters.

Adjacent `CLMM_Key_Binding_2026-11-15.zip`/verification preserves full raw/checkpoints/optimizers, CPU proof, both attempts, source and full branch bundle with artifact SHA/CRC. Compact Git `results/key-binding-2026-11-15` omits heavy arrays/weights/signatures. Rerun from isolated LF executed5e8checkout, not laterdocHEAD; sourceclone `work/clmm-key-binding/source-frozen`. Initial Windows shared-clone MSYS permission failure occurred before model experiments; bundle checkout succeeded and is preserved by source provenance.

Next: a separate record-selection positive control to distinguish key matching from value decoding, with fresh excluded histories and explicit extra-alignment-supervision scope. Keep the full-value task as comparator, same model size initially, and no new training assumed complete. Keep the admitted typed-memory reference for A/B. Current atomic format is unsupported by the digit-only SequenceReplay parser; no adapter, native32, cross-session/long-delay, biological or memory-only-sleep efficacy is claimed. Segment reset is parity-only (95<128); retrievalzero changes outputs/activations while ordinary updates still execute. Codec/prior byte exposure bounds interpretation; latest-truth/noisy-fact authority remains a team choice. Pre-existing user SLEEP_MULTIREVISION.md edit stays byte-for-byte, outside our commits. No GoogleDoc edits/messages/push/PR/tag moves. Historical studies follow.

'''
p.write_text(prefix+marker+history,encoding='utf8')

p=REPO/'docs/MEMBER_C_STATUS.md';text=p.read_text(encoding='utf8')
marker='# Member C: shared training and evaluation\n\n';assert text.startswith(marker)
new='''## Latest: atomic-key control also fails binding

November 15: [KEY_BINDING.md](KEY_BINDING.md) records the completed fixed-size diagnostic on source5e8c618, branch `sequence-key-binding`. Four terminal models use two original language parents and digit/one-token key representations, all at wide128 attention with839,248registered parameters unchanged. Inputs, labels and budgets are paired:2,000updates/32,000forks/2,908,928inputbytes each. The newcommon smaller keypool and acquisition-only curriculum differ from the preceding study in both arms; only the within-current-study encoding contrast is controlled.

All24admissioncells fail, both-correct0–2/256; allfour jointoriginal+swap endpoints are0/256 on both primary groups. Atomic-minus-digits is0pp at an empirical floor; degenerate[0,0]bootstrap intervals are not equivalence evidence. Training reaches100%in the single-fact stage but stays near50%on two-fact batches; fresh-context predictions always choose one observedvalue yet ignore the requestedkey in253–256/256worlds. One-token keys alone do not repair thisrecipe. Larger-model necessity and a specific architecture/optimization cause remain unproven.

109scoped tests, full81,920main+2,640engineering/smoke scores, checkpoint/optimizer/source/data audits, coordinator rawstatistics and three figures are saved. Actual CPU inference checks all2,560selected answers agree and fullscores meet unchangedtolerances (maxlogit9.54e-6/maxprob2.28e-6). Firstsingle-GPU runtimegate stopped beforemain; the documented resource-only two-GPU replacement completed allconditions. Bothjobs areclosed,0.55111allocatedGPUh total. No failure or terminal condition was dropped.

C now supplies a diagnostic modelregistry/codec/evaluation and clear acquisition failure, not an admitted sequence-retention baseline. Next separate selecting the matching record from decoding its value under controlled extra supervision, retaining the full-value task as comparator. Keep existing qualified typed-memory controls for A/B. The full proposal's intended biological/sleep modules, native-memory persistence, delayed recall and corruption tests remain open. Current atomic records do not pass the digit-only replay parser. Future experiments should use the100,526-history exclusion union; these histories are consulted development data.

'''
assert '## Latest: atomic-key control also fails binding' not in text
p.write_text(marker+new+text[len(marker):],encoding='utf8')

p=REPO/'README.md';text=p.read_text(encoding='utf8');marker='## Status\n\n';assert marker in text
new='''The [key-binding encoding diagnostic](docs/KEY_BINDING.md) tests digits versus one-token keys with four matched adaptations of the unchanged839,248-parameter model. Both formats fail:0–2/256both-correct per acquisition cell and0/256joint original+swap success on both primary groups. Atomic keys are not a sufficient repair under thisrecipe. All109scoped tests, saved-score audits and selected2,560-query CPU proof pass. Member C's next diagnostic separates record selection from value decoding before scaling; qualified typed-memory controls remain available for A/B.

'''
assert 'The [key-binding encoding diagnostic]' not in text
p.write_text(text.replace(marker,marker+new,1),encoding='utf8')

# Preserve only this session's new paragraphs when adding spaces to numeric prose.
for name,boundary in [('docs/HANDOFF.md','**COMPLETE: sequence integration, acquisition FAILED, November 15.**'),
                      ('docs/MEMBER_C_STATUS.md','## Latest: complete-sequence integration, failed acquisition')]:
    p=REPO/name;t=p.read_text(encoding='utf8');before,sep,after=t.partition(boundary);assert sep
    replacements={'839248learned':'839,248 registered','839,248registered':'839,248 registered','all24':'all 24','All24':'All 24','allfour':'all four',
        '2,000updates':'2,000 updates','32,000forks':'32,000 forks','2,908,928inputbytes':'2,908,928 input bytes','Same':'Same',
        '0–2/256both':'0–2/256 both','0pp':'0 pp','0/256joint':'0/256 joint','wide128':'wide 128','near50%':'near 50%',
        '0.55111allocated':'0.55111 allocated','0.55111 allocatedGPUh':'0.55111 allocated GPU hours','step2000':'step 2000'}
    for a,b in replacements.items():before=before.replace(a,b)
    p.write_text(before+sep+after,encoding='utf8')
