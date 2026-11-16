# Constant LR versus 500-step warm-up — completed November 16

Warm-up improves the mean on familiar wording but reduces the mean on new
wording. It does not repair the weak atomic initializer or qualify a shared
sequence baseline. Keep constant LR as the reference recipe rather than
adopting warm-up as a general fix. All eight terminal weights remain available
as diagnostics, with their failed gates retained.

## Results

The table reports four-answer joint recall: both requested keys must be
correct in the original and order-swapped views of a semantic world. Each
group has 256 fresh worlds; means give equal weight to four fixed initializer
pairs. Differences are warm-up minus constant.

| Endpoint (fresh assignments) | Constant | Warm-up | Difference, pp (98.3333% CI) |
| --- | ---: | ---: | ---: |
| Training keys, familiar wording | 68.26% | 72.07% | +3.81 [2.05, 5.76] |
| Development keys, familiar wording | 66.31% | 71.58% | +5.27 [2.93, 7.42] |
| Development keys, new facts and queries | 24.51% | 17.87% | -6.64 [-9.86, -3.52] |

The broad fitting-and-transfer criterion fails because the new-wording
co-primary has a negative adjusted interval. Effects vary sharply by pair;
this is neither uniform improvement nor uniform harm:

| Initializer | Familiar C / W | New wording C / W | Gates passed C / W |
| --- | ---: | ---: | ---: |
| 68101-digits | 204 / 254 | 73 / 115 | 31 / 48 |
| 68101-atomic | 0 / 1 | 0 / 0 | 0 / 0 |
| 68105-digits | 255 / 247 | 134 / 13 | 74 / 40 |
| 68105-atomic | 220 / 231 | 44 / 55 | 40 / 40 |

Recall entries above are joint-correct worlds out of 256 on development keys;
gate counts are out of 90 per model. C/W denotes constant/warm-up. The
68105-digit pair accounts for the large new-wording mean decline (134 to 13),
while 68101-digits improves (73 to 115). The 68101-atomic pair stays near
floor; its last-100-update mean NLL is 0.7182/0.7172. These observations do
not identify the internal cause. All four initial S-as-V controls have zero
joint recall. They are pretrained selectors, not random networks.

**Admission is 0/8.** All 90 gates require at least 205/256: both answers in
each of four order/value views, plus joint4, across nine fact/query cells and
both key groups. There are 273 passes and 447 failures among 720 requirements.
No weak initializer, cell or format is waived or selected away.

## Controlled comparison and provenance

Both arms restart independently from each of the same four exact 20k S
checkpoints (68101/68105 × digit/atomic), with fresh empty AdamW states.
They do not continue the previous study's terminal V models. Every main run
has 839,248 registered parameters, attention128, raw 95-byte queries, full256
output, and exactly the same mixed arrays, labels, ordering and model RNG
within its pair. Each receives 6,000 updates, 96,000 answers and 9,120,000
input bytes: 48,000 main updates, 768,000 answers and 72.96M bytes overall.
The two 64-update engineering runs are additional, separate integrity checks.

Constant LR is 3e-4. Warm-up is 3e-4×min(update/500,1), indexed from one:
6e-7 first, reaching 3e-4 on update500 and holding it through6000. No LR-area
compensation is used. Nominal sums are1.8 versus1.72515, and AdamW decay also
depends on LR. These are complete schedule recipes; early gradient shock,
backbone drift and model-capacity explanations are not established.

Training uses the four cells F0Q0/F0Q1/F1Q0/F1Q1, exactly two contexts per
cell per update. Semantic, view and cell RNGs are independent. The new stream
differs from the preceding familiar/mixed study; only this study's paired LR
contrast is controlled. The complete four-view history closures reject all
prior985,652 signatures and each other. Individual facts may recur.

New family3 uses facts `item {key}: its code is {value}.` and
`{key} is paired with code {value}.`, and queries `fetch code tied to key
{key}: ` and `{key}: what code is stored? `. Exact newlines, padding and
atomic aliases are frozen in the packet. Consulted family2 is omitted.
Family3 never enters training or engineering neural execution. All eight
terminal hashes lock at22:33:46.302323UTC before confirmation starts.
Twelve capabilities (eight finals plus four initial S-as-V controls) produce
466,944 confirmation full256 distributions across nine families0/1/3 cells,
four views, both key groups and original-F0Q0 erasure controls.

The three primaries average paired differences across four fixed cases per
world, then bootstrap worlds5,000 times. Train PCG92205 and dev PCG92204
matrices are independent and shared within group. Exact linear quantiles
.05/6 and1−.05/6 give adjusted98.3333% intervals; positive support requires
a positive lower bound. Other cells use descriptive95% intervals. This
conditions on fixed checkpoints, codec, stream and wording, and does not
estimate training-procedure variance or generalization over arbitrary templates.

Executed neural/data source is `ae13353fab1aa117ca24870aabb0169e350f2c57`,
branch `sequence-lr-warmup`. All52 original model-source hashes stay intact.
Packet metadata SHA is
`79657e9d53b158c59c380b979bd3952ddff8b8e77abc230f7ab8e41b774c4d75`.
The prospective protocol is preserved in that source commit and in the packet.
Later analysis/documentation commits do not change the executed source.

## Verification and resources

129 scoped tests pass; final13 LR/resource tests also pass on the frozen
checkout. Independent audit reconstructs all194,048 training-array rows,
78,976 evaluation-packet rows, RNG/rejection streams and the full1,181,812
history union. Full saved-score math checks466,944 confirmation,1,152
engineering and160 first-minibatch distributions, all LR values/budgets,
input collisions and terminal locks. Paired first-logit differences are zero
for all four main pairs and engineering. Separate analysis implementations
agree exactly on18 contrasts,216 case cells,48 controls and720 gate decisions.

All eight actual model/optimizer audits pass:839,248 parameters, finite
states,36 changed memory tensors,72 AdamW states at6000steps and exact
one-query GPU checkpoint reloads. The Windows CPU check passes14,592 selected
actual queries/912batches across all12 capabilities: exact argmax, max logit
delta0.0002899170 and full-probability delta0.0000311042, below the frozen
.002/.0002 bounds. This is selected-batch portability, not all-query or
training reproduction. Additional selected GPU batch checks were not run
before allocation closure. Five numerical figures are visually inspected.

Insomnia job13935872 is COMPLETED, exit0:0,6652seconds on one RTX A6000:
1.847778 allocated GPU-hours. It used2CPUs/8G and ended22:57:06UTC; no study
job or observer remains active. The actual three-hour allocation and engineering
runtime gate bounded the whole design; no renewal or case reduction occurred.
Unrelated NLPjob13933883 was not shared or altered. The remote907,929,600-byte
tar and all230 extracted file hashes pass. Final local recovery verification
is recorded separately inRECOVERY.json.

One initial tensor-audit helper wrongly assumed500 warm-up steps for the
constant arm. Its corrected0/500 assertion passes; no training/model source
or numeric tolerance changed. Raw audit and CPU bounds were inherited and
declared before launch. Their policy/selection files were saved during
engineering/main, before confirmation; file creation is not claimed to predate
all neural execution. Original failure and timing disclosure are retained.

## Member C handoff and next step

The registry retains every final model/optimizer, source/state/initializer
hash and failed gate with archive-relative recovery paths. This remains
visible-record acquisition inside attention128. None is qualified for
native32, stateful retention or the intended sleep/Hebbian sequence adapters.
A/B can continue with the qualified typed-memory reference components.

Next proposed diagnostic: compare a500-update value-output-head alignment
stage followed by5500 all-weight updates against6000 all-weight updates,
using constant LR, the same four S initializers, fixed registered size and
matched V examples. This directly tests whether separating output alignment
from representation adaptation helps; it does not assume that mechanism is
already proven. Freeze fresh histories/wording and all choices before running.
This next experiment is not implemented or launched in this study. Qualify
ordinary acquisition before separately training native32 and assessing recall
after attention/buffer exit.

Team choice remains a canonical digit interface versus broader formats as
the primary project scope. Agree before any new qualification policy; retain
these failures under the current policy. This proposal does not require a
decision to finish the current comparison. Current family3 and histories are
now consulted development material; future exclusions total1,181,812.

[Figures and source tables](../results/lr_warmup_2026-11-16) include all cases,
learning curves and actual logged LRs. Complete local artifacts, SESSION,
log, registry and recovery manifest are at
`C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-lr-warmup-2026-11-16`.
Remote evidence is
`/insomnia001/depts/edu/users/zc2819/clmm/lr-warmup-2026-11-16`.
Nothing was pushed; no interim report, teammate message or Google Doc edit
was made. The pre-existing SLEEP_MULTIREVISION edit and baseline tag are preserved.
