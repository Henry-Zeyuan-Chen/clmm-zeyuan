# Matched training: scoped independent interpretation

The predeclared joint primary criterion is **not met**. Both EMA and local delta achieve 100% observed short revised-query accuracy, initially and finally, so neither route family demonstrates revision superiority. Local delta nevertheless has much higher late stable accuracy, passes stable noninferiority, and qualifies under all original acquisition/revision gates in every fixed parent/map condition. These findings should remain separate: baseline qualification and strong stable performance do not turn the unmet four-constraint primary into success.

## Complete matched comparison and audits

All 24 conditions completed 1200 optimizer steps: two original untrained initializers 90511/90515, content plus all five fixed random maps 90701–90705, EMA and local delta. Every pair starts with the identical nine original parent tensors and zero H. The saved step 0 initial checkpoints were verified directly. All use the same training seed 90802 arrays, batch 32, AdamW learning rate .001 and weight decay 0, norm clip 1, terminal-step checkpoint policy and confirmation seed 90923 episodes. Each run has 279,488 training observations and 179,712 target answers; all 24 total 6,707,712 observations, 4,313,088 answers and 28,800 optimizer steps. Actual saved optimizer state verifies 1200 steps for all six learned tensors per run. No configurations, maps or checkpoints were selected by outcomes.

Here, untrained means the adapter's outer parameters are at training step 0; the shared literal codebook and structured initialization remain supplied. The 90923 initial/final evaluations are fresh development compositions under a fixed reporting recipe, not an untouched final paper test.

`raw_phase_audit.json` independently verifies 319,488 initial+final scorevectors, 1,665,024 causal write events and 5760 resource/serialization batches; exact saved prediction/obsolete/MSE/summary arithmetic; reset/rolled source histories; causal online plans; fixed module assignments; decay/projection exposure; 65,536 bytes persistent state per stream; exact exclusion/training arrays and intended episodes; and captured source bytes against recoverable commit `c9f5c153f786f1b82ac383d01c1d18c6c595946b`. There is zero whole-history overlap with training, prior 90723 or frozen 90823. `saved_weights_audit.json` checks 48 actual initial/final payloads and all 24 optimizer states. These are saved-artifact and arithmetic checks, not a new training or model inference rerun.

## Primary final contrasts

Differences below are local-delta minus EMA, in percentage points. The 5000-draw history-cluster bootstrap uses PCG64 seed 90804, jointly resampling each endpoint's 128 history IDs across methods, fixed initializers and maps. Random means retain all five maps. The 97.5% two-sided intervals supply lower .0125 bounds for the four one-sided Bonferroni constraints.

| Allocation family | Endpoint | Local delta | EMA | Difference |97.5% interval | Criterion |
|---|---|---:|---:|---:|---|---|
| Content | Short revised query |100.00%|100.00%|0.00|[0.00,0.00]|Superiority fails; lower is not>0|
| Mean5random | Short revised query |100.00%|100.00%|0.00|[0.00,0.00]|Superiority fails; lower is not>0|
| Content | K8D64 stable query |95.51%|24.02%|+71.48|[+65.23,+77.73]|Noninferiority passes; lower>-3|
| Mean5random | K8D64 stable query |97.07%|28.52%|+68.55|[+65.66,+71.41]|Noninferiority passes; lower>-3|

The zero-width empirical revision intervals reflect zero observed disagreements at this task's ceiling. They do not establish population equality or prove that either rule never fails. The available short task cannot demonstrate a unique repair advantage in this trained comparison.

Both methods pass all 12 per-rule short/K8D0 admissions; both also pass K16D0 secondary admission. Final local-delta D0 four-original-target all-correct is 100% at both K8 and K16 in every condition. EMA minimum is 99.22% at K8 and 96.09% at K16. Every candidate's K8D64 stable delta versus its exact parent/map comparator is positive, ranging 66.02 to 72.66 points. All 24 initial conditions also pass the short/K8/K16 gates; no initial gate failure was hidden. Full initial/final gate rows are preserved. This does not change the earlier frozen additive failures or single frozen EMA revision failure, which remain in the frozen analysis.

## Secondary paired initial-to-final evidence

Initial/final evaluation on the same 90923 histories was specified in the design. The following within-rule analyses are secondary, with nominal 95% paired-history intervals; they do not replace the primary hypotheses. `initial_to_final_audit.json` verifies identical query identifiers, keys, targets and old labels between the two stages, checking 9216 selected endpoint rows per stage.

| Rule | Allocation | Initial K8D64 stable | Final stable | Final minus initial, points | Nominal95% interval |
|---|---|---:|---:|---:|---|
| EMA |Content|22.66%|24.02%|+1.37|[-3.52,+6.05]|
| EMA |Mean5random|30.31%|28.52%|-1.80|[-4.06,+0.39]|
| Local delta |Content|88.87%|95.51%|+6.64|[+3.71,+9.96]|
| Local delta |Mean5random|91.64%|97.07%|+5.43|[+3.87,+6.99]|

All short revised-query initial-to-final changes are exactly 0 observed points, with nominal empirical intervals [0,0], because every rule/map/initializer starts and ends at 100%. Thus training is not what first enabled short revision in this comparison. The local rule already gives strong late stable recall before optimization, with additional positive late gains after the matched training recipe in both allocation families. Every local-delta parent/map's stable change is positive, ranging 3.52 to 8.20 points. EMA's paired intervals provide no clear late stable benefit from this training recipe; the random mean decreases slightly, with an interval crossing zero. These observations support evaluating initialized as well as trained components when judging Member C's pretraining contribution.

## What is supported

The frozen factorial diagnoses a real failure of the previously dense-trained EMA parents: erasure fixes old-value revisions, and removing active bulk decay supplies most late stable gain. The fair trained comparison supports strong stable behavior and shared-baseline qualification under equal persistent state, observation/label, optimizer-step and routing budgets. It does not demonstrate a distinct short-revision advantage over the equally trained comparator, because the short task was already at ceiling for both rules before optimization. It also does not show that pretraining is necessary: initialized local delta is already strong, while its measured terminal improvement is a narrower secondary finding.

All inference is conditional on two fixed initializers, five fixed random maps and the familiar correlated codebook. History bootstrap intervals do not quantify population uncertainty across training seeds/maps. Equal logical budgets do not mean equal FLOPs: erasure adds contraction/projection arithmetic, and both implementations compute all four module candidates before masking. New codebooks, repeated conflicting revisions, full Titans, teammate-final A/B methods, and replay timing remain outside this evidence. MemberC can retain both qualified methods and the initialized checkpoints for future comparisons; no new task or training was launched by this analysis.
