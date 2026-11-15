"""Write a concise evidence-scoped interpretation from saved final tables."""
from collections import Counter
import json
from pathlib import Path

from independent_tools import read_csv

ROOT=Path(__file__).resolve().parent


def pct(value):
    return f'{float(value)*100:.2f}'


def effect(row):
    return f"{float(row['estimate'])*100:+.2f} pp [{float(row['lower'])*100:+.2f}, {float(row['upper'])*100:+.2f}]"


def main():
    summary=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
    contrasts,gates=read_csv(ROOT/'contrasts.csv'),read_csv(ROOT/'gates.csv')
    telemetry=json.loads((ROOT/'replay_telemetry.json').read_text(encoding='utf-8'))
    supplement=json.loads((ROOT/'supplemental_exclusion_audit.json').read_text(encoding='utf-8'))
    runtime=json.loads((ROOT/'runtime_gate_audit.json').read_text(encoding='utf-8'))
    pick=lambda category,tail,family='balanced',schedule='mean':next(r for r in contrasts if r['category']==category and int(r['retention_tail'])==tail and r['route_family']==family and r['phase']=='mean' and r['schedule']==schedule and r['contrast_kind']=='freshness')
    lines=['# Independent bounded-freshness interpretation','',
           'All 12 fixed C9 step-1200 local-delta checkpoints are included, with no added optimization or model/map selection. The independent reviewer regenerated the data and reconstructed saved scores, event plans, refresh/fallback fields, query diagnostics and resource arithmetic without importing the model/evaluator or running inference. The source commit is `9ab3901860548d2d87a9afc61e8ce58f950a03ce`.','',
           '## Co-primary result','',
           '| D128 category | Bounded accuracy | Literal accuracy | Bounded minus literal (97.5% CI) |','|---|---:|---:|---:|']
    for r in summary['primary']:
        lines.append(f"| {r['category'].capitalize()} | {pct(r['candidate_accuracy'])}% | {pct(r['baseline_accuracy'])}% | {effect(r)} |")
    lines += ['', 'Revised accuracy improves relative to literal replay in this fixed comparison. The stable endpoint interval covers zero; it is not an equivalence test.',
              '', 'Each endpoint equally averages both schedules, both phases and the two fixed parents, with 0.5 weight on content routing and 0.5 on the mean of all five fixed random maps. Intervals use 5,000 paired whole-history PCG64 bootstrap draws (seed 91204), jointly retaining every condition; the two 97.5% intervals implement the declared Bonferroni adjustment. The 128 histories are the resampling units. These intervals condition on the frozen checkpoints and maps; they do not estimate training-seed or routing-map population uncertainty.','',
              '## Admission and repair','']
    common={(r['parent_seed'],r['route']):r['common_checkpoint_admission']=='True' for r in gates}
    failures=[r for r in gates if r['own_full_retention_gate']!='True']
    lines.append(f"Common short/neutral admission passes for {sum(common.values())}/12 checkpoints. The own-D0 six-target threshold fails in {len(failures)}/{len(gates)} arm/phase/checkpoint cells; all remain in the analysis. Overall classification: **{summary['interpretation']}**.")
    lines += ['', '| Arm | Own-D0 failures / 24 | Minimum D0 six-target all-correct |','|---|---:|---:|']
    for arm in ('online','repeat','continuous','periodic','continuous_bounded','periodic_bounded'):
        rows=[r for r in gates if r['arm']==arm]
        lines.append(f"| {arm} | {sum(r['own_full_retention_gate']!='True' for r in rows)}/24 | {pct(min(float(r['own_D0_six_target_all_correct']) for r in rows))}% |")
    lines += ['', f"D0 revised repair (secondary, nominal 95% CI): {effect(pick('revised',0))}. D0 concerns repair under replay/arrival order, so it must be shown alongside later retention rather than treated as a clean retention effect.",'',
              '## Secondary detail','', '| Family | D128 stable freshness | D128 revised freshness |','|---|---:|---:|']
    for family in ('content','mean5random'):
        lines.append(f"| {family} | {effect(pick('stable',128,family))} | {effect(pick('revised',128,family))} |")
    lines += ['', f"D128 early-stable freshness: {effect(pick('early',128))}; initially eligible stable: {effect(pick('eligible',128))}. Early keys receive zero direct replay writes under all four replay policies, so early-group changes concern interference preservation rather than rehearsal.",'',
              '| Category | Continuous freshness, D128 | Periodic freshness, D128 | Timing × freshness, D128 |','|---|---:|---:|---:|']
    for category in ('stable','revised'):
        interaction=next(r for r in contrasts if r['route_family']=='balanced' and r['phase']=='mean' and int(r['retention_tail'])==128 and r['category']==category and r['contrast_kind']=='timing_x_freshness')
        lines.append(f"| {category} | {effect(pick(category,128,schedule='continuous'))} | {effect(pick(category,128,schedule='periodic'))} | {effect(interaction)} |")
    lines += ['', '| Bounded arm minus anchor, D128 | Stable | Revised |','|---|---:|---:|']
    for arm in ('continuous_bounded','periodic_bounded'):
        for baseline in ('online','repeat'):
            rows={category:next(r for r in contrasts if r['route_family']=='balanced' and r['phase']=='mean' and r['retention_tail']=='128' and r['category']==category and r['candidate']==arm and r['baseline']==baseline and r['contrast_kind']=='anchor') for category in ('stable','revised')}
            lines.append(f"| {arm} minus {baseline} | {effect(rows['stable'])} | {effect(rows['revised'])} |")
    lines += ['', 'Both bounded arms have lower stable accuracy than the online and repeat-current anchors. Revised differences versus those anchors have intervals covering zero. The primary benefit repairs an avoidable weakness of literal replay; it does not establish a replay advantage over these simpler anchors.']
    lines += ['', 'All secondary intervals are descriptive 95% intervals. Per-checkpoint/map/phase/delay outcomes, repeat-current and lower-write online anchors, obsolete-any-version errors and query exposure diagnostics remain in the CSV tables. They were not used to select a checkpoint, routing map or primary contrast.','',
              '## What is matched and what changed','',
              f"All {telemetry['candidate_sequences_matched']} pressure history/cell candidate sequences match across the four replay policies, and literal continuous/periodic have identical applied extra source/key/value order. Bounded continuous/periodic differ in {telemetry['bounded_CP_actual_source_differences']:,} applied sources and {telemetry['bounded_CP_actual_value_differences']:,} values because their FIFO lookups happen at different causal times. Calling bounded timing conditions identical-content would be incorrect.",
              '', 'Bounded replay has zero stale-at-emission extra writes in this capacity-eight/block-four grid. Fallback is applied and counted rather than dropping writes. Zero staleness here does not establish a guarantee under arbitrary larger blocks or eviction patterns. An extra write can be obsolete relative to a final target while still current when emitted; those observer labels are kept distinct. Source-only refresh changes provenance but gives the model the same key/value write; changed values are the effective freshness intervention.',
              '', 'The model, real arrivals, candidate-key order, write counts, adaptive trace bytes and retained record capacity match within each comparison. Applied values intentionally change. Bounded configuration adds fixed schema/freshness metadata, so serialized controller byte counts need not be identical. These matches do not imply identical FLOPs or total evaluator process memory.',
              '', '## Provenance and resource qualification','',
              f"The launch union contained {supplement['launch_exclusion_count']:,} signatures from both exact training arrays and the prior `episodes.json` scan. A broader postlaunch structural search found one prior seed-90323 integration history embedded in two `verification.json` files. Its signature was absent from that union. The corrected coverage is {supplement['broader_prior_numeric_files']} numeric-history JSON files: 273 numeric episode files plus two verification files; one additional episode JSON contains semantic byte-task data and no typed numeric histories.",
              '', f"The original launch packets and outcomes are unchanged. Adding the one missing signature to the main launch union gives {supplement['extended_exclusion_count']:,} signatures and zero overlap with all 1,280 main histories. Separately, the engineering launch union plus that signature contains {supplement['engineering_own_extended_exclusion_count']:,} signatures and has zero overlap with its 160 histories. The main exclusion union intentionally contains the engineering histories. Independent generation under each expanded union reproduces every accepted stream exactly. This is a documented launch-provenance coverage omission, with no observed data leakage or outcome change. Exact source paths/hashes, supplementary signature and reusable launch-union superset are saved in `supplemental_exclusion_audit.json`.",
              '', f"The runtime-only preflight padded projection was {runtime['padded_projection_seconds']:.2f} seconds, below the prospective 3,600-second continuation threshold. Engineering efficacy was not inspected. The main pipeline used two CPU workers with one Torch thread each and a 4,200-second hard limit. All {summary['audited_totals']['score_vectors']:,} saved score vectors, {summary['audited_totals']['causal_writes']:,} chronological writes and {summary['audited_totals']['resource_batches']:,} resource batches passed arithmetic checks; this reviewer performed no model inference.",
              '', '## Claim boundary','',
              'Correctness here is defined by the latest real observation for a key. Refresh is a causal recency policy under that authority assumption, not a learned test of whether an observation is trustworthy. No noisy, adversarial, delayed-authority or source-conflict observations were evaluated. The split excludes exact whole histories; the 64-key/16-value typed vocabulary is intentionally shared. The main split was fresh when declared and is now consulted evidence for this fixed study; it should not be reused as untouched selection-free evidence for a later intervention.',
              '', 'The component has 81,921 slow parameter elements and a 65,536-byte fast trace per stream. That is an inspectable capacity for this controlled finite KV diagnostic, not evidence for full Titans/MAC/MAG/MAL, language-scale transfer or biological novelty. The primary Titans v1 section 5.1 reports 170M–760M parameter models trained on 15B–30B tokens: https://arxiv.org/html/2501.00663v1#S5.SS1. Existing complete sequence-model acquisition and held-out assignment/template gates should be addressed before making a full-project retention claim or choosing a larger model size from these component outcomes.','']
    (ROOT/'result_interpretation.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(interpretation='saved',path=str(ROOT/'result_interpretation.md'))))


if __name__=='__main__':
    main()
