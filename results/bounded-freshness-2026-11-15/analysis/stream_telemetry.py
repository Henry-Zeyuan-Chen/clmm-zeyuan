"""Model-free applied-replay telemetry from independently reconstructed plans."""
import argparse
import json
from pathlib import Path

import numpy as np

from independent_tools import ARMS, PRESSURE_CELLS, independent_plan, query_diagnostics, sha, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    args = p.parse_args()
    root, out = args.root.resolve(), Path(__file__).resolve().parent
    data = json.loads((root/'packet/episodes.json').read_text(encoding='utf-8'))
    rows, summaries, differing = [], [], []
    for cell in PRESSURE_CELLS:
        for i, ep in enumerate(data[cell]):
            plans = {arm: independent_plan(ep, arm) for arm in ARMS}
            extra = lambda a: [e for e in plans[a]['events'] if e['kind'] == 'extra']
            literal_cp = [(e['source'],e['key'],e['value']) for e in extra('continuous')]
            assert literal_cp == [(e['source'],e['key'],e['value']) for e in extra('periodic')]
            candidate = lambda a: [(e.get('sampled_source',e['source']),e['key'],e.get('sampled_value',e['value'])) for e in extra(a)]
            assert all(candidate(a) == literal_cp for a in ARMS[2:])
            c, p_extra = extra('continuous_bounded'), extra('periodic_bounded')
            differing.append(dict(cell=cell, episode=i,
                                  actual_source_differences=sum(x['source'] != y['source'] for x,y in zip(c,p_extra)),
                                  actual_value_differences=sum(x['value'] != y['value'] for x,y in zip(c,p_extra))))
            for arm, plan in plans.items():
                extras = extra(arm)
                features = [query_diagnostics(ep, plan, q) for q in ep['queries'][:6]]
                early_extras = sum(f['target_extra_writes'] for q,f in zip(ep['queries'],features) if q['group']=='early')
                if arm in ARMS[2:]:
                    assert early_extras == 0
                rows.append(dict(cell=cell, phase=ep['phase'], retention_tail=ep['retention_tail'], episode=i, arm=arm,
                                 real_observations=len(ep['keys']), actual_writes=plan['writes'], extra_writes=len(extras),
                                 stale_extra_writes=plan['stale_extra_writes'],
                                 refreshed_records=sum(bool(e.get('refreshed',False)) for e in extras),
                                 changed_values=sum(e['value'] != e.get('sampled_value',e['value']) for e in extras),
                                 fallback_writes=sum(bool(e.get('fallback',False)) for e in extras),
                                 obsolete_final_query_extras=sum(f['query_final_obsolete_extra_writes'] for f in features),
                                 early_stable_extra_writes=early_extras))
    for cell in PRESSURE_CELLS:
        for arm in ARMS:
            subset = [r for r in rows if r['cell']==cell and r['arm']==arm]
            summaries.append(dict(cell=cell, phase=subset[0]['phase'], retention_tail=subset[0]['retention_tail'], arm=arm,
                                  histories=len(subset), **{key: float(np.mean([r[key] for r in subset]))
                                   for key in subset[0] if key not in ('cell','phase','retention_tail','episode','arm')}))
    write_csv(out/'plan_telemetry_per_history.csv', rows)
    write_csv(out/'replay_telemetry.csv', summaries)
    write_csv(out/'bounded_schedule_content_differences.csv', differing)
    result = dict(status='passed', scope='independent NumPy schedules only; one world set shared across checkpoints',
                  episodes_sha256=sha(root/'packet/episodes.json'), histories_per_cell=128,
                  rows=len(rows), candidate_sequences_matched=6*128,
                  literal_CP_actual_sequences_match=True,
                  bounded_CP_actual_source_differences=sum(r['actual_source_differences'] for r in differing),
                  bounded_CP_actual_value_differences=sum(r['actual_value_differences'] for r in differing),
                  early_stable_direct_replay_writes=sum(r['early_stable_extra_writes'] for r in rows if r['arm'] in ARMS[2:]),
                  stale_at_emission_distinct_from_final_obsolete=True,
                  tables=['plan_telemetry_per_history.csv','replay_telemetry.csv','bounded_schedule_content_differences.csv'])
    (out/'replay_telemetry.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
