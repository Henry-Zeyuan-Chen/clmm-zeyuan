"""CPU-only independent arithmetic audit of earlier saved revision predictions."""
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
metadata = json.loads((a.root / 'main/metadata.json').read_text())
cases = []
for case in metadata['cases']:
    path = a.root / 'main/runs' / case['name'] / 'evaluation/predictions.csv'
    with path.open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['cell'] == 'revise2' and r['arm'] == 'online']
    revised = [r for r in rows if r['category'] == 'revised']
    stable = [r for r in rows if r['category'] == 'stable']
    assert len(revised) == len(stable) == 128
    for r in rows:
        assert int(r['correct']) == (int(r['target']) == int(r['prediction']))
        assert int(r['obsolete']) == (int(r['old']) >= 0 and int(r['old']) == int(r['prediction']))
    failed = [r for r in revised if not int(r['correct'])]
    cases.append(dict(name=case['name'], rule=case['rule'], parent_seed=case['parent_seed'],
                      policy=case['policy'], random_seed=case['random_seed'], decay=case['inactive_decay'],
                      histories=len(revised), revised_failures=len(failed),
                      obsolete_failures=sum(int(r['obsolete']) for r in failed),
                      stable_failures=sum(not int(r['correct']) for r in stable),
                      failure_episode_ids=[int(r['episode']) for r in failed]))
summary = dict(input_root=str(a.root), cases=cases,
               all_revision_errors_obsolete=all(c['revised_failures'] == c['obsolete_failures'] for c in cases),
               all_stable_queries_correct=all(c['stable_failures'] == 0 for c in cases),
               total_revised_failures=sum(c['revised_failures'] for c in cases),
               scope='Raw prediction arithmetic only; no model inference, training, or performance probing.')
a.out.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k != 'cases'}, indent=2))
