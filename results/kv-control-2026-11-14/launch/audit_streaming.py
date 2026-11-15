"""Check the post-run live policy against every preserved evaluation plan."""
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
ROOT = Path(r'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-kv-control-2026-11-14')
sys.path.insert(0, str(REPO))
from clmm.pretrain.kv_scheduler import StreamingReplay

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    records = []
    for run in sorted(ROOT.iterdir()):
        if not (run / 'metadata.json').exists():
            continue
        for stage in ('evaluation', 'initial-evaluation'):
            folder = run / stage
            if not folder.is_dir():
                continue
            episodes = json.loads((folder / 'episodes.json').read_text())
            plans = json.loads((folder / 'plans.json').read_text())
            writes = 0
            resumes = 0
            for entry in plans:
                ep = episodes[entry['cell']][entry['episode']]
                arm = entry['arm'] if entry['arm'] in ('continuous','periodic','repeat') else 'online'
                policy = StreamingReplay(arm, ep['seed'])
                emitted = []
                for index, (key, value) in enumerate(zip(ep['keys'], ep['values'])):
                    if index < ep['boundary']:
                        emitted.append(policy.prime(key, value))
                    else:
                        emitted.extend(policy.step(key, value))
                        # Includes partial blocks and an already-active snapshot.
                        if index == ep['boundary']:
                            payload = json.loads(json.dumps(policy.export_state()))
                            policy = StreamingReplay.from_state(payload)
                            resumes += 1
                    counts = policy.logical_counts()
                    assert counts['buffer_pairs'] <= (8 if arm in ('continuous','periodic') else 0)
                    assert counts['snapshot_pairs'] <= (4 if arm in ('continuous','periodic') else 0)
                emitted.extend(policy.finish())
                assert emitted == entry['plan']['events'], (run.name, stage, entry['cell'], entry['arm'], entry['episode'])
                writes += len(emitted)
            records.append(dict(run=run.name, stage=stage, plans=len(plans), writes=writes,
                                resumed_policies=resumes, episodes_sha256=sha(folder/'episodes.json'),
                                plans_sha256=sha(folder/'plans.json')))
    assert records
    result = dict(status='passed', scope='Exact event equality, all saved main, smoke and supplemental plans; post-run implementation, no model reruns',
                  frozen_evaluator_commit='ff62df0f85ff7cbf62f3209755ea066959aa7085',
                  scheduler_sha256=sha(REPO/'clmm/pretrain/kv_scheduler.py'),
                  auditor_sha256=sha(Path(__file__)), plans=sum(r['plans'] for r in records),
                  writes=sum(r['writes'] for r in records), resumed_policies=sum(r['resumed_policies'] for r in records),
                  runs=records)
    (ROOT/'analysis/streaming_parity_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))

if __name__ == '__main__':
    main()
