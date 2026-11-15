"""Independent saved-output audit and analysis primitives; no evaluator imports."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def episode_signature(ep):
    return hashlib.sha256(bytes(ep['keys'])+b'|'+bytes(ep['values'])).hexdigest()


def signatures_from_arrays(path):
    with np.load(path) as data:
        assert {'keys','values','lengths'} <= set(data.files)
        keys, values, lengths = data['keys'], data['values'], data['lengths']
        assert keys.shape == values.shape and keys.ndim == 3
        result = set()
        for step, length in enumerate(lengths):
            for b in range(keys.shape[1]):
                result.add(episode_signature(dict(keys=keys[step,b,:length].tolist(),
                                                 values=values[step,b,:length].tolist())))
        return result


def audit_eval(folder, episodes, table, rule, batch_size=16):
    folder = Path(folder)
    table = np.asarray(table, dtype=np.int64)
    assert table.shape == (64,) and table.min() >= 0 and table.max() < 4
    assert json.loads((folder/'episodes.json').read_text()) == episodes
    for eps in episodes.values():
        for ep in eps:
            assert episode_signature(ep) == ep['signature']
    rows = read_csv(folder/'predictions.csv')
    score = np.load(folder/'scores.npy')
    assert score.shape == (len(rows),16) and score.dtype == np.float32
    assert np.isfinite(score).all()
    target = np.asarray([int(r['target']) for r in rows])
    prediction = score.argmax(-1)
    assert np.array_equal(prediction, [int(r['prediction']) for r in rows])
    assert np.array_equal(prediction == target, [int(r['correct']) for r in rows])
    assert np.allclose(((score-np.eye(16)[target])**2).mean(1),
                       [float(r['mse']) for r in rows], atol=1e-10,rtol=1e-8)
    groups = defaultdict(list)
    groups_all = defaultdict(list)
    groups_original = defaultdict(list)
    ids = set()
    bulk = rule in ('ema','erase_ema')
    erase = rule in ('erase_ema','local_delta')
    for r in rows:
        cell, arm = r['cell'],r['arm']
        i, q = int(r['episode']),int(r['rank'])
        ident = (cell,arm,i,q)
        assert ident not in ids
        ids.add(ident)
        ep = episodes[cell][i]
        query = ep['queries'][q]
        assert all(int(r[k]) == query[k] for k in ('key','target','old'))
        assert r['category'] == query['category']
        assert int(r['obsolete']) == (query['old'] >= 0 and int(r['prediction']) == query['old'])
        assert int(r['actual_writes']) == len(ep['keys'])
        assert int(r['state_write_count']) == (0 if arm == 'reset' else len(ep['keys']))
        start = (i//batch_size)*batch_size
        n = min(batch_size,len(episodes[cell])-start)
        source = -1 if arm == 'reset' else start+(i-start-1)%n if arm == 'rolled' else i
        assert int(r['state_source_episode']) == source
        head = int(table[query['key']])
        last = max(i for i,k in enumerate(ep['keys']) if k == query['key'])
        following = ep['keys'][last+1:]
        same = sum(table[k] == head for k in following)
        exposure = dict(query_module=head,after_last_real_target_observation=len(following),
                        following_same_module_writes=same,following_other_module_writes=len(following)-same,
                        target_decay_applications=same*bulk,target_projection_events=same*erase)
        for key,value in exposure.items():
            assert int(r[key]) == value, (folder,key,r[key],value)
        groups[(cell,arm,r['category'])].append(r)
        groups_all[(cell,arm,i)].append(int(r['correct']))
        if r['category'] != 'recent':
            groups[(cell,arm,'targets')].append(r)
            groups_original[(cell,arm,i)].append(int(r['correct']))
    expected_ids = {(c,arm,i,q) for c,eps in episodes.items()
                    for arm in (('online',) if c.startswith('pressure') else ('online','reset','rolled'))
                    for i,ep in enumerate(eps) for q in range(len(ep['queries']))}
    assert ids == expected_ids
    saved = read_csv(folder/'summary.csv')
    assert {(r['cell'],r['arm'],r['category']) for r in saved} == set(groups)
    assert len(saved) == len(groups)
    vectors, summaries = {},[]
    for (cell,arm,category),subset in groups.items():
        per_history = defaultdict(list)
        for r in subset:
            per_history[int(r['episode'])].append(r)
        assert set(per_history) == set(range(len(episodes[cell])))
        vectors[(cell,arm,category)] = np.asarray([np.mean([int(r['correct']) for r in per_history[i]])
                                                 for i in range(len(episodes[cell]))])
        metrics = dict(accuracy=float(np.mean([int(r['correct']) for r in subset])),
                       all_correct=float(np.mean([all(int(r['correct']) for r in group) for group in per_history.values()])),
                       mse=float(np.mean([float(r['mse']) for r in subset])))
        oldrate = float(np.mean([int(r['obsolete']) for r in subset]))
        original = next(r for r in saved if (r['cell'],r['arm'],r['category']) == (cell,arm,category))
        assert int(original['queries']) == len(subset)
        assert int(original['episodes']) == len(episodes[cell])
        for key,value in metrics.items():
            assert np.isclose(float(original[key]),value,atol=1e-12,rtol=1e-12)
        assert (np.isclose(float(original['obsolete_rate']),oldrate,atol=1e-12,rtol=1e-12)
                if category == 'revised' else original['obsolete_rate'] == '')
        summaries.append(dict(cell=cell,arm=arm,category=category,**metrics,obsolete_rate=oldrate))
    all_correct = {(cell,arm):np.asarray([all(groups_original[(cell,arm,i)]) for i in range(len(eps))],dtype=float)
                   for cell,eps in episodes.items()
                   for arm in (('online',) if cell.startswith('pressure') else ('online','reset','rolled'))}
    plan_ids = set()
    writes = 0
    for plan in json.loads((folder/'plans.json').read_text()):
        cell,arm,i = plan['cell'],plan['arm'],int(plan['episode'])
        assert (cell,arm,i) not in plan_ids
        plan_ids.add((cell,arm,i))
        ep = episodes[cell][i]
        expected = [dict(source=j,key=k,value=v,kind='online',after_online=j)
                    for j,(k,v) in enumerate(zip(ep['keys'],ep['values']))]
        assert plan['events'] == expected and plan['boundary'] == ep['boundary']
        writes += len(expected)
    assert plan_ids == set(groups_all)
    resource_ids = set()
    applications = dict(association=0,decay=0,projection=0)
    boundaries = 0
    resources = read_csv(folder/'resources.csv')
    for r in resources:
        cell,arm,start,count = r['cell'],r['arm'],int(r['batch_start']),int(r['batch'])
        ident = (cell,arm,start)
        assert ident not in resource_ids and start % batch_size == 0
        resource_ids.add(ident)
        eps = episodes[cell][start:start+count]
        assert count == min(batch_size,len(episodes[cell])-start)
        occupancy = np.asarray([np.bincount(table[ep['keys']],minlength=4) for ep in eps])
        assert np.array_equal(occupancy,json.loads(r['per_history_module_occupancy']))
        assert np.array_equal(occupancy.sum(0),json.loads(r['module_occupancy']))
        assert np.array_equal(occupancy*bulk,json.loads(r['per_history_decay_applications']))
        assert np.array_equal(occupancy*erase,json.loads(r['per_history_projection_applications']))
        assert int(r['active_module_updates']) == occupancy.sum()
        assert int(r['decay_applications']) == occupancy.sum()*bulk
        assert int(r['projection_applications']) == occupancy.sum()*erase
        applications['association'] += int(r['active_module_updates'])
        applications['decay'] += int(r['decay_applications'])
        applications['projection'] += int(r['projection_applications'])
        assert int(r['actual_writes']) == len(eps[0]['keys'])
        assert int(r['replay_writes']) == 0
        assert int(r['query_state_write_count']) == (0 if arm == 'reset' else len(eps[0]['keys']))
        assert int(r['final_fast_bytes']) == count*65536 == int(r['peak_fast_bytes'])
        assert float(r['per_stream_final_bytes']) == 65536
        assert float(r['boundary_error']) == float(r['midblock_error']) == 0
        assert r['resume_points'] == ('prefill,midblock' if eps[0]['delay'] else 'prefill')
        boundaries += len(r['resume_points'].split(','))
    expected_resources = {(cell,arm,start) for cell,eps in episodes.items()
                          for arm in (('online',) if cell.startswith('pressure') else ('online','reset','rolled'))
                          for start in range(0,len(eps),batch_size)}
    assert resource_ids == expected_resources
    assert applications['association'] == writes
    return dict(vectors=vectors,all_correct=all_correct,summaries=summaries,
                audit=dict(predictions=len(rows),writes=writes,resource_batches=len(resources),
                           serialization_boundaries=boundaries,applications=applications,
                           files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
