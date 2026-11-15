"""Independent complete-phase raw audit, declared contrasts and qualification."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.dont_write_bytecode = True
from raw_eval_tools import audit_eval,read_csv,sha,signatures_from_arrays,write_csv
from paired_analysis_tools import summarize,SEEDS,ROUTES

p = argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--repo',type=Path)
p.add_argument('--prior-root',type=Path)
p.add_argument('--other-phase',type=Path)
a = p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
meta = json.loads((a.root/'metadata.json').read_text())
assert meta['status'] == 'succeeded' and meta['scope'] == 'complete predeclared phase'
mode = meta['args']['mode']
assert mode in ('frozen','train')
assert meta['offline_training'] == (mode == 'train')
assert int(meta['args']['steps']) == 1200 and int(meta['args']['batch_size']) == 32
assert int(meta['args']['eval_episodes']) == int(meta['args']['pressure_episodes']) == 128
assert int(meta['args']['eval_batch_size']) == 16
assert int(meta['args']['dev_seed']) == (90823 if mode == 'frozen' else 90923)
expected_rules = ('ema','additive','erase_ema','local_delta') if mode == 'frozen' else ('ema','local_delta')
expected = {(rule,s,r) for rule in expected_rules for s in SEEDS for r in ROUTES}
assert len(meta['cases']) == len(expected) == int(meta['expected_cases'])
episodes = json.loads((a.root/'episodes.json').read_text())
cells = {'acquire2','acquire4','revise2',*(f'pressure_k{k}_d{d}' for k in (8,16) for d in (0,16,64))}
assert set(episodes) == cells and all(len(eps)==128 for eps in episodes.values())
old = signatures_from_arrays(a.root/'old_training_exclusions.npz')
new = signatures_from_arrays(a.root/'training_data.npz')
seen = old|new
assert len(old) == meta['data']['old_unique_histories']
assert len(new) == meta['data']['new_unique_histories']
assert len(seen) == meta['data']['union_unique_histories']
assert sha(a.root/'training_data.npz') == meta['data']['training_data_sha256']
actual = {ep['signature'] for eps in episodes.values() for ep in eps}
assert not seen.intersection(actual)
stream_path = a.root/'sources/clmm/pretrain/kv_streams.py'
spec = importlib.util.spec_from_file_location('_frozen_streams_revision_audit',stream_path)
streams = importlib.util.module_from_spec(spec)
spec.loader.exec_module(streams)
for seed,filename in [(90502,'old_training_exclusions.npz'),(90802,'training_data.npz')]:
    intended,plan,_ = streams.training_data(1200,32,seed)
    with np.load(a.root/filename) as saved:
        assert set(saved.files)==set(intended)
        assert all(np.array_equal(saved[k],v) for k,v in intended.items())
    if seed == 90802:
        assert json.loads((a.root/'training_plan.json').read_text()) == plan
def selected(seed,excluded):
    data = streams.evaluation_data(seed,128,128,excluded,'full')
    return {k:v for k,v in data.items() if not k.startswith('pressure_k32_')}
prior = selected(90723,old)
prior_signatures = {ep['signature'] for eps in prior.values() for ep in eps}
seen |= prior_signatures
assert meta['data']['excluded_prior_development']['prior90723']==len(prior_signatures)
if mode=='train':
    frozen = selected(90823,seen)
    frozen_signatures = {ep['signature'] for eps in frozen.values() for ep in eps}
    seen |= frozen_signatures
    assert meta['data']['excluded_prior_development']['frozen90823']==len(frozen_signatures)
else:
    assert meta['data']['excluded_prior_development']['frozen90823']==0
assert len(seen)==meta['data']['all_excluded_signatures']
assert not seen.intersection(actual)
intended = streams.evaluation_data(int(meta['args']['dev_seed']),128,128,seen,'full')
intended = {k:v for k,v in intended.items() if not k.startswith('pressure_k32_')}
assert intended == episodes
overlap = {}
for label,path in [('prior_90723',a.prior_root),('other_phase',a.other_phase)]:
    if path is not None:
        prior = json.loads((path/'episodes.json').read_text())
        prior_ids = {ep['signature'] for eps in prior.values() for ep in eps}
        overlap[label] = len(actual&prior_ids)
        assert overlap[label] == 0
sources = {}
for filename,digest in meta['source_sha256'].items():
    saved = a.root/'sources'/filename
    assert sha(saved) == digest
    sources[filename] = dict(saved_sha256=digest)
    if a.repo is not None:
        blob = subprocess.check_output(['git','show',f"{meta['source_commit']}:{filename}"],cwd=a.repo)
        import hashlib
        assert hashlib.sha256(blob).hexdigest()==digest,filename
        sources[filename]['exact_frozen_git_blob_match'] = True
if 'design_sha256' in meta:
    assert sha(a.root/'design.json') == meta['design_sha256']
with np.load(a.root/'training_data.npz') as data:
    lengths,counts = data['lengths'],data['counts']
    expected_observations = 32*int(lengths.sum())
    expected_answers = 32*int(counts.sum())
case_data = {stage:{} for stage in (('final',) if mode=='frozen' else ('initial','final'))}
audits,summaries,training_rows = [],[],[]
initial_identities = defaultdict(set)
tables = {}
actual_case_ids = set()
for record in meta['cases']:
    folder = a.root/'runs'/record['name']
    info = json.loads((folder/'case.json').read_text())
    assert info == record and info['status']=='succeeded'
    route = 'content' if info['policy']=='content' else f"random-{info['random_seed']}"
    key = (info['revision_rule'],int(info['parent_seed']),route)
    assert key in expected and key not in actual_case_ids
    actual_case_ids.add(key)
    assert info['source_commit'] == meta['source_commit']
    assert info['offline_training'] == (mode=='train')
    assert info['parent_training']['step'] == (1200 if mode=='frozen' else 0)
    assert int(info['registered_parameters']) == 81921
    assert info['config']['revision_rule'] == key[0]
    assert info['config']['projection_eps'] == 1e-8
    assert info['config']['inactive_decay']=='protected'
    table = np.asarray(info['route_table'])
    assert np.array_equal(np.bincount(table,minlength=4),[14,20,9,21])
    assert np.array_equal(info['route_cardinalities'],[14,20,9,21])
    if route in tables:
        assert np.array_equal(tables[route],table)
    else:
        tables[route]=table
    initial_identities[key[1]].add((info['parent_sha256'],info['initial_slow_sha256']))
    assert sha(folder/'checkpoint.pt')==info['checkpoint_sha256']
    assert info['training']['step']==(1200 if mode=='train' else 0)
    assert info['training']['observations']==(expected_observations if mode=='train' else 0)
    assert info['training']['answers']==(expected_answers if mode=='train' else 0)
    if mode=='frozen':
        assert info['slow_sha256']==info['initial_slow_sha256']
        assert not (folder/'training.csv').exists()
    else:
        assert sha(folder/'initial_checkpoint.pt')==info['initial_checkpoint_sha256']
        trace = read_csv(folder/'training.csv')
        assert len(trace)==1200
        for i,row in enumerate(trace):
            assert int(row['step'])==i+1
            assert int(row['observations'])==32*int(lengths[i])
            assert int(row['answers'])==32*int(counts[i])
            assert np.isfinite(float(row['loss'])) and np.isfinite(float(row['gradient_norm']))
            assert 0<float(row['eta'])<1
        training_rows.append(dict(case=info['name'],rule=key[0],parent_seed=key[1],route=route,
                                  optimizer_steps=len(trace),observations=sum(int(r['observations']) for r in trace),
                                  answers=sum(int(r['answers']) for r in trace),terminal_loss=float(trace[-1]['loss']),
                                  training_seconds=info['training_seconds']))
    for stage in case_data:
        path = folder/('evaluation' if stage=='final' else 'initial-evaluation')
        metric = info['evaluation' if stage=='final' else 'initial_evaluation']
        assert metric['pure_reads'] and metric['slow_parameters_and_rng_unchanged']
        assert sha(path/'scores.npy')==metric['scores_sha256']
        result = audit_eval(path,episodes,table,key[0])
        assert metric['queries']==result['audit']['predictions']
        case_data[stage][key] = result
        audits.append(dict(case=info['name'],stage=stage,**result['audit']))
        for row in result['summaries']:
            summaries.append(dict(case=info['name'],stage=stage,rule=key[0],parent_seed=key[1],route=route,**row))
    print(f"audited {mode} {info['name']}",flush=True)
assert actual_case_ids==expected
assert all(len(values)==1 for values in initial_identities.values())
for route,table in tables.items():
    if route!='content':
        intended = np.random.Generator(np.random.PCG64(int(route.split('-')[1]))).permutation(tables['content'])
        assert np.array_equal(intended,table)
write_csv(a.out/'per_case_metrics.csv',summaries)
write_csv(a.out/'training_counters.csv',training_rows)
analyses = {}
for stage,data in case_data.items():
    analysis = summarize(data,draw_seed=90804)
    for name in ('gates','contrasts','aggregate','permap'):
        write_csv(a.out/f'{stage}_{name}.csv',analysis[name])
    analyses[stage] = {k:v for k,v in analysis.items() if k not in ('gates','contrasts','aggregate','permap')}
report = dict(status='passed',root=str(a.root.resolve()),mode=mode,source_commit=meta['source_commit'],
              full_phase_complete=True,case_count=len(expected),raw_audits=audits,sources=sources,
              histories_per_cell=128,exact_expected_arrays_and_plans=True,exact_expected_evaluation_episodes=True,
              exclusion_union=len(seen),zero_training_overlap=True,other_overlap=overlap,
              total_predictions=sum(r['predictions'] for r in audits),
              total_causal_writes=sum(r['writes'] for r in audits),
              total_resource_batches=sum(r['resource_batches'] for r in audits),
              analyses=analyses,
              scope='Saved artifacts and arithmetic only; no training rerun or efficacy inference rerun.')
(a.out/'raw_phase_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('raw_audits','sources')},indent=2))
