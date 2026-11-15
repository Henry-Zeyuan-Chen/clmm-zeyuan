"""Independently reconstruct saved integration traces and static checkpoints."""
from __future__ import annotations
import argparse
from collections import OrderedDict,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def rows(path):
    with path.open(newline='') as handle:return list(csv.DictReader(handle))
def signature(keys,values):return hashlib.sha256(bytes(keys)+b'|'+bytes(values)).hexdigest()
def near(a,b,tol=1e-10):assert np.isfinite(float(a)) and abs(float(a)-float(b))<=tol,(a,b)
def engineering(name):return name.startswith(('smoke','throughput','preflight'))
def codebook_hash(tensor):
    header=json.dumps(dict(shape=list(tensor.shape),dtype=str(tensor.dtype)),sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(header+b'\0'+tensor.detach().cpu().numpy().astype('<f4',copy=False).tobytes()).hexdigest()

def expected_events(ep,arm):
    observed=list(zip(ep['keys'],ep['values']))
    fifo=OrderedDict()
    result=[]
    def arrive(i):
        key=observed[i][0]
        fifo.pop(key,None);fifo[key]=i
        while len(fifo)>8:fifo.popitem(last=False)
    def event(i,kind,after):return dict(source=i,key=observed[i][0],value=observed[i][1],kind=kind,after_online=after)
    for i in range(ep['boundary']):result.append(event(i,'online',i));arrive(i)
    rng=np.random.default_rng(np.random.SeedSequence([ep['seed'],77]))
    for start in range(ep['boundary'],len(observed),4):
        stop=min(start+4,len(observed))
        selected=[int(rng.choice(list(fifo.values()))) for _ in range(stop-start)]
        for offset,i in enumerate(range(start,stop)):
            result.append(event(i,'online',i));arrive(i)
            if arm=='repeat':result.append(event(i,'extra',i))
            elif arm=='continuous':result.append(event(selected[offset],'extra',i))
        if arm=='periodic':result.extend(event(i,'extra',stop-1) for i in selected)
    assert len(result)==ep['boundary']+ep['delay']*(1 if arm=='online' else 2)
    return result

def training_data(run,meta):
    assert sha(run/'training_data.npz')==meta['training_data_sha256']
    with np.load(run/'training_data.npz') as raw:arrays={k:raw[k] for k in raw.files}
    plan=read(run/'training_plan.json');args=meta['args']
    assert len(plan)==args['steps']
    seen=set();answers=observations=0;bindings=set();key_ids=set()
    for index,spec in enumerate(plan):
        length,count=int(arrays['lengths'][index]),int(arrays['counts'][index])
        assert count==spec['facts'] and length==count+spec['revisions']+spec['delay']
        for field,n in (('keys',length),('values',length),('queries',count),('targets',count)):
            assert np.all(arrays[field][index,:,n:]==255)
        for b in range(args['batch_size']):
            keys=arrays['keys'][index,b,:length].tolist();values=arrays['values'][index,b,:length].tolist()
            query=arrays['queries'][index,b,:count].tolist();target=arrays['targets'][index,b,:count].tolist()
            assert all(0<=k<64 for k in keys) and all(0<=v<16 for v in values)
            assert query==keys[:count] and target==[dict(zip(keys,values))[k] for k in query]
            assert len(set(target))==count
            assert not set(keys[:count]).intersection(keys[count+spec['revisions']:])
            seen.add(signature(keys,values));bindings.update(zip(keys,values));key_ids.update(keys)
        answers+=args['batch_size']*count;observations+=args['batch_size']*length
    if meta['offline_training']:
        actual=rows(run/'training.csv')
        assert len(actual)==args['steps'] and [int(r['step']) for r in actual]==list(range(1,args['steps']+1))
        assert sum(int(r['answers']) for r in actual)==answers==meta['training']['answers']
        assert sum(int(r['observations']) for r in actual)==observations==meta['training']['observations']
        assert all(np.isfinite(float(r['loss'])) and np.isfinite(float(r['gradient_norm'])) and 0<float(r['eta'])<1 for r in actual)
    return seen,dict(planned_answers=answers,planned_observations=observations,unique_training_histories=len(seen),
                     training_atomic_bindings=len(bindings),training_key_ids=len(key_ids))

def check_episode(ep,seen):
    keys,vals=ep['keys'],ep['values'];k=ep['facts'];boundary=ep['boundary']
    assert signature(keys,vals)==ep['signature'] and ep['signature'] not in seen
    assert len(keys)==len(vals)==boundary+ep['delay'] and boundary==k+ep['revisions']
    assert len(set(keys[:k]))==k and not set(keys[:k]).intersection(keys[boundary:])
    latest={};old={}
    for key,value in zip(keys[:boundary],vals[:boundary]):
        if key in latest:assert latest[key]!=value;old[key]=latest[key]
        latest[key]=value
    distractors={}
    for key,value in zip(keys[boundary:],vals[boundary:]):
        assert key not in latest
        if key in distractors:assert distractors[key]==value
        distractors[key]=value
    target_values=[]
    for i,q in enumerate(ep['queries']):
        if q['category']=='recent':
            assert ep['delay'] and i==4 and q['key']==keys[-1] and q['target']==vals[-1] and q['old']==-1
        else:
            assert q['key']==keys[i] and q['target']==latest[q['key']]
            assert q['old']==old.get(q['key'],-1)
            assert q['category']==('revised' if q['key'] in old else 'stable')
            target_values.append(q['target'])
    assert len(set(target_values))==len(target_values)

def checkpoint_path(run,meta,root,stage):
    if meta['offline_training']:return run/('initial_checkpoint.pt' if stage=='initial-evaluation' else 'checkpoint.pt')
    remote=meta['args']['checkpoint'].replace('\\','/')
    parts=remote.split('/')
    for count in range(1,min(6,len(parts))+1):
        candidate=root.joinpath(*parts[-count:])
        if candidate.is_file():return candidate
    raise AssertionError(f'No retained parent checkpoint resolves {remote}')

def audit_stage(run,meta,seen,root,stage):
    folder=run/stage;data=read(folder/'episodes.json');pred=rows(folder/'predictions.csv');scores=np.load(folder/'scores.npy')
    assert scores.shape==(len(pred),16) and np.isfinite(scores).all()
    assert sha(folder/'scores.npy')==meta['initial_evaluation' if stage=='initial-evaluation' else 'evaluation']['scores_sha256']
    for eps in data.values():
        for ep in eps:check_episode(ep,seen)
    initial=stage=='initial-evaluation' or meta['args']['mode']=='fork'
    expected_arms={cell:(('online',) if initial else ('online','repeat','continuous','periodic')) if cell.startswith('pressure') else ('online','reset','rolled') for cell in data}
    plans=read(folder/'plans.json');plan_map={};writes=0
    for plan in plans:
        cell,arm,eid=plan['cell'],plan['arm'],plan['episode'];key=(cell,arm,eid)
        assert key not in plan_map and arm in expected_arms[cell]
        ep=data[cell][eid];ordinary=arm if arm in ('repeat','continuous','periodic') else 'online'
        want=expected_events(ep,ordinary)
        assert plan['events']==want and plan['boundary']==ep['boundary']
        plan_map[key]=want;writes+=len(want)
    assert len(plan_map)==sum(len(eps)*len(expected_arms[cell]) for cell,eps in data.items())
    actual_keys=set();groups=defaultdict(list)
    for i,row in enumerate(pred):
        cell,arm,eid,rank=row['cell'],row['arm'],int(row['episode']),int(row['rank'])
        key=(cell,arm,eid,rank);assert key not in actual_keys;actual_keys.add(key)
        ep=data[cell][eid];q=ep['queries'][rank]
        for field in ('key','target','old'):assert int(row[field])==q[field]
        assert row['category']==q['category'] and arm in expected_arms[cell]
        prediction=int(scores[i].argmax());assert prediction==int(row['prediction'])
        assert int(row['correct'])==int(prediction==q['target']) and int(row['obsolete'])==int(q['old']>=0 and prediction==q['old'])
        near(row['mse'],np.mean((scores[i].astype(float)-np.eye(16)[q['target']])**2))
        count=len(plan_map[cell,arm,eid]);assert int(row['actual_writes'])==count
        assert int(row['state_write_count'])==(0 if arm=='reset' else count)
        batch_start=eid//meta['args']['eval_batch_size']*meta['args']['eval_batch_size']
        batch_count=min(meta['args']['eval_batch_size'],len(data[cell])-batch_start)
        donor=-1 if arm=='reset' else batch_start+(eid-batch_start-1)%batch_count if arm=='rolled' else eid
        assert int(row['state_source_episode'])==donor
        groups[cell,arm].append(row)
    assert len(pred)==sum(len(ep['queries'])*len(expected_arms[cell]) for cell,eps in data.items() for ep in eps)
    summaries=rows(folder/'summary.csv')
    for s in summaries:
        chosen=[r for r in groups[s['cell'],s['arm']] if (r['category']!='recent' if s['category']=='targets' else r['category']==s['category'])]
        by_id=defaultdict(list)
        for r in chosen:by_id[r['episode']].append(r)
        assert len(chosen)==int(s['queries']) and len(by_id)==int(s['episodes'])
        near(s['accuracy'],np.mean([int(r['correct']) for r in chosen]));near(s['mse'],np.mean([float(r['mse']) for r in chosen]))
        near(s['all_correct'],np.mean([all(int(r['correct']) for r in group) for group in by_id.values()]))
        if s['category']=='revised':near(s['obsolete_rate'],np.mean([int(r['obsolete']) for r in chosen]))
    resources=rows(folder/'resources.csv')
    expected_batches=sum((len(eps)+meta['args']['eval_batch_size']-1)//meta['args']['eval_batch_size']*len(expected_arms[cell]) for cell,eps in data.items())
    assert len(resources)==expected_batches
    checkpoint=checkpoint_path(run,meta,root,stage)
    if not meta['offline_training']:assert sha(checkpoint)==meta['parent_sha256']
    checkpoint_payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    tensors=checkpoint_payload['model_state']
    assert all(isinstance(t,torch.Tensor) and t.dtype==torch.float32 and torch.isfinite(t).all() for t in tensors.values())
    assert codebook_hash(tensors['codebook.keys'])==meta['codebook_hash']==checkpoint_payload['codebook_hash']
    assert torch.equal(tensors['codebook.values'],torch.eye(64)[:16])
    if meta['method'] in ('hebbian','associative_gradient'):
        assert torch.equal(tensors['router_prototypes'],tensors['codebook.keys'][[0,16,32,48]])
        config=dict(checkpoint_payload['config'])
        if meta['args']['mode']=='fork':config.update(update_rule=meta['args']['rule'],routing=meta['args']['routing'])
        assert config==meta['config'] and meta['method']==('hebbian' if config['update_rule']=='hebbian' else 'associative_gradient')
    else:assert checkpoint_payload['config']==meta['config']
    literal=tensors['codebook.keys'].numpy();prototypes=literal[[0,16,32,48]]
    cosine=(literal/np.linalg.norm(literal,axis=1,keepdims=True))@(prototypes/np.linalg.norm(prototypes,axis=1,keepdims=True)).T
    routing=cosine.argmax(-1)
    for resource in resources:
        cell,arm,start=resource['cell'],resource['arm'],int(resource['batch_start']);size=int(resource['batch'])
        batch=data[cell][start:start+size];assert len(batch)==size
        plan=plan_map[cell,arm,start];online=len(batch[0]['keys']);extra=len(plan)-online
        assert int(resource['actual_writes'])==len(plan) and int(resource['replay_writes'])==extra
        assert int(resource['query_state_write_count'])==(0 if arm=='reset' else len(plan))
        near(resource['boundary_error'],0);near(resource['midblock_error'],0)
        assert resource['resume_points']==('prefill,midblock' if batch[0]['delay'] else 'prefill')
        assert int(resource['peak_buffer_pairs'])==(min(8,batch[0]['facts']) if arm in ('continuous','periodic') else 0)
        assert int(resource['peak_snapshot_pairs'])==(4 if arm in ('continuous','periodic') and batch[0]['delay'] else 0)
        if meta['method'] in ('hebbian','associative_gradient'):
            modules=4 if meta['config']['routing']=='dense' else 1
            assert int(resource['active_module_updates'])==size*len(plan)*modules
            expected=np.zeros(4,dtype=int)
            for eid in range(start,start+size):
                for event in plan_map[cell,arm,eid]:
                    if modules==4:expected+=1
                    else:expected[routing[event['key']]]+=1
            assert json.loads(resource['module_occupancy'])==expected.tolist()
            assert int(resource['final_fast_bytes'])==size*65536 and int(resource['peak_fast_bytes'])==size*65536
            near(resource['per_stream_final_bytes'],65536)
            assert 0<=float(resource['trace_saturation'])<=1
        else:
            want=65536 if arm=='reset' else size*131072 if arm=='rolled' or batch[0]['delay'] else size*65536
            assert int(resource['final_fast_bytes'])==want
            assert not resource['active_module_updates'] and not resource['trace_saturation']
    return dict(run=run.name,stage=stage,status='passed',prediction_rows=len(pred),planned_writes=writes,
                serialization_batches=len(resources),retained_checkpoint_sha256=sha(checkpoint)),data

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    results=[];data_groups={};seen_cache={};training=[];source_groups={};episode_groups={};incomplete=[]
    initial_by_seed={}
    for run in sorted(args.root.iterdir()):
        if not (run/'metadata.json').is_file():continue
        meta=read(run/'metadata.json')
        for name,digest in meta['source_sha256'].items():assert sha(run/'sources'/name)==digest
        if meta.get('design_sha256'):assert sha(run/'design.json')==meta['design_sha256']
        if meta['status']!='succeeded':incomplete.append(dict(run=run.name,status=meta['status'],failure=meta.get('failure')));continue
        digest=meta['training_data_sha256']
        if digest not in seen_cache:seen_cache[digest]=training_data(run,meta)
        else:training_data(run,meta)
        seen,budget=seen_cache[digest]
        if meta['offline_training']:
            for filename,key in (('initial_checkpoint.pt','initial_checkpoint_sha256'),('checkpoint.pt','checkpoint_sha256')):
                assert sha(run/filename)==meta[key]
            a=torch.load(run/'initial_checkpoint.pt',map_location='cpu',weights_only=True)
            b=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
            assert a['training']['step']==0 and b['training']==meta['training']
            changed=[name for name,t in a['model_state'].items() if not torch.equal(t,b['model_state'][name])]
            assert set(changed)<=set(('key_encoder','value_encoder','decoder','W0','alpha','raw_eta'))
            assert set(('codebook.keys','codebook.values','router_prototypes')).isdisjoint(changed)
            seed=meta['args']['init_seed']
            key=(run.name if engineering(run.name) else 'main',seed)
            if key in initial_by_seed:assert all(torch.equal(t,initial_by_seed[key][name]) for name,t in a['model_state'].items())
            else:initial_by_seed[key]=a['model_state']
            training.append(dict(run=run.name,answers=budget['planned_answers'],observations=budget['planned_observations'],changed_slow_tensors=changed))
        for stage in ('initial-evaluation','evaluation'):
            if not (run/stage).is_dir():continue
            result,data=audit_stage(run,meta,seen,args.root,stage);result.update(**budget)
            results.append(result)
            pairing=(digest,meta['args']['dev_seed'],meta['args']['eval_episodes'],meta['args']['pressure_episodes'],stage=='initial-evaluation' or meta['args']['mode']=='fork')
            hashed=sha(run/stage/'episodes.json')
            assert pairing not in episode_groups or episode_groups[pairing]==hashed
            episode_groups[pairing]=hashed
        if not engineering(run.name):
            identity=(meta['source_commit'],json.dumps(meta['source_sha256'],sort_keys=True))
            source_groups[run.name]=identity
    main_results=[r for r in results if not engineering(r['run'])]
    main_training=[r for r in training if not engineering(r['run'])]
    assert len(set(source_groups.values()))<=1
    report=dict(status='passed',scope='Saved provenance, labels, scores, plans, budgets and static tensors; no model inference or retraining',
        runs=results,incomplete_runs=incomplete,main_prediction_rows=sum(r['prediction_rows'] for r in main_results),
        main_planned_writes=sum(r['planned_writes'] for r in main_results),main_serialization_batches=sum(r['serialization_batches'] for r in main_results),
        total_prediction_rows=sum(r['prediction_rows'] for r in results),actual_training_runs=main_training,
        main_actual_training_answers=sum(r['answers'] for r in main_training),main_actual_training_observations=sum(r['observations'] for r in main_training),
        shared_training_array_groups=len(seen_cache),shared_episode_groups=len(episode_groups),
        main_frozen_source_commit=next(iter(source_groups.values()))[0] if source_groups else None)
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('runs','actual_training_runs')},indent=2))

if __name__=='__main__':main()
