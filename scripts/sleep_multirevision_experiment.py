"""Fixed-checkpoint replay timing with K16 and three genuine revisions per key."""
from __future__ import annotations
import argparse
from collections import OrderedDict
from datetime import datetime,timezone
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from clmm.pretrain.kv_streams import training_data,signature
from clmm.sleep.multirevision import make_datasets,classify_events
from scripts.adapter_integration_experiment import evaluate
from scripts.revision_local_experiment import load_checkpoint,slow_sha
from scripts import fact_revision_experiment as common


def setup():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def collect_histories(value):
    if isinstance(value,dict):
        keys,values=value.get('keys'),value.get('values')
        if (isinstance(keys,list) and isinstance(values,list) and len(keys)==len(values)
            and keys and all(type(k) is int and 0<=k<64 for k in keys)
            and all(type(v) is int and 0<=v<16 for v in values)):
            return {signature(keys,values)}
        return set().union(*(collect_histories(v) for v in value.values())) if value else set()
    if isinstance(value,list):
        return set().union(*(collect_histories(v) for v in value)) if value else set()
    return set()


def capture(out):
    sources=common.capture_sources(out)
    for path in [*sorted((REPO/'clmm').rglob('*.py')),*sorted((REPO/'scripts').glob('*.py'))]:
        name=path.relative_to(REPO).as_posix();target=out/'sources'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);sources[name]=common.sha256(target)
    return sources


def prepare(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    seen=set();inputs=[]
    for seed in (90502,90802):
        arrays,_,ids=training_data(1200,32,seed);seen|=ids
        np.savez_compressed(out/f'training_exclusions_{seed}.npz',**arrays)
        inputs.append(dict(kind='training',seed=seed,histories=len(ids),sha256=common.sha256(out/f'training_exclusions_{seed}.npz')))
    if args.exclusions:
        seen|=set(json.loads(args.exclusions.read_text(encoding='utf8')))
        inputs.append(dict(kind='supplied_exclusions',sha256=common.sha256(args.exclusions)))
    seen_hashes=set()
    for root in ([] if args.exclusions else sorted(args.history_root.glob('clmm-*'))):
        if not root.is_dir() or root.resolve()==out:continue
        for path in sorted(root.rglob('episodes.json')):
            digest=common.sha256(path)
            if digest in seen_hashes:continue
            seen_hashes.add(digest)
            ids=collect_histories(json.loads(path.read_text(encoding='utf8')));seen|=ids
            inputs.append(dict(kind='prior_evaluation',path=path.relative_to(args.history_root).as_posix(),sha256=digest,histories=len(ids)))
    common.json_file(out/'excluded_signatures.json',sorted(seen))
    common.json_file(out/'exclusion_manifest.json',dict(inputs=inputs,union_histories=len(seen),signature_format='sha256(bytes(keys)+b|+bytes(values))'))
    data=make_datasets(args.dev_seed,args.episodes,seen)
    common.json_file(out/'episodes.json',data)
    source=json.loads((args.trained/'metadata.json').read_text(encoding='utf8'))
    assert source['status']=='succeeded' and source['source_commit']=='c9f5c153f786f1b82ac383d01c1d18c6c595946b'
    (out/'checkpoints').mkdir();models=[]
    for record in source['cases']:
        if record['revision_rule']!='local_delta':continue
        path=args.trained/'runs'/record['name']/'checkpoint.pt'
        assert common.sha256(path)==record['checkpoint_sha256'] and record['training']['step']==1200
        target=out/'checkpoints'/f"{record['name']}.pt";shutil.copy2(path,target)
        models.append(dict(name=record['name'],parent_seed=record['parent_seed'],policy=record['policy'],random_seed=record['random_seed'],config=record['config'],slow_sha256=record['slow_sha256'],route_table=record['route_table'],checkpoint=target.relative_to(out).as_posix(),checkpoint_sha256=common.sha256(target),original_training=record['training']))
    assert len(models)==12
    common.json_file(out/'models.json',models)
    sources=capture(out)
    if args.design:shutil.copy2(args.design,out/'design.json')
    meta=dict(status='succeeded',mode='prepare',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),source_sha256=sources,
        data_sha256=common.sha256(out/'episodes.json'),exclusions_sha256=common.sha256(out/'excluded_signatures.json'),episodes=args.episodes,development_seed=args.dev_seed,
        model_count=len(models),offline_training=False,command=[sys.executable,*sys.argv],environment=dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__)))
    common.json_file(out/'metadata.json',meta)
    print(json.dumps(dict(prepared=str(out),models=len(models),cells=list(data),histories=args.episodes,exclusions=len(seen))),flush=True)


def annotate(out,model,data):
    plans=json.loads((out/'evaluation/plans.json').read_text(encoding='utf8'))
    lookup={};histories=[]
    for plan in plans:
        cell,arm,e=plan['cell'],plan['arm'],plan['episode'];ep=data[cell][e]
        classified=classify_events(ep,plan['events'])
        extras=[event for event in classified if event['kind']=='extra']
        history=dict(cell=cell,arm=arm,episode=e,events=len(classified),extra_writes=len(extras),stale_extra_writes=sum(event['stale'] for event in extras),final_obsolete_query_extras=sum(event['obsolete_anyquery'] for event in extras))
        histories.append(history)
        fifo=OrderedDict();eligible={q['key']:0 for q in ep['queries']}
        for i,(key,value) in enumerate(zip(ep['keys'],ep['values'])):
            if i>=ep['boundary'] and (i-ep['boundary'])%4==0:
                for q in ep['queries']:eligible[q['key']]+=int(q['key'] in fifo)
            fifo.pop(key,None);fifo[key]=value
            while len(fifo)>8:fifo.popitem(last=False)
        for rank,q in enumerate(ep['queries']):
            versions=q.get('versions')
            if versions is None:
                versions=[]
                for key,value in zip(ep['keys'],ep['values']):
                    if key==q['key'] and (not versions or versions[-1]!=value):versions.append(value)
            direct=[(i,event) for i,event in enumerate(classified) if event['key']==q['key']]
            correct=[i for i,event in direct if event['value']==q['target']]
            assert direct and correct
            last_good=correct[-1];last=direct[-1][1];queryextras=[event for event in extras if event['key']==q['key']]
            lookup[(cell,arm,e,rank)]=dict(query_group=q.get('group',q['category']),initial_value=versions[0],
                version_history=json.dumps(versions),stale_extra_writes=history['stale_extra_writes'],
                query_extra_writes=len(queryextras),query_stale_extra_writes=sum(event['stale'] for event in queryextras),
                query_final_obsolete_extra_writes=sum(event['obsolete_anyquery'] for event in queryextras),
                fifo_eligible_blocks=eligible[q['key']] if arm in ('continuous','periodic') else None,
                final_fifo_has_key=int(q['key'] in fifo) if arm in ('continuous','periodic') else None,
                final_applied_value_for_key=last['value'],last_direct_write_kind=last['kind'],last_direct_write_source=last['source'],
                last_target_valued_event=last_good,subsequent_obsolete_same_key_writes=sum(i>last_good and event['value']!=q['target'] for i,event in direct))
    with (out/'evaluation/predictions.csv').open(newline='',encoding='utf8') as handle:rows=list(csv.DictReader(handle))
    for row in rows:
        key=(row['cell'],row['arm'],int(row['episode']),int(row['rank']));q=data[key[0]][key[2]]['queries'][key[3]]
        row.update(lookup[key]);row['obsolete_any']=int(int(row['prediction'])!=q['target'] and int(row['prediction']) in json.loads(row['version_history'])[:-1])
    common.write_csv(out/'evaluation/predictions.csv',rows)
    common.write_csv(out/'evaluation/replay_audit.csv',histories)
    # Same literal extras in the same sequence; only placement among arrivals changes.
    pairs={}
    for plan in plans:
        if plan['arm'] not in ('continuous','periodic'):continue
        extra=[(v['source'],v['key'],v['value']) for v in plan['events'] if v['kind']=='extra']
        pairs.setdefault((plan['cell'],plan['episode']),{})[plan['arm']]=extra
    assert all(set(pair)=={'continuous','periodic'} and pair['continuous']==pair['periodic'] for pair in pairs.values())
    return dict(matched_literal_extra_sequences=len(pairs),observer_annotations_only=True,query_metadata_not_passed_to_writes=True)


def evaluate_case(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+args.max_seconds
    packet=json.loads((args.packet/'metadata.json').read_text(encoding='utf8'))
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if source_commit!=packet['source_commit']:raise ValueError('Prepared source commit mismatch')
    for name,digest in packet['source_sha256'].items():
        if common.sha256(REPO/name)!=digest:raise ValueError('Prepared source bytes changed')
    assert common.sha256(args.packet/'episodes.json')==packet['data_sha256']
    data=json.loads((args.packet/'episodes.json').read_text(encoding='utf8'))
    record=next(r for r in json.loads((args.packet/'models.json').read_text(encoding='utf8')) if r['name']==args.model)
    checkpoint=args.packet/record['checkpoint'];assert common.sha256(checkpoint)==record['checkpoint_sha256']
    model,payload=load_checkpoint(checkpoint,torch.device('cpu'));model.eval().requires_grad_(False)
    assert payload['training']['step']==1200 and model.config.revision_rule=='local_delta'
    assert model.config.to_dict()==record['config'] and slow_sha(model)==record['slow_sha256']
    meta=dict(**record,status='running',source_commit=source_commit,offline_training=False,additional_optimizer_steps=0,
        data_sha256=packet['data_sha256'],command=[sys.executable,*sys.argv],
        started_utc=datetime.now(timezone.utc).isoformat(),environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device='cpu',threads=1,deterministic=True,dtype='float32'))
    common.json_file(out/'metadata.json',meta)
    try:
        namespace=argparse.Namespace(eval_batch_size=16)
        meta['evaluation']=evaluate(model,namespace,out/'evaluation',set(),deadline,data_override=data)
        meta['observer']=annotate(out,model,data)
        assert slow_sha(model)==meta['slow_sha256']
        for name,digest in packet['source_sha256'].items():assert common.sha256(REPO/name)==digest
        common.before_deadline(deadline);meta['status']='succeeded'
    except BaseException as error:
        meta.update(status='failed',failure=repr(error),traceback=traceback.format_exc());raise
    finally:
        meta.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
        common.json_file(out/'metadata.json',meta)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','evaluate'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--trained',type=Path);p.add_argument('--history-root',type=Path)
    p.add_argument('--design',type=Path);p.add_argument('--dev-seed',type=int,default=91123);p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--exclusions',type=Path)
    p.add_argument('--packet',type=Path);p.add_argument('--model');p.add_argument('--max-seconds',type=int,default=1200)
    return p


if __name__=='__main__':
    args=parser().parse_args();setup();(prepare if args.mode=='prepare' else evaluate_case)(args)
