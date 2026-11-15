"""Bounded frozen-parent content/random x inactive-decay diagnostic (no training)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from clmm.hebbian.routing_decay import RoutingDecayAdapter, RoutingDecayConfig
from clmm.pretrain.adapter_registry import load_adapter_checkpoint
from clmm.pretrain.kv_streams import training_data
from scripts.adapter_integration_experiment import evaluate, datasets
from scripts import fact_revision_experiment as common


def case_specs():
    return [dict(name=f'{rule}-{seed}-{policy if map_seed is None else f"random-{map_seed}"}-{decay}',
                 rule=rule,parent_seed=seed,policy=policy,random_seed=map_seed,inactive_decay=decay,
                 parent=f'{rule}-dense-{seed}.pt')
            for rule in ('gradient','hebbian') for seed in (90511,90515)
            for policy,map_seed in [('content',None),*[('random',s) for s in range(90701,90706)]]
            for decay in ('protected','global')]


def slow_sha(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name=='route_table':continue
        t=value.detach().cpu().contiguous()
        digest.update(name.encode()+b'\0'+str(tuple(t.shape)).encode()+b'\0'+t.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path,model,meta):
    model.assert_codebook_valid()
    torch.save(dict(format='clmm.routing_decay.v1',config=model.config.to_dict(),
                    model_state=model.state_dict(),method=model.method,
                    parent_sha256=meta['parent_sha256'],slow_sha256=slow_sha(model),
                    source_commit=meta['source_commit']),path)


def load_checkpoint(path,device):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format')!='clmm.routing_decay.v1':raise ValueError('Wrong routing checkpoint format')
    config=RoutingDecayConfig(**payload['config'])
    state=payload['model_state']
    with torch.random.fork_rng(devices=[]):
        model=RoutingDecayAdapter(config,state['codebook.keys'],state['codebook.values'],route_table=state['route_table']).to(device)
    model.load_state_dict(state,strict=True)
    if payload['method']!=model.method or payload['slow_sha256']!=slow_sha(model):
        raise ValueError('Checkpoint slow identity differs')
    return model,payload


def run(args):
    if min(args.eval_episodes,args.pressure_episodes,args.eval_batch_size,args.max_seconds)<1:raise ValueError('Positive counts required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    device=torch.device(args.device)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();deadline=started+args.max_seconds
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if args.source_commit and args.source_commit!=commit:raise ValueError('Source commit mismatch')
    sources=common.capture_sources(out)
    for path in [*sorted((REPO/'clmm').rglob('*.py')),Path(__file__),REPO/'scripts/adapter_integration_experiment.py',REPO/'scripts/kv_representation_control.py']:
        name=path.relative_to(REPO).as_posix();destination=out/'sources'/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        sources[name]=common.sha256(destination)
    if args.design:shutil.copy2(args.design,out/'design.json')
    meta=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),source_commit=commit,
              source_sha256=sources,command=[sys.executable,*sys.argv],offline_training=False,
              args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},cases=[],
              scope='full48 factorial' if args.case_limit==48 else 'engineering subset',expected_cases=args.case_limit,
              environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device=str(device),
                  device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',deterministic=True,dtype='float32'))
    if args.design:meta['design_sha256']=common.sha256(out/'design.json')
    def save():common.json_file(out/'metadata.json',meta)
    save()
    try:
        # Reconstructed only for exact whole-history exclusions; never optimized.
        arrays,plans,seen=training_data(1200,32,90502)
        np.savez_compressed(out/'exclusion_training_data.npz',**arrays)
        common.json_file(out/'exclusion_training_plan.json',plans)
        meta['exclusion_unique_histories']=len(seen)
        data=datasets(args,seen,'comparison')
        common.json_file(out/'episodes.json',data)
        (out/'runs').mkdir()
        specs=case_specs()[:args.case_limit]
        for spec in specs:
            common.before_deadline(deadline)
            tick=time.perf_counter();folder=out/'runs'/spec['name'];folder.mkdir()
            checkpoint=args.parents/spec['parent']
            parent,payload=load_adapter_checkpoint(checkpoint,device)
            if parent.config.update_rule!=spec['rule'] or parent.config.routing!='dense':raise ValueError('Wrong dense parent')
            parent_slow=slow_sha(parent)
            model=RoutingDecayAdapter.from_parent(parent,policy=spec['policy'],inactive_decay=spec['inactive_decay'],random_seed=spec['random_seed'])
            if slow_sha(model)!=parent_slow:raise AssertionError('Frozen parent bits changed')
            model.eval().requires_grad_(False);del parent
            record=dict(**spec,status='running',source_commit=commit,parent_sha256=common.sha256(checkpoint),
                        parent_training=payload['training'],slow_sha256=parent_slow,config=model.config.to_dict(),
                        codebook_hash=model.codebook_hash,parameters=model.parameter_counts(),offline_training=False,
                        route_table=model.route_table.cpu().tolist(),route_cardinalities=torch.bincount(model.route_table,minlength=4).cpu().tolist(),
                        deviations=model.config_deviations)
            meta['cases'].append(record);save();common.json_file(folder/'case.json',record)
            save_checkpoint(folder/'checkpoint.pt',model,record)
            reloaded,_=load_checkpoint(folder/'checkpoint.pt',device)
            if any(not torch.equal(t,reloaded.state_dict()[k]) for k,t in model.state_dict().items()):raise AssertionError('Checkpoint bits changed')
            del reloaded
            record['evaluation']=evaluate(model,args,folder/'evaluation',seen,deadline,data_override=data,pressure_arms_override=('online',))
            record.update(status='succeeded',elapsed_seconds=time.perf_counter()-tick,checkpoint_sha256=common.sha256(folder/'checkpoint.pt'))
            common.json_file(folder/'case.json',record);save()
            print(json.dumps(dict(completed_case=spec['name'],elapsed_seconds=record['elapsed_seconds'])),flush=True)
            del model
        for name,digest in sources.items():
            if common.sha256(REPO/name)!=digest or common.sha256(out/'sources'/name)!=digest:raise AssertionError('Source changed during run')
        common.before_deadline(deadline);meta['status']='succeeded'
    except BaseException as error:
        meta.update(status='failed',failure=repr(error),traceback=traceback.format_exc());raise
    finally:
        meta.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
        if device.type=='cuda':meta['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        save()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--design',type=Path);p.add_argument('--source-commit')
    p.add_argument('--dev-seed',type=int,default=90723)
    p.add_argument('--eval-episodes',type=int,default=128);p.add_argument('--pressure-episodes',type=int,default=128)
    p.add_argument('--eval-batch-size',type=int,default=16);p.add_argument('--case-limit',type=int,choices=range(1,49),default=48)
    p.add_argument('--device',default='cpu');p.add_argument('--max-seconds',type=int,default=3300)
    return p


if __name__=='__main__':run(parser().parse_args())
