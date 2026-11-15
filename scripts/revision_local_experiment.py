"""Reference key-local revision: frozen factorial and matched initial/final training."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
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
import torch.nn.functional as F
from clmm.hebbian.revision import RevisionAdapter, RevisionConfig
from clmm.pretrain.adapter_registry import load_adapter_checkpoint
from clmm.pretrain.kv_streams import training_data,evaluation_data
from scripts.adapter_integration_experiment import evaluate, datasets
from scripts.routing_decay_experiment import slow_sha
from scripts import fact_revision_experiment as common


def case_specs(mode):
    rules=('ema','additive','erase_ema','local_delta') if mode=='frozen' else ('ema','local_delta')
    return [dict(name=f'{seed}-{label}-{rule}',parent_seed=seed,policy=policy,random_seed=map_seed,
                 revision_rule=rule,parent=f'hebbian-{"dense" if mode=="frozen" else "initial"}-{seed}.pt')
        for seed in (90511,90515)
        for policy,map_seed,label in [('content',None,'content'),*[('random',s,f'random-{s}') for s in range(90701,90706)]]
        for rule in rules]


def save_checkpoint(path,model,record,training):
    model.assert_codebook_valid()
    torch.save(dict(format='clmm.revision.local.v1',config=model.config.to_dict(),
        model_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()},
        method=model.method,slow_sha256=slow_sha(model),parent_sha256=record['parent_sha256'],
        source_commit=record['source_commit'],training=training),path)


def load_checkpoint(path,device):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format')!='clmm.revision.local.v1':raise ValueError('Wrong revision checkpoint format')
    config=RevisionConfig(**payload['config']);state=payload['model_state']
    with torch.random.fork_rng(devices=[]):
        model=RevisionAdapter(config,state['codebook.keys'],state['codebook.values'],route_table=state['route_table']).to(device)
    model.load_state_dict(state,strict=True)
    if model.method!=payload['method'] or slow_sha(model)!=payload['slow_sha256']:raise ValueError('Checkpoint identity changed')
    return model,payload


def phase_data(args,old_seen,new_seen):
    def selected(seed,seen):
        return {k:v for k,v in evaluation_data(seed,128,128,seen,'full').items() if not k.startswith('pressure_k32_')}
    # Exact prior90723 protocol, then disjoint new development phases.
    prior=selected(90723,old_seen)
    prior_signatures={ep['signature'] for eps in prior.values() for ep in eps}
    seen=old_seen|new_seen|prior_signatures
    excluded_frozen=set()
    if args.mode=='train':
        frozen=selected(90823,seen)
        excluded_frozen={ep['signature'] for eps in frozen.values() for ep in eps}
        seen=seen|excluded_frozen
    return datasets(args,seen,'comparison'),seen,dict(prior90723=len(prior_signatures),frozen90823=len(excluded_frozen))


def run(args):
    if min(args.steps,args.batch_size,args.eval_episodes,args.pressure_episodes,args.eval_batch_size,args.max_seconds)<1:
        raise ValueError('Positive counts required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    train_device=torch.device(args.train_device);eval_device=torch.device(args.eval_device)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();deadline=started+args.max_seconds
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if args.source_commit and args.source_commit!=commit:raise ValueError('Source commit mismatch')
    sources=common.capture_sources(out)
    for path in [*sorted((REPO/'clmm').rglob('*.py')),*sorted((REPO/'scripts').glob('*.py'))]:
        name=path.relative_to(REPO).as_posix();destination=out/'sources'/name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        sources[name]=common.sha256(destination)
    if args.design:shutil.copy2(args.design,out/'design.json')
    meta=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),source_commit=commit,
        source_sha256=sources,command=[sys.executable,*sys.argv],offline_training=args.mode=='train',
        scope='complete predeclared phase' if args.case_limit is None else 'engineering subset',
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},cases=[],
        environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,
            train_device=str(train_device),eval_device=str(eval_device),deterministic=True,dtype='float32',
            gpu_name=torch.cuda.get_device_name(train_device) if train_device.type=='cuda' else None))
    if args.design:meta['design_sha256']=common.sha256(out/'design.json')
    def save():common.json_file(out/'metadata.json',meta)
    save()
    try:
        old,old_plan,old_seen=training_data(1200,32,90502)
        arrays,plans,new_seen=training_data(args.steps,args.batch_size,90802)
        np.savez_compressed(out/'old_training_exclusions.npz',**old)
        np.savez_compressed(out/'training_data.npz',**arrays)
        common.json_file(out/'training_plan.json',plans)
        data,seen,prior_development=phase_data(args,old_seen,new_seen)
        meta['data']=dict(old_unique_histories=len(old_seen),new_unique_histories=len(new_seen),
            union_unique_histories=len(old_seen|new_seen),all_excluded_signatures=len(seen),excluded_prior_development=prior_development,training_data_sha256=common.sha256(out/'training_data.npz'),
            note='Frozen mode only regenerates exclusions and never optimizes these arrays')
        common.json_file(out/'episodes.json',data)
        specs=case_specs(args.mode)
        if args.case_limit is not None:specs=specs[:args.case_limit]
        meta['expected_cases']=len(specs);(out/'runs').mkdir();save()
        for spec in specs:
            common.before_deadline(deadline);tick=time.perf_counter()
            folder=out/'runs'/spec['name'];folder.mkdir()
            parent_path=args.parents/spec['parent'];parent,payload=load_adapter_checkpoint(parent_path,torch.device('cpu'))
            expected_parent_step=1200 if args.mode=='frozen' else 0
            if payload['training']['step']!=expected_parent_step or parent.config.update_rule!='hebbian' or parent.config.routing!='dense':
                raise ValueError('Incorrect parent training state/rule/access')
            model=RevisionAdapter.from_parent(parent,revision_rule=spec['revision_rule'],policy=spec['policy'],random_seed=spec['random_seed'])
            initial_hash=slow_sha(model)
            if initial_hash!=slow_sha(parent):raise AssertionError('Initial parent tensor values changed')
            del parent
            record=dict(**spec,status='running',source_commit=commit,parent_sha256=common.sha256(parent_path),
                parent_training=payload['training'],initial_slow_sha256=initial_hash,config=model.config.to_dict(),
                route_table=model.route_table.tolist(),route_cardinalities=torch.bincount(model.route_table,minlength=4).tolist(),
                codebook_hash=model.codebook_hash,registered_parameters=model.parameter_counts()['registered'],
                deviations={**model.config_deviations,'experiment':args.mode},offline_training=args.mode=='train')
            meta['cases'].append(record);save()
            def save_case():common.json_file(folder/'case.json',record)
            save_case()
            training=dict(step=0,init_seed=spec['parent_seed'],train_seed=90802,observations=0,answers=0)
            model=model.to(eval_device).eval().requires_grad_(False)
            if args.mode=='train':
                save_checkpoint(folder/'initial_checkpoint.pt',model,record,training)
                t=time.perf_counter()
                record['initial_evaluation']=evaluate(model,args,folder/'initial-evaluation',seen,deadline,data_override=data,pressure_arms_override=('online',))
                record['initial_evaluation_seconds']=time.perf_counter()-t
                model=model.to(train_device).train().requires_grad_(True)
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
                train_start=time.perf_counter();trace=[]
                for step in range(args.steps):
                    common.before_deadline(deadline)
                    length,count=int(arrays['lengths'][step]),int(arrays['counts'][step])
                    state=model.initial_state(args.batch_size);optimizer.zero_grad(set_to_none=True)
                    for index in range(length):
                        state=model.write(torch.tensor(arrays['keys'][step,:,index].astype(np.int64),device=train_device),
                            torch.tensor(arrays['values'][step,:,index].astype(np.int64),device=train_device),state)
                    query=torch.tensor(arrays['queries'][step,:,:count].astype(np.int64),device=train_device)
                    target=torch.tensor(arrays['targets'][step,:,:count].astype(np.int64),device=train_device)
                    scores=model.read(query,state);loss=F.mse_loss(scores,F.one_hot(target,16).float())
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();common.sync(train_device)
                    row=dict(step=step+1,answers=args.batch_size*count,observations=args.batch_size*length,
                        loss=float(loss.detach()),gradient_norm=float(norm),eta=float(model.eta.detach()),
                        elapsed_seconds=time.perf_counter()-train_start)
                    trace.append(row)
                    with (folder/'training.csv').open('a',newline='') as handle:
                        import csv
                        writer=csv.DictWriter(handle,fieldnames=row)
                        if step==0:writer.writeheader()
                        writer.writerow(row)
                    if step==0 or (step+1)%100==0:print(json.dumps(dict(case=spec['name'],training=row)),flush=True)
                training.update(step=args.steps,answers=sum(r['answers'] for r in trace),observations=sum(r['observations'] for r in trace))
                record['training_seconds']=time.perf_counter()-train_start
                torch.save(optimizer.state_dict(),folder/'optimizer.pt');del optimizer
                model=model.to(eval_device).eval().requires_grad_(False)
            record['training']=training
            save_checkpoint(folder/'checkpoint.pt',model,record,training)
            reloaded,_=load_checkpoint(folder/'checkpoint.pt',eval_device)
            if any(not torch.equal(t,reloaded.state_dict()[k]) for k,t in model.state_dict().items()):raise AssertionError('Checkpoint bits changed')
            del reloaded
            t=time.perf_counter()
            record['evaluation']=evaluate(model,args,folder/'evaluation',seen,deadline,data_override=data,pressure_arms_override=('online',))
            record['evaluation_seconds']=time.perf_counter()-t
            record.update(status='succeeded',elapsed_seconds=time.perf_counter()-tick,slow_sha256=slow_sha(model),checkpoint_sha256=common.sha256(folder/'checkpoint.pt'))
            if args.mode=='train':record['initial_checkpoint_sha256']=common.sha256(folder/'initial_checkpoint.pt')
            save_case();save();del model
            print(json.dumps(dict(completed_case=spec['name'],mode=args.mode,elapsed_seconds=record['elapsed_seconds'])),flush=True)
        for name,digest in sources.items():
            if common.sha256(REPO/name)!=digest or common.sha256(out/'sources'/name)!=digest:raise AssertionError('Source changed during phase')
        common.before_deadline(deadline);meta['status']='succeeded'
    except BaseException as error:
        meta.update(status='failed',failure=repr(error),traceback=traceback.format_exc());raise
    finally:
        meta.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
        if train_device.type=='cuda':meta['peak_training_cuda_bytes']=torch.cuda.max_memory_allocated(train_device)
        save()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('frozen','train'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--design',type=Path);p.add_argument('--source-commit')
    p.add_argument('--train-device',default='cpu');p.add_argument('--eval-device',default='cpu')
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--dev-seed',type=int,required=True)
    p.add_argument('--eval-episodes',type=int,default=128);p.add_argument('--pressure-episodes',type=int,default=128)
    p.add_argument('--eval-batch-size',type=int,default=16);p.add_argument('--case-limit',type=int)
    p.add_argument('--max-seconds',type=int,default=3300)
    return p


if __name__=='__main__':run(parser().parse_args())
