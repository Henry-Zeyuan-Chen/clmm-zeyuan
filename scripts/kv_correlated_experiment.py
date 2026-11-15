"""Bounded nonlinear correlated-key calibration and frozen replay evaluation."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
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
import torch.nn.functional as F
from clmm.pretrain.kv_correlated import CorrelatedConfig,CorrelatedNeuralAdapter
from clmm.pretrain.kv_streams import training_data,evaluation_data
from scripts import fact_revision_experiment as common
from scripts.kv_representation_control import evaluate


def save_checkpoint(path,model,meta,step):
    payload=dict(format='clmm.kv.correlated.v1',method=model.method,config=model.config.to_dict(),
        model_state=model.state_dict(),source_commit=meta['source_commit'],source_sha256=meta['source_sha256'],
        codebook_hash=model.codebook_hash,
        training=dict(step=step,init_seed=meta['args']['init_seed'],train_seed=meta['args']['train_seed'],
                      objective='mean squared read-coordinate error versus one-hot value',outer_lr=meta['args']['lr']))
    torch.save(payload,path)
    return common.sha256(path)


def load_checkpoint(path,device):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format')!='clmm.kv.correlated.v1' or payload.get('method')!='neural_correlated':
        raise ValueError('Not a correlated nonlinear checkpoint')
    model=CorrelatedNeuralAdapter(CorrelatedConfig(**payload['config'])).to(device)
    model.load_state_dict(payload['model_state'],strict=True)
    model.assert_codebook_valid()
    if payload['codebook_hash']!=model.codebook_hash:
        raise ValueError('Checkpoint codebook identity mismatch')
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite checkpoint')
    return model,payload


def dataset(args,seen):
    data=evaluation_data(args.dev_seed,args.eval_episodes,args.pressure_episodes,seen,'full')
    if args.phase in ('calibration','confirmation','smoke'):
        data={name:episodes for name,episodes in data.items() if not name.startswith('pressure') or name.startswith('pressure_k8_')}
    return data


def run(args):
    if min(args.steps,args.batch_size,args.eval_episodes,args.pressure_episodes,args.eval_batch_size,args.threads,args.max_seconds)<1:
        raise ValueError('Positive resource and sample limits required')
    if args.mode=='train' and args.checkpoint:
        raise ValueError('Calibration trains from declared fresh initialization, not warm starts')
    if args.mode=='train' and args.phase not in ('calibration','smoke'):
        raise ValueError('Training is only a calibration operation')
    args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    device=torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.init_seed)
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    sources=common.capture_sources(args.out)
    for path in (Path(__file__),REPO/'scripts/kv_representation_control.py'):
        name=path.relative_to(REPO).as_posix()
        (args.out/'sources'/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,args.out/'sources'/name)
        sources[name]=common.sha256(args.out/'sources'/name)
    meta=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_sha256=sources,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        offline_training=args.mode=='train',evaluation_pressure_arms=['online'] if args.phase!='comparison' else ['online','repeat','continuous','periodic','boundary_reset'],
        environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device=str(device),
            device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',dtype='float32',deterministic=True))
    if args.design:
        shutil.copy2(args.design,args.out/'design.json')
        meta['design_sha256']=common.sha256(args.out/'design.json')
    common.json_file(args.out/'metadata.json',meta)
    start=time.perf_counter()
    deadline=start+args.max_seconds
    model=None
    try:
        arrays,plan,seen=training_data(args.steps,args.batch_size,args.train_seed)
        np.savez_compressed(args.out/'training_data.npz',**arrays)
        common.json_file(args.out/'training_plan.json',plan)
        meta['training_data_sha256']=common.sha256(args.out/'training_data.npz')
        meta['planned_exclusion_histories']=len(seen)
        if args.checkpoint:
            model,parent=load_checkpoint(args.checkpoint,device)
            meta['parent_sha256']=common.sha256(args.checkpoint)
            meta['parent_training']=parent['training']
            if model.config.rank!=args.rank or model.config.codebook_seed!=args.codebook_seed:
                raise ValueError('CLI and checkpoint geometry disagree')
        else:
            model=CorrelatedNeuralAdapter(CorrelatedConfig(rank=args.rank,codebook_seed=args.codebook_seed)).to(device)
        meta.update(method=model.method,config=model.config.to_dict(),codebook_hash=model.codebook_hash,
                    parameters=model.parameter_counts(),deviations=model.config_deviations)
        data=dataset(args,seen)
        pressure_arms=('online',) if args.phase!='comparison' else None
        if args.mode=='train':
            meta['initial_checkpoint_sha256']=save_checkpoint(args.out/'initial_checkpoint.pt',model,meta,0)
            meta['initial_evaluation']=evaluate(model,args,args.out/'initial-evaluation',seen,deadline,
                                                  datasets=data,pressure_arms=pressure_arms)
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=0.)
            rows=[]
            for step in range(args.steps):
                common.before_deadline(deadline)
                length,count=int(arrays['lengths'][step]),int(arrays['counts'][step])
                state=model.initial_state(args.batch_size)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for index in range(length):
                    state=model.write(torch.tensor(arrays['keys'][step,:,index].astype(np.int64),device=device),
                                      torch.tensor(arrays['values'][step,:,index].astype(np.int64),device=device),state)
                queries=torch.tensor(arrays['queries'][step,:,:count].astype(np.int64),device=device)
                targets=torch.tensor(arrays['targets'][step,:,:count].astype(np.int64),device=device)
                scores=model.read(queries,state)
                loss=F.mse_loss(scores,F.one_hot(targets,16).float())
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite outer loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step()
                common.sync(device)
                row=dict(step=step+1,observations=args.batch_size*length,answers=args.batch_size*count,
                    loss=float(loss.detach()),accuracy=float((scores.argmax(-1)==targets).float().mean()),
                    gradient_norm=float(norm),elapsed_seconds=time.perf_counter()-start)
                rows.append(row)
                with (args.out/'training.csv').open('a',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=row)
                    if step==0:writer.writeheader()
                    writer.writerow(row)
                if step==0 or (step+1)%100==0:print(json.dumps(row),flush=True)
            meta['checkpoint_sha256']=save_checkpoint(args.out/'checkpoint.pt',model,meta,args.steps)
            torch.save(optimizer.state_dict(),args.out/'optimizer.pt')
            meta.update(training_answers=sum(r['answers'] for r in rows),training_observations=sum(r['observations'] for r in rows))
        elif not args.checkpoint:
            meta['checkpoint_sha256']=save_checkpoint(args.out/'checkpoint.pt',model,meta,0)
        meta['evaluation']=evaluate(model,args,args.out/'evaluation',seen,deadline,datasets=data,pressure_arms=pressure_arms)
        model.assert_codebook_valid()
        for name,digest in sources.items():
            if common.sha256(REPO/name)!=digest or common.sha256(args.out/'sources'/name)!=digest:
                raise AssertionError('Source changed during invocation')
        common.before_deadline(deadline)
        meta['status']='succeeded'
    except BaseException as error:
        meta.update(status='failed',failure=repr(error),traceback=traceback.format_exc())
        if model is not None:torch.save(model.state_dict(),args.out/'partial_state.pt')
        raise
    finally:
        meta.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        if device.type=='cuda':meta['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        common.json_file(args.out/'metadata.json',meta)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('train','evaluate'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--design',type=Path)
    p.add_argument('--phase',choices=('calibration','confirmation','comparison','smoke'),default='calibration')
    p.add_argument('--rank',type=int,choices=(8,16,32),default=32)
    p.add_argument('--codebook-seed',type=int,default=90501)
    p.add_argument('--init-seed',type=int,default=90511)
    p.add_argument('--train-seed',type=int,default=90502)
    p.add_argument('--dev-seed',type=int,default=90503)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--eval-episodes',type=int,default=128)
    p.add_argument('--pressure-episodes',type=int,default=32)
    p.add_argument('--eval-batch-size',type=int,default=16)
    p.add_argument('--suite',default='full',choices=('full',))
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='cpu')
    p.add_argument('--max-seconds',type=int,default=900)
    return p


if __name__=='__main__':run(parser().parse_args())
