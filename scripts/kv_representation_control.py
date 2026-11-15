"""Shared stateful key/value positive control; not a complete Titans model."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import fact_revision_experiment as common
import numpy as np
import torch
import torch.nn.functional as F
from clmm.pretrain.kv_control import KVConfig,DeltaAdapter,NeuralAdapter,physical_state_bytes
from clmm.pretrain.kv_streams import training_data,evaluation_data,replay_plan


def save_checkpoint(path,model,meta,step):
    payload=dict(format='clmm.kv.control.v1',method=model.method,config=model.config.to_dict(),
        model_state=model.state_dict(),source_commit=meta['source_commit'],source_sha256=meta['source_sha256'],
        training=dict(step=step,init_seed=meta['args']['init_seed'],train_seed=meta['args']['train_seed'],
            objective='mean squared read-coordinate error versus one-hot value',outer_lr=meta['args']['lr']))
    torch.save(payload,path);return common.sha256(path)


def load_checkpoint(path,device):
    p=torch.load(path,map_location='cpu',weights_only=True)
    if p.get('format')!='clmm.kv.control.v1' or p.get('method') not in ('delta','neural'):raise ValueError('Not a KV-control checkpoint')
    model=(DeltaAdapter if p['method']=='delta' else NeuralAdapter)(KVConfig(**p['config'])).to(device)
    model.load_state_dict(p['model_state'],strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite checkpoint')
    return model,p


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,tuple):return isinstance(b,tuple) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def roll_tree(x):
    if isinstance(x,torch.Tensor):return x.roll(1,dims=0)
    if isinstance(x,dict):return {k:roll_tree(v) for k,v in x.items()}
    if isinstance(x,tuple):return tuple(roll_tree(v) for v in x)
    return x


@torch.no_grad()
def evaluate(model,args,out,seen,deadline):
    out.mkdir();model.eval();device=next(model.buffers()).device
    cpu_rng=torch.random.get_rng_state().clone()
    gpu_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    datasets=evaluation_data(args.dev_seed,args.eval_episodes,args.pressure_episodes,seen,args.suite)
    common.json_file(out/'episodes.json',datasets)
    rows,logits_saved,resources,plans_saved=[],[],[],[]
    for cell,episodes in datasets.items():
        pressure=cell.startswith('pressure')
        arms=('online','repeat','continuous','periodic','boundary_reset') if pressure else ('online','reset','rolled')
        for arm in arms:
            for start in range(0,len(episodes),args.eval_batch_size):
                batch_started=time.perf_counter()
                common.before_deadline(deadline)
                batch=episodes[start:start+args.eval_batch_size];count=len(batch)
                plans=[replay_plan(ep,arm if arm in ('repeat','continuous','periodic') else 'online') for ep in batch]
                if len({p['writes'] for p in plans})!=1:raise AssertionError('Nonrectangular plans')
                state=model.initial_state(count);peak=physical_state_bytes(state)
                queries=torch.tensor([[q['key'] for q in ep['queries']] for ep in batch],device=device)
                boundary_before=boundary_after=0;roundtrip_delta=0.
                for index in range(plans[0]['writes']):
                    common.before_deadline(deadline)
                    keys=torch.tensor([p['events'][index]['key'] for p in plans],device=device)
                    values=torch.tensor([p['events'][index]['value'] for p in plans],device=device)
                    state=model.write(keys,values,state)
                    peak=max(peak,physical_state_bytes(state))
                    if index+1==batch[0]['boundary']:
                        boundary_before=physical_state_bytes(state)
                        probe_queries=queries[:,:4] if pressure else queries
                        before=model.read(probe_queries,state)
                        raw=io.BytesIO();torch.save(model.export_state(state),raw);raw.seek(0)
                        payload=torch.load(raw,map_location='cpu',weights_only=True)
                        state=model.import_state(payload);boundary_after=physical_state_bytes(state)
                        roundtrip_delta=float((model.read(probe_queries,state)-before).abs().max())
                        if roundtrip_delta!=0:raise AssertionError('State roundtrip changed logits')
                        if arm=='boundary_reset':state=model.initial_state(count)
                        if cell=='pressure_k8_d16' and arm=='online' and start==0:
                            torch.save(payload,out/'example_boundary_state.pt')
                if arm=='reset':state=model.initial_state(count)
                elif arm=='rolled':
                    payload=model.export_state(state)
                    payload['weights']=roll_tree(payload['weights']);payload['aux']=roll_tree(payload['aux'])
                    state=model.import_state(payload)
                snapshot=model.export_state(state)
                logits=model.read(queries,state)
                if not equal_tree(snapshot,model.export_state(state)):raise AssertionError('Read mutated fast state')
                if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite logits')
                predicted=logits.argmax(-1).cpu().numpy();raw_logits=logits.cpu().numpy()
                logits_saved.append(raw_logits.reshape(-1,model.config.nvalues))
                for b,ep in enumerate(batch):
                    plans_saved.append(dict(cell=cell,arm=arm,episode=start+b,plan=plans[b]))
                    source_episode=start+(b-1)%count if arm=='rolled' else start+b
                    for rank,q in enumerate(ep['queries']):
                        vector=raw_logits[b,rank];target=q['target']
                        rows.append(dict(cell=cell,arm=arm,episode=start+b,rank=rank,key=q['key'],
                            category=q['category'],target=target,old=q['old'],prediction=int(predicted[b,rank]),
                            correct=int(predicted[b,rank]==target),obsolete=int(q['old']>=0 and predicted[b,rank]==q['old']),
                            mse=float(np.mean((vector-np.eye(model.config.nvalues)[target])**2)),
                            actual_writes=plans[b]['writes'],state_write_count=state.write_count,
                            state_source_episode=source_episode if arm!='reset' else -1))
                resources.append(dict(cell=cell,arm=arm,batch_start=start,batch=count,
                    boundary_before_bytes=boundary_before,boundary_restored_bytes=boundary_after,
                    final_bytes=physical_state_bytes(state),peak_post_write_bytes=peak,
                    per_episode_final_bytes=physical_state_bytes(state)//count,
                    roundtrip_max_abs_delta=roundtrip_delta,
                    logical_buffer_pairs=8 if pressure and arm in ('continuous','periodic') else 0,
                    logical_snapshot_pairs=4 if pressure and arm in ('continuous','periodic') else 0,
                    elapsed_seconds=time.perf_counter()-batch_started))
                if cell=='pressure_k8_d16' and arm=='online' and start==0:torch.save(snapshot,out/'example_final_state.pt')
                common.write_csv(out/'predictions.csv',rows)
            print(json.dumps(dict(completed=cell,arm=arm)),flush=True)
    common.write_csv(out/'resources.csv',resources);common.json_file(out/'plans.json',plans_saved)
    np.save(out/'scores.npy',np.concatenate(logits_saved))
    summary=[]
    for cell in datasets:
        for arm in sorted({r['arm'] for r in rows if r['cell']==cell}):
            subset=[r for r in rows if r['cell']==cell and r['arm']==arm]
            for category in ('all','stable','revised','recent'):
                group=[r for r in subset if category=='all' or r['category']==category]
                if not group:continue
                by_episode={}
                for r in group:by_episode.setdefault(r['episode'],[]).append(r)
                summary.append(dict(cell=cell,arm=arm,category=category,queries=len(group),episodes=len(by_episode),
                    accuracy=float(np.mean([r['correct'] for r in group])),
                    all_correct=float(np.mean([all(r['correct'] for r in e) for e in by_episode.values()])),
                    obsolete_rate=float(np.mean([r['obsolete'] for r in group])) if category=='revised' else None,
                    mse=float(np.mean([r['mse'] for r in group]))))
    common.write_csv(out/'summary.csv',summary)
    if not all(torch.equal(v,model.state_dict()[k]) for k,v in original.items()):raise AssertionError('Evaluation changed slow parameters/buffers')
    if not torch.equal(cpu_rng,torch.random.get_rng_state()):raise AssertionError('Evaluation consumed CPU RNG')
    if gpu_rng is not None and not torch.equal(gpu_rng,torch.cuda.get_rng_state(device)):raise AssertionError('Evaluation consumed CUDA RNG')
    return dict(queries=len(rows),summary=summary,scores_sha256=common.sha256(out/'scores.npy'),
        predictions_sha256=common.sha256(out/'predictions.csv'),outer_parameters_unchanged=True)


def run(args):
    if min(args.steps,args.batch_size,args.eval_episodes,args.pressure_episodes,args.eval_batch_size,args.threads,args.max_seconds)<1:
        raise ValueError('Positive limits required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    device=torch.device(args.device);torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.manual_seed(args.init_seed)
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources=common.capture_sources(out)
    name=Path(__file__).relative_to(REPO).as_posix();(out/'sources'/name).parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,out/'sources'/name);sources[name]=common.sha256(out/'sources'/name)
    meta=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),source_sha256=sources,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device=str(device),
            device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',dtype='float32',deterministic=True))
    if args.design:shutil.copy2(args.design,out/'design.json');meta['design_sha256']=common.sha256(out/'design.json')
    common.json_file(out/'metadata.json',meta);start=time.perf_counter();deadline=start+args.max_seconds
    model=None
    try:
        data,plan,seen=training_data(args.steps,args.batch_size,args.train_seed)
        common.json_file(out/'training_plan.json',plan)
        np.savez_compressed(out/'training_data.npz',**data)
        meta['training_data_sha256']=common.sha256(out/'training_data.npz')
        if args.checkpoint:
            model,parent=load_checkpoint(args.checkpoint,device);meta['parent_sha256']=common.sha256(args.checkpoint)
        else:model=(DeltaAdapter if args.method=='delta' else NeuralAdapter)(KVConfig(step_size=args.step_size)).to(device)
        meta.update(method=model.method,config=model.config.to_dict(),parameters=model.parameter_counts(),deviations=model.config_deviations)
        if args.mode=='train':
            if model.method!='neural':raise ValueError('Delta has no offline training')
            meta['initial_checkpoint_sha256']=save_checkpoint(out/'initial_checkpoint.pt',model,meta,0)
            meta['initial_evaluation']=evaluate(model,args,out/'initial-evaluation',seen,deadline)
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=0.)
            rows=[]
            for step in range(args.steps):
                common.before_deadline(deadline)
                length,count=int(data['lengths'][step]),int(data['counts'][step])
                state=model.initial_state(args.batch_size);model.train();optimizer.zero_grad(set_to_none=True)
                for j in range(length):
                    state=model.write(torch.tensor(data['keys'][step,:,j].astype(np.int64),device=device),
                        torch.tensor(data['values'][step,:,j].astype(np.int64),device=device),state)
                query=torch.tensor(data['queries'][step,:,:count].astype(np.int64),device=device)
                target=torch.tensor(data['targets'][step,:,:count].astype(np.int64),device=device)
                scores=model.read(query,state);loss=F.mse_loss(scores,F.one_hot(target,16).float())
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite outer loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step();common.sync(device)
                row=dict(step=step+1,observations=args.batch_size*length,answers=args.batch_size*count,
                    loss=float(loss.detach()),accuracy=float((scores.argmax(-1)==target).float().mean()),
                    gradient_norm=float(norm),elapsed_seconds=time.perf_counter()-start)
                rows.append(row)
                with (out/'training.csv').open('a',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=row)
                    if step==0:writer.writeheader()
                    writer.writerow(row)
                if step==0 or (step+1)%100==0:print(json.dumps(row),flush=True)
            meta['checkpoint_sha256']=save_checkpoint(out/'checkpoint.pt',model,meta,args.steps)
            torch.save(optimizer.state_dict(),out/'optimizer.pt')
            meta.update(training_answers=sum(r['answers'] for r in rows),training_observations=sum(r['observations'] for r in rows))
        elif not args.checkpoint:
            meta['checkpoint_sha256']=save_checkpoint(out/'checkpoint.pt',model,meta,0)
        meta['evaluation']=evaluate(model,args,out/'evaluation',seen,deadline)
        for path,digest in sources.items():
            if common.sha256(REPO/path)!=digest or common.sha256(out/'sources'/path)!=digest:raise AssertionError('Source changed')
        common.before_deadline(deadline);meta['status']='succeeded'
    except BaseException as error:
        meta.update(status='failed',failure=repr(error),traceback=traceback.format_exc())
        if model is not None:torch.save(model.state_dict(),out/'partial_state.pt')
        raise
    finally:
        meta.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        if device.type=='cuda':meta['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        common.json_file(out/'metadata.json',meta)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('train','evaluate'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--design',type=Path)
    p.add_argument('--method',choices=('delta','neural'),default='neural');p.add_argument('--step-size',type=float,default=.5)
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--init-seed',type=int,default=90301);p.add_argument('--train-seed',type=int,default=90302)
    p.add_argument('--dev-seed',type=int,default=90303);p.add_argument('--eval-episodes',type=int,default=128)
    p.add_argument('--pressure-episodes',type=int,default=32);p.add_argument('--eval-batch-size',type=int,default=16)
    p.add_argument('--suite',choices=('gate','full'),default='gate');p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--device',default='cpu');p.add_argument('--max-seconds',type=int,default=900)
    return p

if __name__=='__main__':run(parser().parse_args())
