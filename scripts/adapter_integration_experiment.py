"""Reference sleep/associative adapter integration and bounded common pilot."""
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
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import numpy as np
import torch
import torch.nn.functional as F
from clmm.pretrain.adapter_registry import new_associative,load_adapter_checkpoint,save_adapter_checkpoint,fork_associative
from clmm.pretrain.kv_control import physical_state_bytes
from clmm.pretrain.kv_streams import training_data,evaluation_data,replay_plan
from clmm.sleep import SleepSession
from scripts import fact_revision_experiment as common
from scripts.kv_representation_control import equal_tree,roll_tree


def datasets(args,seen,phase):
    data=evaluation_data(args.dev_seed,args.eval_episodes,args.pressure_episodes,seen,'full')
    if phase=='initial':
        return {k:v for k,v in data.items() if not k.startswith('pressure') or k in ('pressure_k8_d0','pressure_k8_d64')}
    if phase=='fork':
        return {k:v for k,v in data.items() if not k.startswith('pressure') or k in ('pressure_k8_d0','pressure_k8_d64')}
    return {k:v for k,v in data.items() if not k.startswith('pressure_k32_')}


@torch.no_grad()
def evaluate(model,args,out,seen,deadline,phase='comparison',*,data_override=None,pressure_arms_override=None):
    out.mkdir()
    model.eval()
    device=next(model.buffers()).device
    slow={name:t.clone() for name,t in model.state_dict().items()}
    cpu_rng=torch.random.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    data=datasets(args,seen,phase) if data_override is None else data_override
    common.json_file(out/'episodes.json',data)
    rows,scores,plans,resources=[],[],[],[]
    for cell,episodes in data.items():
        pressure=cell.startswith('pressure')
        arms=(('online',) if phase in ('initial','fork') else ('online','repeat','continuous','periodic')) if pressure else ('online','reset','rolled')
        if pressure and pressure_arms_override is not None:arms=pressure_arms_override
        for arm in arms:
            for start in range(0,len(episodes),args.eval_batch_size):
                common.before_deadline(deadline)
                tick=time.perf_counter()
                batch=episodes[start:start+args.eval_batch_size]
                size=len(batch)
                session=SleepSession(model,size,arm if pressure else 'online',[ep['seed'] for ep in batch])
                emitted=[[] for _ in batch]
                queries=torch.tensor([[q['key'] for q in ep['queries']] for ep in batch],device=device)
                peak=physical_state_bytes(session.state)
                occupancy=torch.zeros(4,device=device,dtype=torch.long)
                per_history_occupancy=torch.zeros(size,4,device=device,dtype=torch.long)
                per_history_decay=torch.zeros_like(per_history_occupancy)
                max_pairs=max_snapshot=0
                boundary_error=midblock_error=0.
                resume_points=[]
                for index in range(len(batch[0]['keys'])):
                    common.before_deadline(deadline)
                    keys=torch.tensor([ep['keys'][index] for ep in batch],device=device)
                    values=torch.tensor([ep['values'][index] for ep in batch],device=device)
                    events=session.prime(keys,values) if index<batch[0]['boundary'] else session.observe(keys,values)
                    for b,items in enumerate(events):emitted[b].extend(items)
                    if hasattr(model,'routing_mask'):
                        for offset in range(len(events[0])):
                            ids=torch.tensor([items[offset]['key'] for items in events],device=device)
                            active=model.routing_mask(ids)
                            occupancy+=active.sum(0)
                            per_history_occupancy+=active
                            if hasattr(model,'decay_mask'):per_history_decay+=model.decay_mask(ids)
                    peak=max(peak,physical_state_bytes(session.state))
                    counts=[p.logical_counts() for p in session.policies]
                    max_pairs=max(max_pairs,max(c['buffer_pairs'] for c in counts))
                    max_snapshot=max(max_snapshot,max(c['snapshot_pairs'] for c in counts))
                    if index in (batch[0]['boundary']-1,batch[0]['boundary']):
                        probe=queries[:,:4] if pressure else queries
                        before=session.read(probe).clone()
                        raw=io.BytesIO()
                        torch.save(session.export_state(),raw)
                        raw.seek(0)
                        payload=torch.load(raw,map_location='cpu',weights_only=True)
                        session=SleepSession.from_state(model,payload)
                        error=float((session.read(probe)-before).abs().max())
                        if error:raise AssertionError('Session serialization changed scores')
                        point='prefill' if index==batch[0]['boundary']-1 else 'midblock'
                        resume_points.append(point)
                        if point=='prefill':boundary_error=error
                        else:midblock_error=error
                        if cell=='pressure_k8_d16' and arm=='periodic' and start==0:
                            torch.save(payload,out/f'example_{point}_session.pt')
                terminal=session.finish()
                for b,items in enumerate(terminal):emitted[b].extend(items)
                expected=[replay_plan(ep,arm if pressure else 'online') for ep in batch]
                assert all(got==want['events'] for got,want in zip(emitted,expected))
                query_state=session.state
                if arm=='reset':query_state=model.initial_state(size)
                elif arm=='rolled':
                    payload=model.export_state(query_state)
                    payload['weights']=roll_tree(payload['weights'])
                    payload['aux']=roll_tree(payload['aux'])
                    query_state=model.import_state(payload)
                before=model.export_state(query_state)
                logits=model.read(queries,query_state)
                assert equal_tree(before,model.export_state(query_state))
                if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite read')
                vectors=logits.cpu().numpy()
                predicted=vectors.argmax(-1)
                scores.append(vectors.reshape(-1,16))
                # Exposure instrumentation is computed after all state updates.
                # Future query keys/labels never enter a write or the router.
                route_table=model.route_table.cpu().tolist() if hasattr(model,'route_table') else None
                for b,ep in enumerate(batch):
                    plans.append(dict(cell=cell,arm=arm,episode=start+b,events=emitted[b],boundary=ep['boundary']))
                    for q,query in enumerate(ep['queries']):
                        target=query['target']
                        exposure={}
                        if route_table is not None:
                            head=route_table[query['key']]
                            last=max(i for i,event in enumerate(emitted[b]) if event['key']==query['key'] and event['kind']=='online')
                            following=emitted[b][last+1:]
                            same=sum(route_table[event['key']]==head for event in following)
                            exposure=dict(query_module=head,after_last_real_target_observation=len(following),
                                following_same_module_writes=same,following_other_module_writes=len(following)-same,
                                target_decay_applications=len(following) if model.config.inactive_decay=='global' else same)
                        rows.append(dict(cell=cell,arm=arm,episode=start+b,rank=q,key=query['key'],category=query['category'],
                            target=target,old=query['old'],prediction=int(predicted[b,q]),correct=int(predicted[b,q]==target),
                            obsolete=int(query['old']>=0 and predicted[b,q]==query['old']),
                            mse=float(np.mean((vectors[b,q]-np.eye(16)[target])**2)),
                            actual_writes=len(emitted[b]),state_write_count=query_state.write_count,
                            state_source_episode=(-1 if arm=='reset' else start+(b-1)%size if arm=='rolled' else start+b),**exposure))
                budget=session.budget()
                saturation=float((query_state.weights['trace'].abs()>=model.config.trace_bound-1e-6).float().mean()) if 'trace' in query_state.weights else None
                common.sync(device)
                resources.append(dict(cell=cell,arm=arm,batch_start=start,batch=size,actual_writes=budget['total_writes'],
                    replay_writes=budget['replay_writes'],peak_fast_bytes=peak,final_fast_bytes=physical_state_bytes(query_state),
                    per_stream_final_bytes=physical_state_bytes(query_state)/size,
                    peak_buffer_pairs=max_pairs,peak_snapshot_pairs=max_snapshot,query_state_write_count=query_state.write_count,
                    boundary_error=boundary_error,midblock_error=midblock_error,resume_points=','.join(resume_points),
                    active_module_updates=int(occupancy.sum()) if hasattr(model,'routing_mask') else None,module_occupancy=json.dumps(occupancy.cpu().tolist()) if hasattr(model,'routing_mask') else None,
                    per_history_module_occupancy=json.dumps(per_history_occupancy.cpu().tolist()) if hasattr(model,'routing_mask') else None,
                    decay_applications=int(per_history_decay.sum()) if hasattr(model,'decay_mask') else None,
                    per_history_decay_applications=json.dumps(per_history_decay.cpu().tolist()) if hasattr(model,'decay_mask') else None,
                    trace_saturation=saturation,elapsed_seconds=time.perf_counter()-tick))
            print(json.dumps(dict(completed=cell,arm=arm)),flush=True)
    summary=[]
    for cell in data:
        for arm in sorted({r['arm'] for r in rows if r['cell']==cell}):
            for category in ('targets','stable','revised','recent'):
                subset=[r for r in rows if r['cell']==cell and r['arm']==arm and
                        (r['category']!='recent' if category=='targets' else r['category']==category)]
                if not subset:continue
                groups={}
                for row in subset:groups.setdefault(row['episode'],[]).append(row)
                summary.append(dict(cell=cell,arm=arm,category=category,queries=len(subset),episodes=len(groups),
                    accuracy=float(np.mean([r['correct'] for r in subset])),
                    all_correct=float(np.mean([all(r['correct'] for r in group) for group in groups.values()])),
                    mse=float(np.mean([r['mse'] for r in subset])),
                    obsolete_rate=float(np.mean([r['obsolete'] for r in subset])) if category=='revised' else None))
    common.write_csv(out/'predictions.csv',rows)
    common.write_csv(out/'summary.csv',summary)
    common.write_csv(out/'resources.csv',resources)
    common.json_file(out/'plans.json',plans)
    np.save(out/'scores.npy',np.concatenate(scores))
    assert all(torch.equal(t,model.state_dict()[name]) for name,t in slow.items())
    assert torch.equal(cpu_rng,torch.random.get_rng_state())
    if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state(device))
    return dict(queries=len(rows),scores_sha256=common.sha256(out/'scores.npy'),
                pure_reads=True,slow_parameters_and_rng_unchanged=True,
                learned=model.learned_metrics() if hasattr(model,'learned_metrics') else None)


def run(args):
    if min(args.steps,args.batch_size,args.eval_episodes,args.pressure_episodes,args.eval_batch_size,args.max_seconds)<1:
        raise ValueError('Positive counts and limits required')
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    device=torch.device(args.device)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    sources=common.capture_sources(out)
    for path in [*sorted((REPO/'clmm').rglob('*.py')),Path(__file__),REPO/'scripts/kv_representation_control.py']:
        name=path.relative_to(REPO).as_posix()
        (out/'sources'/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,out/'sources'/name)
        sources[name]=common.sha256(out/'sources'/name)
    meta=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),source_sha256=sources,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},offline_training=args.mode=='train',
        environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device=str(device),
            device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',deterministic=True,dtype='float32'))
    if args.design:
        shutil.copy2(args.design,out/'design.json')
        meta['design_sha256']=common.sha256(out/'design.json')
    common.json_file(out/'metadata.json',meta)
    started=time.perf_counter()
    deadline=started+args.max_seconds
    model=None
    try:
        arrays,plans,seen=training_data(args.steps,args.batch_size,90502)
        np.savez_compressed(out/'training_data.npz',**arrays)
        common.json_file(out/'training_plan.json',plans)
        meta['training_data_sha256']=common.sha256(out/'training_data.npz')
        if args.mode=='train':
            parent,_=load_adapter_checkpoint(args.shared_checkpoint,torch.device('cpu'))
            torch.manual_seed(args.init_seed)
            model=new_associative(parent,update_rule=args.rule,routing=args.routing).to(device)
            del parent
            meta['shared_checkpoint_sha256']=common.sha256(args.shared_checkpoint)
            meta['shared_transfer']='Literal codebook only; nonlinear slow weights are incompatible and not copied'
        else:
            model,parent=load_adapter_checkpoint(args.checkpoint,device)
            meta['parent_sha256']=common.sha256(args.checkpoint)
            meta['parent_training']=parent.get('training')
            if args.mode=='fork':
                model=fork_associative(model,update_rule=args.rule,routing=args.routing)
                meta['fork']='Same slow tensor values, new rule/access config, zero traces; no adaptation'
        meta.update(method=model.method,config=model.config.to_dict(),parameters=model.parameter_counts(),
                    codebook_hash=model.codebook_hash,deviations=model.config_deviations)
        if args.mode=='train':
            training=dict(step=0,init_seed=args.init_seed,train_seed=90502)
            save_adapter_checkpoint(out/'initial_checkpoint.pt',model,source_commit=meta['source_commit'],source_sha256=sources,training=training)
            meta['initial_checkpoint_sha256']=common.sha256(out/'initial_checkpoint.pt')
            meta['initial_evaluation']=evaluate(model,args,out/'initial-evaluation',seen,deadline,'initial')
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
            trace=[]
            for step in range(args.steps):
                common.before_deadline(deadline)
                length,count=int(arrays['lengths'][step]),int(arrays['counts'][step])
                state=model.initial_state(args.batch_size)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for index in range(length):
                    state=model.write(torch.tensor(arrays['keys'][step,:,index].astype(np.int64),device=device),
                                      torch.tensor(arrays['values'][step,:,index].astype(np.int64),device=device),state)
                keys=torch.tensor(arrays['queries'][step,:,:count].astype(np.int64),device=device)
                targets=torch.tensor(arrays['targets'][step,:,:count].astype(np.int64),device=device)
                scores=model.read(keys,state)
                loss=F.mse_loss(scores,F.one_hot(targets,16).float())
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite outer objective')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                common.sync(device)
                row=dict(step=step+1,answers=args.batch_size*count,observations=args.batch_size*length,
                         loss=float(loss.detach()),gradient_norm=float(norm),eta=float(model.eta.detach()),
                         elapsed_seconds=time.perf_counter()-started)
                trace.append(row)
                with (out/'training.csv').open('a',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=row)
                    if step==0:writer.writeheader()
                    writer.writerow(row)
                if step==0 or (step+1)%100==0:print(json.dumps(row),flush=True)
            training.update(step=args.steps,answers=sum(r['answers'] for r in trace),observations=sum(r['observations'] for r in trace))
            save_adapter_checkpoint(out/'checkpoint.pt',model,source_commit=meta['source_commit'],source_sha256=sources,training=training)
            torch.save(optimizer.state_dict(),out/'optimizer.pt')
            meta.update(training=training,checkpoint_sha256=common.sha256(out/'checkpoint.pt'))
        meta['evaluation']=evaluate(model,args,out/'evaluation',seen,deadline,'fork' if args.mode=='fork' else 'comparison')
        for name,digest in sources.items():
            if common.sha256(REPO/name)!=digest or common.sha256(out/'sources'/name)!=digest:raise AssertionError('Source changed')
        common.before_deadline(deadline)
        meta['status']='succeeded'
    except BaseException as error:
        meta.update(status='failed',failure=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        meta.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started)
        if device.type=='cuda':meta['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        common.json_file(out/'metadata.json',meta)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('train','evaluate','fork'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--shared-checkpoint',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--design',type=Path)
    p.add_argument('--rule',choices=('gradient','hebbian'),default='hebbian')
    p.add_argument('--routing',choices=('dense','top1'),default='dense')
    p.add_argument('--init-seed',type=int,default=90511)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--dev-seed',type=int,default=90623)
    p.add_argument('--eval-episodes',type=int,default=128)
    p.add_argument('--pressure-episodes',type=int,default=64)
    p.add_argument('--eval-batch-size',type=int,default=16)
    p.add_argument('--device',default='cpu')
    p.add_argument('--max-seconds',type=int,default=900)
    return p

if __name__=='__main__':run(parser().parse_args())
