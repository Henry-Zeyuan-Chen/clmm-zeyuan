"""Fixed-set fitting and novel-assignment probes for the shared byte model."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import fact_revision_experiment as common
import numpy as np
import torch
import torch.nn.functional as F
from clmm.pretrain import build_model,load_checkpoint,save_checkpoint,reset_memory_at_segments
from clmm.pretrain.fact_streams import Episode,encode_example,render_query
from clmm.pretrain.association_probe import build_bank,fresh_same,fresh_heldout,fresh_training


def training_batch(bank,step,batch_size,seed,regime):
    count=batch_size//2
    if batch_size<2 or batch_size%2 or len(bank['episodes'])%count:
        raise ValueError('Even batch required; context count must divide the bank size')
    if regime=='fixed':
        batches=len(bank['episodes'])//count
        rng=np.random.default_rng(np.random.SeedSequence([seed,step//batches]))
        order=rng.permutation(len(bank['episodes']))
        indices=order[(step%batches)*count:(step%batches+1)*count].tolist()
        episodes=[Episode.from_dict(bank['episodes'][i]) for i in indices]
        plan=dict(step=step+1,bank_indices=indices)
    elif regime=='fresh':
        episodes=fresh_training(bank,seed,step,count)
        plan=dict(step=step+1,episodes=[ep.to_dict() for ep in episodes])
    else:raise ValueError('regime must be fixed or fresh')
    samples=[encode_example(ep,q,'online') for ep in episodes for q in ep.queries]
    x=np.stack([np.frombuffer(s[0],dtype=np.uint8) for s in samples])
    y=np.array([s[1] for s in samples],dtype=np.uint8)
    return x,y,plan


def evaluation_groups(bank,args):
    train=[Episode.from_dict(ep) for ep in bank['episodes']]
    return {('train' if args.regime=='fixed' else 'reference_bank'):train,
        'novel_values':fresh_same(bank,args.dev_seed,args.eval_episodes),
        'heldout_keys':fresh_heldout(args.dev_seed+1,args.eval_episodes),
        'heldout_keys_templates':fresh_heldout(args.dev_seed+1,args.eval_episodes,'heldout'),
        'train_query_only':train}


@torch.no_grad()
def evaluate(model,bank,args,out,device,deadline):
    out.mkdir()
    model.eval()
    cpu_rng=torch.random.get_rng_state().clone()
    gpu_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    rows,logits_all,semantics=[],[],[]
    for group,episodes in evaluation_groups(bank,args).items():
        semantics.append(dict(group=group,episodes=[ep.to_dict() for ep in episodes]))
        for condition in ('online','reset_memory'):
            samples=[]
            for i,ep in enumerate(episodes):
                for rank,q in enumerate(ep.queries):
                    prefix=render_query(q) if group=='train_query_only' else encode_example(ep,q,'online')[0]
                    samples.append(dict(group=group,episode=i,
                        base_pair=i%len(bank['episodes']) if group=='novel_values' else i,
                        rank=rank,condition=condition,key=q.key,target=q.answer_byte,prefix=prefix.decode('ascii')))
            context=reset_memory_at_segments(model) if condition=='reset_memory' else nullcontext()
            with context:
                for start in range(0,len(samples),args.eval_batch_size):
                    common.before_deadline(deadline)
                    chunk=samples[start:start+args.eval_batch_size]
                    x=torch.tensor([list(s['prefix'].encode('ascii')) for s in chunk],device=device)
                    y=torch.tensor([s['target'] for s in chunk],device=device)
                    logits=model(x)[:,-1]
                    scores={k:v.cpu().tolist() for k,v in common.score_logits(logits,y).items()}
                    logits_all.append(logits.cpu().numpy())
                    for j,row in enumerate(chunk):
                        row.update({k:v[j] for k,v in scores.items()})
                        row['correct']=int(row['prediction']==row['target'])
                        rows.append(row)
    if not all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()):
        raise AssertionError('Evaluation changed outer checkpoint tensors')
    if not torch.equal(cpu_rng,torch.random.get_rng_state()):
        raise AssertionError('Evaluation consumed CPU training RNG')
    if gpu_rng is not None and not torch.equal(gpu_rng,torch.cuda.get_rng_state(device)):
        raise AssertionError('Evaluation consumed CUDA training RNG')
    np.save(out/'answer_logits.npy',np.concatenate(logits_all))
    common.write_csv(out/'predictions.csv',rows)
    with gzip.open(out/'episodes.json.gz','wt',encoding='utf8') as f:json.dump(semantics,f)
    summaries=[]
    for group in evaluation_groups(bank,args):
        for condition in ('online','reset_memory'):
            subset=[r for r in rows if r['group']==group and r['condition']==condition]
            pairs=[subset[i:i+2] for i in range(0,len(subset),2)]
            summaries.append(dict(group=group,condition=condition,episodes=len(pairs),queries=len(subset),
                accuracy=float(np.mean([r['correct'] for r in subset])),
                both_correct=float(np.mean([all(r['correct'] for r in p) for p in pairs])),
                first_accuracy=float(np.mean([p[0]['correct'] for p in pairs])),
                second_accuracy=float(np.mean([p[1]['correct'] for p in pairs])),
                prediction_changes=float(np.mean([p[0]['prediction']!=p[1]['prediction'] for p in pairs])),
                nll=float(np.mean([r['nll'] for r in subset])),
                legal_probability_mass=float(np.mean([r['legal_probability_mass'] for r in subset]))))
    common.json_file(out/'summary.json',summaries)
    return summaries


def run(args):
    if (args.steps<10 or args.steps%10 or args.batch_size<2 or args.batch_size%2
            or min(args.max_seconds,args.eval_episodes,args.eval_batch_size,args.threads)<1
            or not np.isfinite(args.lr) or args.lr<=0):
        raise ValueError('Positive limits/lr, even batch and steps multiple of10 required')
    out,device,meta=common.initialize(args)
    name=Path(__file__).relative_to(REPO).as_posix()
    (out/'sources'/name).parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,out/'sources'/name)
    meta['source_sha256'][name]=common.sha256(out/'sources'/name)
    start=time.perf_counter();deadline=start+args.max_seconds
    rows,plans,inputs,targets=[],[],[],[]
    model=None
    try:
        bank=build_bank(args.bank_seed,args.contexts,args.key_count)
        common.json_file(out/'bank.json',bank)
        model,config,payload=load_checkpoint(args.checkpoint,map_location='cpu')
        seed=payload['training_info'].get('init_seed')
        if (seed not in (68101,68105) or payload['training_info'].get('task') is not None
                or payload['training_info'].get('parent_checkpoint_sha256') is not None
                or (config.segment_len,config.memory_segment_len,config.memory_batch_size)!=(32,8,32)):
            raise ValueError('Require original shared language parent, geometry32/8/32')
        if config.segment_len!=args.segment_len:
            config=replace(config,segment_len=args.segment_len)
            with torch.random.fork_rng(devices=[]):changed=build_model(config)
            changed.load_state_dict(model.state_dict(),strict=True);model=changed
        model.to(device)
        meta.update(parent_init_seed=seed,parameter_count=sum(p.numel() for p in model.parameters()))
        torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},out/'initial_state.pt')
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
        checkpoints=[]
        def checkpoint(step):
            common.before_deadline(deadline)
            folder=out/f'step-{step:04d}';folder.mkdir()
            info=dict(task='fixed-set-generalization',parent_init_seed=seed,
                parent_checkpoint_sha256=meta['checkpoint_sha256'],steps=step,regime=args.regime,
                bank_seed=args.bank_seed,contexts=args.contexts,key_count=args.key_count,
                train_seed=args.train_seed,lr=args.lr,fixed_final_checkpoint=step==args.steps)
            digest=save_checkpoint(folder/'checkpoint.pt',model,config,
                source_info=dict(repo_commit=meta['repo_commit'],source_sha256=meta['source_sha256']),training_info=info)
            torch.save(optimizer.state_dict(),folder/'optimizer.pt')
            summary=evaluate(model,bank,args,folder/'evaluation',device,deadline)
            common.json_file(folder/'metadata.json',dict(step=step,checkpoint_sha256=digest,
                selection='Fixed stage endpoint; final step primary',summary=summary))
            checkpoints.append(dict(step=step,checkpoint_sha256=digest,summary=summary))
            meta['checkpoints']=checkpoints;common.json_file(out/'metadata.json',meta)
            print(json.dumps(dict(checkpoint=step,summary=summary)),flush=True)
        checkpoint(0)
        stages={args.steps//10,args.steps//2,args.steps}
        for step in range(args.steps):
            common.before_deadline(deadline)
            x_np,y_np,plan=training_batch(bank,step,args.batch_size,args.train_seed,args.regime)
            x,y=torch.tensor(x_np.astype(np.int64),device=device),torch.tensor(y_np.astype(np.int64),device=device)
            model.train();optimizer.zero_grad(set_to_none=True)
            logits=model(x)[:,-1];loss=F.cross_entropy(logits,y)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():raise RuntimeError('Nonfinite training loss/logits')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();common.sync(device)
            row=dict(step=step+1,loss=float(loss.detach()),accuracy=float((logits.argmax(-1)==y).float().mean()),
                gradient_norm=float(norm),supervised_answers=len(y),input_bytes=x.numel(),elapsed_seconds=time.perf_counter()-start)
            rows.append(row);plans.append(plan);inputs.append(x_np);targets.append(y_np)
            with (out/'training.csv').open('a',newline='',encoding='utf8') as f:
                import csv
                writer=csv.DictWriter(f,fieldnames=row)
                if step==0:writer.writeheader()
                writer.writerow(row)
            if step==0 or (step+1)%100==0:print(json.dumps(row),flush=True)
            if step+1 in stages:checkpoint(step+1)
        meta.update(status='succeeded',completed_steps=len(rows),supervised_answers=len(rows)*args.batch_size,
                    input_bytes=sum(r['input_bytes'] for r in rows))
    except BaseException as error:
        meta.update(status='failed',completed_steps=len(rows),failure=repr(error),traceback=traceback.format_exc())
        if model is not None:torch.save(model.state_dict(),out/'partial_state.pt')
        raise
    finally:
        if inputs:
            np.savez_compressed(out/'training_inputs.npz',input_bytes=np.stack(inputs),targets=np.stack(targets))
            common.json_file(out/'training_plan.json',plans)
        changed=[]
        for name,digest in meta['source_sha256'].items():
            if common.sha256(REPO/name)!=digest or common.sha256(out/'sources'/name)!=digest:changed.append(name)
        if changed:meta.update(status='failed',source_changed=changed)
        meta.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        if device.type=='cuda':meta['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        common.json_file(out/'metadata.json',meta)
        if changed:raise AssertionError('Sources changed during execution')


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.set_defaults(mode='fixed_set_generalization')
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--design',type=Path);p.add_argument('--regime',choices=('fixed','fresh'),default='fixed')
    p.add_argument('--segment-len',type=int,choices=(32,128),default=32)
    p.add_argument('--contexts',type=int,default=32);p.add_argument('--key-count',type=int,default=16)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--bank-seed',type=int,default=90201);p.add_argument('--train-seed',type=int,default=90202)
    p.add_argument('--dev-seed',type=int,default=90203);p.add_argument('--eval-episodes',type=int,default=128)
    p.add_argument('--eval-batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-seconds',type=int,default=600)
    return p

if __name__=='__main__':run(parser().parse_args())
