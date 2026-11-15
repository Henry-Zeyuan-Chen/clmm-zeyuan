"""Bounded full-byte-sequence replay integration at unchanged 839248 parameters."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,os,shutil,subprocess,sys,time,traceback
from contextlib import nullcontext
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
import torch
import torch.nn.functional as F
from clmm.pretrain import load_checkpoint,build_model,save_checkpoint,reset_memory_at_segments
from clmm.pretrain.fact_streams import Episode,render_query
from clmm.pretrain.sequence_freshness_data import (ARMS,collect_histories,signature,training_worlds,
    training_arrays,evaluation_worlds,examples,replay_context)
from scripts import fact_revision_experiment as common
from scripts.freshness_experiment import capture

def state_sha(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def setup(args):
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.manual_seed(91302)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def prepare(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    worlds=training_worlds(args.steps,args.contexts,args.train_seed)
    arrays=training_arrays(worlds);np.savez_compressed(out/'training.npz',**arrays)
    with gzip.open(out/'training_plan.json.gz','wt',encoding='utf8') as f:json.dump(worlds,f)
    seen=collect_histories(worlds);manifest=[];digests=set()
    # The rg scan is structural; gzip episode/training records are also inspected.
    scan=subprocess.run(['rg','-l','-U',r'"key"\s*:\s*"[0-9]{3}"',str(args.history_root),'-g','*.json'],capture_output=True,text=True)
    if scan.returncode not in (0,1):raise RuntimeError(scan.stderr)
    candidates={Path(s) for s in scan.stdout.splitlines()}
    candidates.update(args.history_root.rglob('*.json.gz'))
    for p in sorted(candidates):
        if p.resolve()==(out/'training_plan.json.gz').resolve():continue
        digest=common.sha256(p)
        if digest in digests:continue
        digests.add(digest)
        with (gzip.open(p,'rt',encoding='utf8') if p.suffix=='.gz' else p.open(encoding='utf8')) as f:ids=collect_histories(json.load(f))
        if ids:
            seen.update(ids);manifest.append(dict(path=str(p),sha256=digest,histories=len(ids)))
    common.json_file(out/'excluded_signatures.json',sorted(seen))
    common.json_file(out/'exclusion_manifest.json',dict(inputs=manifest,union_count=len(seen),current_training_unique=len(collect_histories(worlds))))
    data=evaluation_worlds(seen,args.dev_seed,args.episodes,args.revision_episodes)
    common.json_file(out/'episodes.json',data)
    plans=[]
    for group,eps in data.items():
        for i,row in enumerate(eps):
            ep=Episode.from_dict(row)
            for arm in (ARMS if group.startswith('revision') else ('online',)):
                raw,plan=replay_context(ep,arm);plans.append(dict(group=group,episode=i,**plan))
    common.json_file(out/'plans.json',plans)
    common.json_file(out/'samples.json',make_samples(data))
    (out/'parents').mkdir();parents=[]
    for seed in (68101,68105):
        origin=args.parents/f'lm-{seed}/checkpoint.pt';target=out/'parents'/f'{seed}.pt';shutil.copy2(origin,target)
        model,config,payload=load_checkpoint(target,map_location='cpu')
        assert sum(p.numel() for p in model.parameters())==839248
        assert payload['training_info']['init_seed']==seed and payload['training_info'].get('task') is None
        assert (config.segment_len,config.memory_segment_len,config.memory_batch_size)==(32,8,32)
        parents.append(dict(seed=seed,path=target.relative_to(out).as_posix(),sha256=common.sha256(target),state_sha256=state_sha(model),config=asdict(config),original_training=payload['training_info']))
    common.json_file(out/'parents.json',parents)
    sources=capture(out)
    if args.design:shutil.copy2(args.design,out/'design.json')
    meta=dict(status='succeeded',mode='prepare',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),source_sha256=sources,
        train_seed=args.train_seed,development_seed=args.dev_seed,steps=args.steps,contexts=args.contexts,episodes=args.episodes,revision_episodes=args.revision_episodes,
        files_sha256={n:common.sha256(out/n) for n in ('training.npz','training_plan.json.gz','episodes.json','plans.json','samples.json','parents.json','excluded_signatures.json','exclusion_manifest.json')},
        training_answers=args.steps*args.contexts*2,training_input_bytes=int(np.sum(arrays['lengths']))*args.contexts*2,
        command=[sys.executable,*sys.argv],environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__),elapsed_seconds=time.perf_counter()-start)
    common.json_file(out/'metadata.json',meta);print(json.dumps({k:v for k,v in meta.items() if k in ('status','steps','episodes','training_answers','training_input_bytes','elapsed_seconds')}),flush=True)

def verify_packet(packet):
    meta=json.loads((packet/'metadata.json').read_text(encoding='utf8'))
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==meta['source_commit']
    for n,digest in meta['source_sha256'].items():assert common.sha256(REPO/n)==digest,n
    for n,digest in meta['files_sha256'].items():assert common.sha256(packet/n)==digest,n
    return meta

def parent_model(args):
    row=next(r for r in json.loads((args.packet/'parents.json').read_text()) if r['seed']==args.parent)
    assert common.sha256(args.packet/row['path'])==row['sha256']
    model,config,payload=load_checkpoint(args.packet/row['path'],map_location='cpu')
    assert state_sha(model)==row['state_sha256']
    if args.segment!=config.segment_len:
        config=replace(config,segment_len=args.segment)
        with torch.random.fork_rng(devices=[]):changed=build_model(config)
        changed.load_state_dict(model.state_dict(),strict=True);model=changed
    assert state_sha(model)==row['state_sha256'] and sum(p.numel() for p in model.parameters())==839248
    return model.to(args.device),config,row

def make_samples(data):
    rows=[]
    for group,episodes in data.items():
        arms=ARMS if group.startswith('revision') else ('online',)
        for arm in arms:
            conditions=('normal','reset_memory') if not group.startswith('revision') or arm in ('online','periodic_bounded') else ('normal',)
            for condition in conditions:
                for i,raw in enumerate(episodes):
                    ep=Episode.from_dict(raw);samples,plan=examples(ep,arm)
                    for rank,(x,target) in enumerate(samples):
                        rows.append(dict(group=group,arm=arm,condition=condition,episode=i,rank=rank,key=ep.queries[rank].key,target=target,
                            old=ord(ep.queries[rank].old_value) if ep.queries[rank].old_value else -1,prefix_hex=x.hex(),input_bytes=len(x),
                            real_records=plan['real_records'],extra_records=plan['extra_records'],stale_extras=plan['stale_extras']))
    for i,raw in enumerate(data['acquire_train_seen']):
        ep=Episode.from_dict(raw)
        for rank,q in enumerate(ep.queries):
            rows.append(dict(group='query_only_train_seen',arm='online',condition='normal',episode=i,rank=rank,key=q.key,target=q.answer_byte,old=-1,
                             prefix_hex=render_query(q).hex(),input_bytes=31,real_records=0,extra_records=0,stale_extras=0))
    return rows

@torch.no_grad()
def evaluate(model,args,out,deadline):
    out.mkdir();tick=time.perf_counter();model.eval();before=state_sha(model)
    cpu=torch.random.get_rng_state().clone();cuda=torch.cuda.get_rng_state(args.device).clone() if str(args.device).startswith('cuda') else None
    rows=json.loads((args.packet/'samples.json').read_text());logits=np.empty((len(rows),256),dtype=np.float32);resources=[]
    buckets={}
    for i,row in enumerate(rows):buckets.setdefault((row['condition'],row['input_bytes']),[]).append(i)
    for (condition,length),indices in buckets.items():
        with (reset_memory_at_segments(model) if condition=='reset_memory' else nullcontext()):
            for start in range(0,len(indices),args.eval_batch):
                common.before_deadline(deadline);ix=indices[start:start+args.eval_batch];t=time.perf_counter()
                x=torch.tensor([list(bytes.fromhex(rows[i]['prefix_hex'])) for i in ix],device=args.device)
                y=torch.tensor([rows[i]['target'] for i in ix],device=args.device)
                result=model(x)[:,-1];assert torch.isfinite(result).all()
                logits[ix]=result.cpu().numpy();scores={k:v.cpu().tolist() for k,v in common.score_logits(result,y).items()}
                for j,i in enumerate(ix):rows[i].update(score_index=i,**{k:v[j] for k,v in scores.items()},correct=int(scores['prediction'][j]==rows[i]['target']))
                common.sync(torch.device(args.device));resources.append(dict(condition=condition,input_length=length,row_indices=json.dumps(ix),elapsed_seconds=time.perf_counter()-t))
    assert before==state_sha(model) and torch.equal(cpu,torch.random.get_rng_state())
    if cuda is not None:assert torch.equal(cuda,torch.cuda.get_rng_state(args.device))
    np.save(out/'answer_logits.npy',logits);common.write_csv(out/'predictions.csv',rows);common.write_csv(out/'resources.csv',resources)
    summaries=[]
    for group,arm,condition in dict.fromkeys((r['group'],r['arm'],r['condition']) for r in rows):
        subset=[r for r in rows if (r['group'],r['arm'],r['condition'])==(group,arm,condition)];pairs={}
        for r in subset:pairs.setdefault(r['episode'],[]).append(r)
        assert all(len(p)==2 for p in pairs.values())
        summaries.append(dict(group=group,arm=arm,condition=condition,episodes=len(pairs),answers=len(subset),accuracy=float(np.mean([r['correct'] for r in subset])),
            both_correct=float(np.mean([all(r['correct'] for r in p) for p in pairs.values()])),
            prediction_changes=float(np.mean([p[0]['prediction']!=p[1]['prediction'] for p in pairs.values()])),
            nll=float(np.mean([r['nll'] for r in subset])),legal_probability_mass=float(np.mean([r['legal_probability_mass'] for r in subset]))))
    common.write_csv(out/'summary.csv',summaries)
    return dict(status='succeeded',queries=len(rows),elapsed_seconds=time.perf_counter()-tick,pure_checkpoint=True,RNG_unchanged=True,logits_sha256=common.sha256(out/'answer_logits.npy'))

def execute(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();deadline=start+args.max_seconds
    packet=verify_packet(args.packet);model,config,parent=parent_model(args)
    meta=dict(status='running',mode=args.mode,parent_seed=args.parent,segment=args.segment,training_arm=args.arm,
        source_commit=packet['source_commit'],data_sha256=packet['files_sha256'],parent=parent,parameter_count=839248,
        initial_state_sha256=state_sha(model),command=[sys.executable,*sys.argv],started_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device=str(args.device),device_name=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else 'CPU',dtype='float32',tf32=False,deterministic=True,threads=args.threads))
    common.json_file(out/'metadata.json',meta);logs=[]
    try:
        if args.mode in ('train','preflight'):
            arrays=np.load(args.packet/'training.npz');steps=packet['steps']
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);train_start=time.perf_counter()
            for step in range(steps):
                common.before_deadline(deadline);n=int(arrays['lengths'][step]);x=torch.tensor(arrays[args.arm][step,:,:n].astype(np.int64),device=args.device);y=torch.tensor(arrays['targets'][step].astype(np.int64),device=args.device)
                model.train();optimizer.zero_grad(set_to_none=True);result=model(x)[:,-1];loss=F.cross_entropy(result,y)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward()
                memory_grads=[p.grad.detach().float().square().sum() for name,p in model.named_parameters() if name.startswith('layers.') and '.4.' in name and p.grad is not None]
                memory_norm=torch.stack(memory_grads).sum().sqrt() if memory_grads else loss.new_tensor(0.)
                if not torch.isfinite(memory_norm):raise RuntimeError('Nonfinite memory gradient')
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();common.sync(torch.device(args.device))
                row=dict(step=step+1,loss=float(loss.detach()),accuracy=float((result.argmax(-1)==y).float().mean()),gradient_norm=float(norm),memory_gradient_norm=float(memory_norm),answers=len(y),input_bytes=x.numel(),elapsed_seconds=time.perf_counter()-train_start);logs.append(row)
                with (out/'training.csv').open('a',newline='',encoding='utf8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(row))
                    if step==0:writer.writeheader()
                    writer.writerow(row)
                if step==0 or (step+1)%100==0:print(json.dumps(row),flush=True)
                if step+1==1000:
                    save_checkpoint(out/'midpoint.pt',model,config,source_info=dict(source_commit=packet['source_commit']),training_info=dict(task='sequence-bounded-freshness-midpoint',steps=1000,selection='Recovery only; terminal2000 primary'))
                    torch.save(optimizer.state_dict(),out/'midpoint-optimizer.pt')
            meta.update(completed_steps=len(logs),training_elapsed_seconds=time.perf_counter()-train_start,supervised_answers=sum(r['answers'] for r in logs),training_input_bytes=sum(r['input_bytes'] for r in logs),memory_nonzero_gradient_steps=sum(r['memory_gradient_norm']>0 for r in logs))
            torch.save(optimizer.state_dict(),out/'optimizer.pt')
        else:meta.update(completed_steps=0,training_elapsed_seconds=0,supervised_answers=0,training_input_bytes=0)
        info=dict(task='sequence-bounded-freshness',parent_init_seed=args.parent,parent_checkpoint_sha256=parent['sha256'],steps=meta['completed_steps'],training_arm=args.arm,segment=args.segment,fixed_terminal=True)
        meta['checkpoint_sha256']=save_checkpoint(out/'checkpoint.pt',model,config,source_info=dict(source_commit=packet['source_commit'],source_sha256=packet['source_sha256']),training_info=info)
        meta['final_state_sha256']=state_sha(model)
        # Reload proof uses an ordinary two-fact prefix and all256 output scores.
        probe=np.array(list(bytes.fromhex(json.loads((args.packet/'samples.json').read_text())[0]['prefix_hex'])),dtype=np.int64)[None]
        model.eval()
        with torch.no_grad():expected=model(torch.tensor(probe,device=args.device))[:,-1].cpu()
        loaded,_,_=load_checkpoint(out/'checkpoint.pt',map_location=args.device);loaded.eval()
        with torch.no_grad():actual=loaded(torch.tensor(probe,device=args.device))[:,-1].cpu()
        assert torch.equal(expected,actual);meta['reload_max_error']=float((expected-actual).abs().max());del loaded
        meta['evaluation']=evaluate(model,args,out/'evaluation',deadline)
        verify_packet(args.packet);meta['status']='succeeded'
    except BaseException as error:
        meta.update(status='failed',failure=repr(error),traceback=traceback.format_exc(),completed_steps=len(logs));raise
    finally:
        meta.update(elapsed_seconds=time.perf_counter()-start,ended_utc=datetime.now(timezone.utc).isoformat())
        if str(args.device).startswith('cuda'):meta['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
        common.json_file(out/'metadata.json',meta)

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','initial','preflight','train'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--packet',type=Path);p.add_argument('--parents',type=Path);p.add_argument('--history-root',type=Path);p.add_argument('--design',type=Path)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--contexts',type=int,default=8);p.add_argument('--train-seed',type=int,default=91302);p.add_argument('--dev-seed',type=int,default=91323);p.add_argument('--episodes',type=int,default=128);p.add_argument('--revision-episodes',type=int,default=64)
    p.add_argument('--parent',type=int,choices=[68101,68105],default=68101);p.add_argument('--segment',type=int,choices=[32,128],default=32);p.add_argument('--arm',choices=['periodic','periodic_bounded'],default='periodic')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2);p.add_argument('--eval-batch',type=int,default=16);p.add_argument('--max-seconds',type=int,default=4000)
    return p

if __name__=='__main__':
    args=parser().parse_args();setup(args);(prepare if args.mode=='prepare' else execute)(args)
