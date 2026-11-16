"""Frozen paired format training; reserved wording stays sealed until all runs finish."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,os,shutil,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
import torch
import torch.nn.functional as F
from clmm.pretrain import load_checkpoint,save_checkpoint
from clmm.pretrain.format_training import fresh_training,training_arrays,evaluation_samples,templates,validate_confirmation_lock
from clmm.pretrain.format_factorial import fresh_worlds
from clmm.pretrain.record_selection import make_codec,collect_variant_signatures
from scripts.selection_experiment import score_logits
from scripts.sequence_freshness_experiment import state_sha
from scripts.format_factorial_experiment import read,write,sha,csvfile,utc


def verify(packet):
    m=read(packet/'metadata.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==m['source_commit']
    for name,digest in m['source_sha256'].items():assert sha(REPO/name)==digest,name
    for name,digest in m['files_sha256'].items():assert sha(packet/name)==digest,name
    return m


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False);o=a.out;prior=set(read(a.exclusions));assert len(prior)==790004;codec=make_codec()
    protocol=read(a.protocol);assert protocol['reserved_fact_templates']==templates()['facts'][2] and protocol['reserved_query_templates']==templates()['queries'][2]
    write(o/'protocol.json',protocol);write(o/'templates.json',templates());write(o/'key_codec.json',codec.to_dict());shutil.copy2(a.exclusions,o/'prior_exclusions.json.gz')
    eng,seen=fresh_training(prior,steps=64,seed=92102,codec=codec)
    eng_eval=fresh_worlds(seen,seed=92113,n_train=8,n_dev=8,codec=codec);seen.update(collect_variant_signatures(eng_eval))
    main,seen=fresh_training(seen,steps=6000,seed=92002,codec=codec)
    confirmation=fresh_worlds(seen,seed=92023,n_train=128,n_dev=256,codec=codec)
    for name,plan in [('engineering',eng),('main',main)]:
        write(o/f'{name}_training_plan.json.gz',plan)
        np.savez_compressed(o/f'{name}_training.npz',**training_arrays(plan,codec))
    for name,worlds in [('engineering',eng_eval),('confirmation',confirmation)]:
        write(o/f'{name}_worlds.json',worlds);write(o/f'{name}_samples.json.gz',evaluation_samples(worlds,codec,confirmation=name=='confirmation'))
    seen.update(collect_variant_signatures(confirmation));write(o/'all_consulted_histories.json.gz',sorted(seen))
    assert len(seen)==985652
    write(o/'history_counts.json',dict(prior=790004,engineering_training=2048,engineering_evaluation=64,main_training=192000,confirmation=1536,total=len(seen)))
    initializers=[];(o/'initializers').mkdir()
    for c in read(a.prior_packet/'cases.json'):
        if c['stage']!='trained' or c['task']!='select':continue
        p=o/'initializers'/f"{c['parent']}-{c['encoding']}.pt";shutil.copy2(a.prior_packet/c['checkpoint'],p);assert sha(p)==c['checkpoint_sha256']
        model,cfg,payload=load_checkpoint(p,map_location='cpu');assert state_sha(model)==c['state_sha256']
        info=payload['training_info'];assert info['task']=='select' and info['steps']==20000 and cfg.segment_len==128
        assert sum(p.numel() for p in model.parameters())==839248
        initializers.append(dict(id=f"{c['parent']}-{c['encoding']}",parent=c['parent'],encoding=c['encoding'],checkpoint=p.relative_to(o).as_posix(),checkpoint_sha256=sha(p),state_sha256=c['state_sha256'],inherited_updates=20000))
    assert len(initializers)==4;write(o/'initializers.json',initializers)
    sources={}
    for p in [*sorted((REPO/'clmm').rglob('*.py')),*sorted((REPO/'titans_pytorch').glob('*.py')),*sorted((REPO/'scripts').glob('*.py')),REPO/'pyproject.toml',REPO/'requirements-lock.txt']:
        relative=p.relative_to(REPO);dest=o/'sources'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);sources[relative.as_posix()]=sha(p)
    old=read(a.prior_packet/'original_model_source.json')
    for name,digest in old.items():assert sha(REPO/name)==digest,name
    hashes={p.relative_to(o).as_posix():sha(p) for p in o.rglob('*') if p.is_file() and 'sources' not in p.relative_to(o).parts}
    meta=dict(status='prepared',created_utc=utc(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),source_sha256=sources,original_model_source=old,files_sha256=hashes,
        command=[sys.executable,*sys.argv],confirmation_packet_rows=58368,confirmation_vectors=350208,environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__))
    write(o/'metadata.json',meta);print(json.dumps(dict(status='prepared',source=meta['source_commit'],histories=len(seen))),flush=True)


def get_initializer(packet,parent,encoding):return next(c for c in read(packet/'initializers.json') if c['parent']==parent and c['encoding']==encoding)


def train(a):
    packet=verify(a.packet);a.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();deadline=start+a.max_seconds
    init=get_initializer(a.packet,a.parent,a.encoding);model,cfg,payload=load_checkpoint(a.packet/init['checkpoint'],map_location=a.device)
    assert state_sha(model)==init['state_sha256'] and cfg.segment_len==128 and sum(p.numel() for p in model.parameters())==839248
    steps=64 if a.dataset=='engineering' else 6000
    with np.load(a.packet/f'{a.dataset}_training.npz') as f:xplan=f[f'{a.arm}_{a.encoding}'];yplan=f['targets']
    assert xplan.shape==(steps,16,95) and yplan.shape==(steps,16)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);assert not opt.state
    meta=dict(status='running',started_utc=utc(),source_commit=packet['source_commit'],packet_metadata_sha256=sha(a.packet/'metadata.json'),initializer=init,arm=a.arm,
        dataset=a.dataset,steps_planned=steps,completed_steps=0,registered_parameters=839248,optimizer_reset=True,command=[sys.executable,*sys.argv],
        environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device=a.device,device_name=torch.cuda.get_device_name(a.device),threads=a.threads,dtype='float32',TF32=False,deterministic=True))
    write(a.out/'metadata.json',meta);durations=[];nonzero=0
    with (a.out/'training.csv').open('w',newline='',encoding='utf8') as f:
        writer=None
        try:
            for step in range(steps):
                if time.perf_counter()>deadline:raise TimeoutError('Training deadline')
                tick=time.perf_counter();x=torch.tensor(xplan[step].astype(np.int64),device=a.device);y=torch.tensor(yplan[step].astype(np.int64),device=a.device)
                model.train();opt.zero_grad(set_to_none=True);logits=model(x)[:,-1];loss=F.cross_entropy(logits,y)
                assert torch.isfinite(loss);loss.backward()
                grads=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if n.startswith('layers.') and '.4.' in n and p.grad is not None]
                memory_norm=torch.stack(grads).sum().sqrt() if grads else loss.new_tensor(0.);assert torch.isfinite(memory_norm)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize(a.device)
                seconds=time.perf_counter()-tick;durations.append(seconds);nonzero+=float(memory_norm)>0
                correct=logits.detach().argmax(-1)==y
                row=dict(step=step+1,loss=float(loss.detach()),accuracy=float(correct.float().mean()),both_correct=float(correct.reshape(8,2).all(1).float().mean()),
                    gradient_norm=float(norm),memory_gradient_norm=float(memory_norm),answers=16,input_bytes=1520,step_seconds=seconds)
                if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                writer.writerow(row)
                if step==0 or (step+1)%100==0:f.flush();print(json.dumps(row),flush=True)
                if (step+1)%100==0:meta.update(completed_steps=step+1);write(a.out/'metadata.json',meta)
            assert nonzero>0
            digest=save_checkpoint(a.out/'model.pt',model,cfg,source_info=dict(source_commit=packet['source_commit'],source_sha256=packet['source_sha256']),
                training_info=dict(task='value',phase='format_training',arm=a.arm,steps=steps,parent_init_seed=a.parent,encoding=a.encoding,inherited_selector_updates=20000,init_checkpoint_sha256=init['checkpoint_sha256'],init_state_sha256=init['state_sha256'],optimizer_reset=True))
            torch.save(opt.state_dict(),a.out/'optimizer.pt')
            # Familiar engineering observation only; reserved wording is never used here.
            row=next(r for r in read(a.packet/'engineering_samples.json.gz') if r['encoding']==a.encoding)
            x=torch.tensor([list(bytes.fromhex(row['prefix_hex']))],device=a.device);model.eval()
            with torch.no_grad():expected=model(x)[:,-1].cpu()
            loaded,_,_=load_checkpoint(a.out/'model.pt',map_location=a.device);loaded.eval()
            with torch.no_grad():actual=loaded(x)[:,-1].cpu()
            assert torch.equal(expected,actual)
            meta.update(status='succeeded',completed_steps=steps,checkpoint_sha256=digest,state_sha256=state_sha(model),optimizer_sha256=sha(a.out/'optimizer.pt'),exact_reload=True)
        except BaseException:
            meta.update(status='failed',traceback=traceback.format_exc());raise
        finally:
            meta.update(completed_steps=len(durations),ended_utc=utc(),elapsed_seconds=time.perf_counter()-start,training_seconds=sum(durations),warm_step_seconds=float(np.mean(durations[8:] or durations)),
                supervised_answers=len(durations)*16,training_input_bytes=len(durations)*1520,memory_nonzero_gradient_steps=nonzero,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
            write(a.out/'metadata.json',meta)


@torch.no_grad()
def evaluate(a):
    packet=verify(a.packet);start=time.perf_counter();deadline=start+a.max_seconds;checkpoint_digest=sha(a.checkpoint)
    if a.dataset=='confirmation':
        lock=read(a.decision_lock);validate_confirmation_lock(lock,sha(a.packet/'metadata.json'),checkpoint_digest)
        assert lock['created_utc']<=utc()
    model,cfg,payload=load_checkpoint(a.checkpoint,map_location=a.device);assert cfg.segment_len==128 and sum(p.numel() for p in model.parameters())==839248
    model.eval();before=state_sha(model);rng=torch.random.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(a.device).clone()
    rows=[dict(r,case=a.case) for r in read(a.packet/f'{a.dataset}_samples.json.gz') if r['encoding']==a.encoding]
    for i,r in enumerate(rows):r['score_index']=i
    buckets={}
    for i,r in enumerate(rows):buckets.setdefault((r['condition'],r['cell'],r['group'],r['variant']),[]).append(i)
    logits=np.empty((len(rows),256),np.float32);resources=[];a.out.mkdir(parents=True,exist_ok=False)
    for key,indices in buckets.items():
        for offset in range(0,len(indices),16):
            if time.perf_counter()>deadline:raise TimeoutError('Evaluation deadline')
            ix=indices[offset:offset+16];x=torch.tensor([list(bytes.fromhex(rows[i]['prefix_hex'])) for i in ix],device=a.device)
            torch.cuda.synchronize(a.device);tick=time.perf_counter();y=model(x)[:,-1];torch.cuda.synchronize(a.device);seconds=time.perf_counter()-tick
            targets=torch.tensor([rows[i]['target'] for i in ix],device=a.device);scores={k:v.cpu().tolist() for k,v in score_logits(y,targets,'value').items()};logits[ix]=y.cpu().numpy()
            for j,i in enumerate(ix):rows[i].update(**{k:v[j] for k,v in scores.items()},correct=int(scores['prediction'][j]==rows[i]['target']))
            resources.append(dict(condition=key[0],cell=key[1],group=key[2],variant=key[3],row_indices=json.dumps(ix),queries=len(ix),forward_seconds=seconds))
    assert before==state_sha(model) and torch.equal(rng,torch.random.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state(a.device))
    np.save(a.out/'logits.npy',logits);csvfile(a.out/'predictions.csv',rows);csvfile(a.out/'resources.csv',resources)
    write(a.out/'metadata.json',dict(status='succeeded',case=a.case,encoding=a.encoding,dataset=a.dataset,ended_utc=utc(),queries=len(rows),registered_parameters=839248,
        checkpoint_sha256=checkpoint_digest,state_sha256=before,weights_and_RNG_pure=True,source_commit=packet['source_commit'],packet_metadata_sha256=sha(a.packet/'metadata.json'),
        logits_sha256=sha(a.out/'logits.npy'),predictions_sha256=sha(a.out/'predictions.csv'),forward_seconds=sum(r['forward_seconds'] for r in resources),elapsed_seconds=time.perf_counter()-start,
        environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device=a.device,device_name=torch.cuda.get_device_name(a.device)),command=[sys.executable,*sys.argv]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','train','evaluate']);p.add_argument('--out',type=Path,required=True)
    for n in ['packet','protocol','prior-packet','exclusions','checkpoint','decision-lock']:p.add_argument('--'+n,type=Path)
    p.add_argument('--parent',type=int,default=68101);p.add_argument('--encoding',choices=['digits','atomic'],default='digits');p.add_argument('--arm',choices=['familiar','mixed'],default='familiar')
    p.add_argument('--dataset',choices=['engineering','main','confirmation'],default='main');p.add_argument('--case');p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=2);p.add_argument('--max-seconds',type=int,default=13000)
    a=p.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(92007);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'prepare':prepare,'train':train,'evaluate':evaluate}[a.mode](a)
