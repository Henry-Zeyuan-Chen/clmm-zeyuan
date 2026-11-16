"""Fixed-size record selection, given-index decoding and value controls.

Training reads only frozen input/target arrays. Final evaluation additionally
requires a decision lock. Oracle fields are explicit experimental cues; they
are never used to route composed inference from a selector prediction.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,os,shutil,subprocess,sys,time,traceback
from contextlib import nullcontext
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
import torch
import torch.nn.functional as F
from clmm.pretrain import load_checkpoint,save_checkpoint,reset_memory_at_segments,zero_memory_output
from clmm.pretrain.fact_streams import Episode
from clmm.pretrain.record_selection import (TASKS,ENCODINGS,VARIANTS,CONDITIONS,make_codec,training_worlds,
    training_arrays,evaluation_worlds,make_samples,collect_variant_signatures,render_samples,render_decode_query,variant_records)
from scripts.sequence_freshness_experiment import state_sha,parent_model,verify_packet
from scripts.freshness_experiment import capture
from scripts import fact_revision_experiment as common

GATE_CONDITIONS={task:('normal',) if task.startswith('select') else ('normal','oracle') for task in TASKS}


def read(path):
    with (gzip.open(path,'rt',encoding='utf8') if path.suffix=='.gz' else path.open(encoding='utf8')) as f:
        return json.load(f)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(path,'wt',encoding='utf8',compresslevel=6) if path.suffix=='.gz' else path.open('w',encoding='utf8')) as f:
        json.dump(value,f,allow_nan=False)


def legal_bytes(task):
    return tuple(range(48,50)) if task.startswith('select') else tuple(range(97,113))


def score_logits(logits,targets,task):
    """Full256 argmax is primary; restricted scores are explicitly diagnostic."""
    legal=legal_bytes(task)
    if logits.ndim!=2 or logits.shape[-1]!=256 or not torch.isfinite(logits).all():
        raise ValueError('Invalid full256 logits')
    if not torch.isin(targets,torch.tensor(legal,device=targets.device)).all():
        raise ValueError('Target outside task legal alphabet')
    logp=F.log_softmax(logits,dim=-1)
    chosen=logits[:,legal[0]:legal[-1]+1]
    return dict(prediction=logits.argmax(-1),forced_prediction=chosen.argmax(-1)+legal[0],
        nll=F.cross_entropy(logits,targets,reduction='none'),
        forced_nll=F.cross_entropy(chosen,targets-legal[0],reduction='none'),
        legal_probability_mass=logp[:,legal[0]:legal[-1]+1].exp().sum(-1))


def composition_samples(data,codec):
    rows=[]
    for task in ('select','select_aligned'):
        for encoding in ENCODINGS:
            for group,episodes in data.items():
                for variant in VARIANTS:
                    for index,raw in enumerate(episodes):
                        ep=Episode.from_dict(raw);records=variant_records(ep,variant)
                        for sample in render_samples(ep,codec,task,encoding,variant):
                            q=ep.queries[sample['rank']];context=sample['prefix'][:64]
                            row=dict(task=task,encoding=encoding,group=group,variant=variant,condition='normal',episode=index,
                                rank=sample['rank'],key=sample['key'],slot=sample['slot'],selection_target=sample['target'],
                                target=ord(records[sample['slot']].value),prefix_hex=sample['prefix'].hex(),input_bytes=95)
                            for slot in (0,1):
                                row[f'decode{slot}_hex']=(context+render_decode_query(slot,q.template_family,q.template_index)).hex()
                            row['decode_erased_hex']=(context+render_decode_query(0,q.template_family,q.template_index,index_erased=True)).hex()
                            rows.append(row)
    return rows


def predicted_decode_prefix(row,prediction):
    """Only predicted byte selects a frozen candidate, never truth slot/target."""
    name='decode0_hex' if int(prediction)==48 else 'decode1_hex' if int(prediction)==49 else 'decode_erased_hex'
    return bytes.fromhex(row[name])


def prepare(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();codec=make_codec()
    write(out/'key_codec.json',codec.to_dict())
    manifest=[];seen=set(read(args.exclusions))
    shutil.copy2(args.exclusions,out/'prior_exclusions.json')
    manifest.append(dict(kind='supplied_prior',path='prior_exclusions.json',original_path=str(args.exclusions),sha256=common.sha256(args.exclusions),histories=len(seen)))
    for packet in args.exclude_packet:
        for name in ('training_plan.json.gz','bridge_plan.json.gz','gate_worlds.json','final_worlds.json'):
            path=packet/name
            if not path.exists():continue
            ids=collect_variant_signatures(read(path));seen.update(ids)
            destination=out/'exclusion_inputs'/packet.name/name
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
            manifest.append(dict(kind='earlier_engineering',path=destination.relative_to(out).as_posix(),original_path=str(path),sha256=common.sha256(path),histories=len(ids)))
    worlds=training_worlds(args.steps,args.contexts,args.train_seed,codec)
    write(out/'training_plan.json.gz',worlds)
    arrays=training_arrays(worlds,codec)
    np.savez_compressed(out/'training.npz',**arrays)
    own=collect_variant_signatures(worlds);seen.update(own)
    bridge_worlds=training_worlds(args.bridge_steps,args.contexts,args.bridge_seed,codec)
    write(out/'bridge_plan.json.gz',bridge_worlds)
    bridge_arrays=training_arrays(bridge_worlds,codec)
    np.savez_compressed(out/'bridge_training.npz',**bridge_arrays)
    bridge_ids=collect_variant_signatures(bridge_worlds);seen.update(bridge_ids)
    write(out/'excluded_signatures.json',sorted(seen))
    gate=evaluation_worlds(seen,codec,args.gate_seed,args.gate_n)
    write(out/'gate_worlds.json',gate)
    write(out/'gate_samples.json.gz',make_samples(gate,codec,conditions=GATE_CONDITIONS))
    write(out/'gate_composition.json.gz',composition_samples(gate,codec))
    gate_ids=collect_variant_signatures(gate)
    final=evaluation_worlds(seen|gate_ids,codec,args.final_seed,args.final_n)
    write(out/'final_worlds.json',final)
    write(out/'final_samples.json.gz',make_samples(final,codec))
    write(out/'final_composition.json.gz',composition_samples(final,codec))
    write(out/'exclusion_manifest.json',dict(inputs=manifest,union_count=len(seen),main_variant_histories=len(own),
        possible_bridge_variant_histories=len(bridge_ids),gate_histories=len(gate_ids),
        final_histories=len(collect_variant_signatures(final)),novelty='Full two-record history; masked individual facts may recur'))
    del worlds,arrays,bridge_worlds,bridge_arrays
    (out/'parents').mkdir();parents=[]
    for seed in (68101,68105):
        source=args.parents/f'lm-{seed}/checkpoint.pt';target=out/'parents'/f'{seed}.pt';shutil.copy2(source,target)
        model,config,payload=load_checkpoint(target,map_location='cpu')
        from dataclasses import asdict
        assert sum(p.numel() for p in model.parameters())==839248
        assert payload['training_info']['init_seed']==seed and payload['training_info'].get('task') is None
        assert (config.segment_len,config.memory_segment_len,config.memory_batch_size)==(32,8,32)
        parents.append(dict(seed=seed,path=target.relative_to(out).as_posix(),sha256=common.sha256(target),state_sha256=state_sha(model),
                            config=asdict(config),original_training=payload['training_info']))
    write(out/'parents.json',parents)
    shutil.copy2(args.protocol,out/'protocol.json')
    sources=capture(out)
    names=[p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and 'sources' not in p.relative_to(out).parts]
    meta=dict(status='succeeded',kind=args.kind,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_sha256=sources,files_sha256={n:common.sha256(out/n) for n in names},steps=args.steps,bridge_steps=args.bridge_steps,
        contexts=args.contexts,main_seed=args.train_seed,bridge_seed=args.bridge_seed,gate_seed=args.gate_seed,final_seed=args.final_seed,
        gate_n=args.gate_n,final_n=args.final_n,per_update_answers=args.contexts*2,per_update_input_bytes=args.contexts*2*95,
        command=[sys.executable,*sys.argv],environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__),
        elapsed_seconds=time.perf_counter()-start)
    write(out/'metadata.json',meta)
    print(json.dumps({k:v for k,v in meta.items() if k not in ('source_sha256','files_sha256','command')}),flush=True)


def guard_final(args,packet):
    if args.dataset!='final' or packet['kind']!='main':return
    if args.decision_lock is None:raise ValueError('Final model outputs are sealed until decision lock')
    lock=read(args.decision_lock)
    assert lock['source_commit']==packet['source_commit']
    assert lock['packet_metadata_sha256']==common.sha256(args.packet/'metadata.json')
    assert lock['phase']=='confirmation_authorized' and lock['adaptive_training_complete']


def scope_for(model,condition):
    if condition=='reset_memory':return reset_memory_at_segments(model)
    if condition=='retrieval_zero':return zero_memory_output(model)
    return nullcontext()


def summarize(rows):
    groups=[];pairmap={}
    for row in rows:
        key=(row['group'],row['variant'],row['condition'],int(row['episode']))
        pairmap.setdefault(key,[]).append(row)
    assert all(len(p)==2 for p in pairmap.values())
    for group,variant,condition in dict.fromkeys(k[:3] for k in pairmap):
        pairs=[p for k,p in pairmap.items() if k[:3]==(group,variant,condition)]
        flat=[r for p in pairs for r in p]
        groups.append(dict(group=group,variant=variant,condition=condition,episodes=len(pairs),
            answers=len(flat),correct_count=sum(r['correct'] for r in flat),accuracy=float(np.mean([r['correct'] for r in flat])),
            both_correct_count=sum(all(r['correct'] for r in p) for p in pairs),
            both_correct=float(np.mean([all(r['correct'] for r in p) for p in pairs])),
            prediction_changes=float(np.mean([p[0]['prediction']!=p[1]['prediction'] for p in pairs])),
            nll=float(np.mean([r['nll'] for r in flat])),legal_probability_mass=float(np.mean([r['legal_probability_mass'] for r in flat]))))
    joint=[]
    for group,condition in dict.fromkeys((k[0],k[2]) for k in pairmap):
        ids=sorted({k[3] for k in pairmap if k[0]==group and k[2]==condition})
        if not all((group,v,condition,i) in pairmap for v in VARIANTS for i in ids):continue
        counts={}
        for label,vs in [('joint4',('original','order')),('joint4_swapped',('value','order_value')),('joint8',VARIANTS)]:
            counts[label]=sum(all(r['correct'] for v in vs for r in pairmap[group,v,condition,i]) for i in ids)
        joint.append(dict(group=group,condition=condition,episodes=len(ids),**counts))
    return dict(groups=groups,joint=joint)


@torch.no_grad()
def evaluate_model(model,args,out,deadline,*,task=None,encoding=None,compose=False,packet=None):
    task=args.task if task is None else task;encoding=args.encoding if encoding is None else encoding
    packet=verify_packet(args.packet) if packet is None else packet
    guard_final(args,packet);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();model.eval()
    before=state_sha(model);rng=torch.random.get_rng_state().clone()
    cuda=torch.cuda.get_rng_state(args.device).clone() if args.device.startswith('cuda') else None
    source=read(args.packet/f'{args.dataset}_{"composition" if compose else "samples"}.json.gz')
    rows=[r for r in source if r['task']==task and r['encoding']==encoding];del source
    assert rows
    logits=np.empty((len(rows),256),dtype=np.float32)
    selector_logits=np.empty_like(logits) if compose else None
    resources=[];buckets={}
    for i,row in enumerate(rows):buckets.setdefault(row['condition'],[]).append(i)
    for condition,indices in buckets.items():
        with scope_for(model,condition):
            for offset in range(0,len(indices),args.eval_batch):
                common.before_deadline(deadline);ix=indices[offset:offset+args.eval_batch];tick=time.perf_counter()
                x=torch.tensor([list(bytes.fromhex(rows[i]['prefix_hex'])) for i in ix],device=args.device)
                result=model(x)[:,-1]
                if compose:
                    selector_logits[ix]=result.cpu().numpy();predicted=result.argmax(-1).cpu().tolist()
                    raw=[predicted_decode_prefix(rows[i],prediction) for i,prediction in zip(ix,predicted)]
                    for i,prediction,prefix in zip(ix,predicted,raw):
                        rows[i].update(selection_prediction=prediction,route_index=prediction-48 if prediction in (48,49) else -1,decode_prefix_hex=prefix.hex())
                    result=model(torch.tensor([list(p) for p in raw],device=args.device))[:,-1]
                y=torch.tensor([rows[i]['target'] for i in ix],device=args.device)
                scores={k:v.cpu().tolist() for k,v in score_logits(result,y,'value' if compose else task).items()}
                logits[ix]=result.cpu().numpy()
                for j,i in enumerate(ix):
                    rows[i].update(score_index=i,**{k:v[j] for k,v in scores.items()},correct=int(scores['prediction'][j]==rows[i]['target']))
                common.sync(torch.device(args.device))
                resources.append(dict(condition=condition,input_length=95,row_indices=json.dumps(ix),
                    physical_forward_calls=2 if compose else 1,elapsed_seconds=time.perf_counter()-tick))
    assert before==state_sha(model) and torch.equal(rng,torch.random.get_rng_state())
    if cuda is not None:assert torch.equal(cuda,torch.cuda.get_rng_state(args.device))
    reset_pairs_checked=0
    if not compose:
        pairs={}
        for i,row in enumerate(rows):
            pairs.setdefault((row['group'],row['variant'],row['episode'],row['rank']),{})[row['condition']]=i
        for index in pairs.values():
            if 'reset_memory' in index:
                assert np.array_equal(logits[index['normal']],logits[index['reset_memory']]);reset_pairs_checked+=1
        erased='index_erased' if task=='decode' else 'key_erased'
        blocks={}
        for i,row in enumerate(rows):
            if row['condition']==erased:blocks.setdefault((row['group'],row['variant'],row['episode']),[]).append(i)
        for pair in blocks.values():
            assert bytes.fromhex(rows[pair[0]]['prefix_hex'])==bytes.fromhex(rows[pair[1]]['prefix_hex'])
            assert np.array_equal(logits[pair[0]],logits[pair[1]])
    np.save(out/'answer_logits.npy',logits)
    if compose:np.save(out/'selection_logits.npy',selector_logits)
    common.write_csv(out/'predictions.csv',rows);common.write_csv(out/'resources.csv',resources)
    summary=summarize(rows);write(out/'summary.json',summary)
    meta=dict(status='succeeded',task=task,encoding=encoding,dataset=args.dataset,compose=compose,queries=len(rows),
        checkpoint_pure=True,RNG_unchanged=True,normal_reset_parity=True if reset_pairs_checked else None,reset_pairs_checked=reset_pairs_checked,
        answer_logits_sha256=common.sha256(out/'answer_logits.npy'),predictions_sha256=common.sha256(out/'predictions.csv'),
        source_commit=packet['source_commit'],packet_sha256=common.sha256(args.packet/'metadata.json'),
        decision_lock_sha256=common.sha256(args.decision_lock) if args.decision_lock else None,
        elapsed_seconds=time.perf_counter()-start,physical_forward_calls=sum(r['physical_forward_calls'] for r in resources))
    write(out/'metadata.json',meta);return meta


def model_from_args(args,packet):
    if args.init_checkpoint:
        model,config,payload=load_checkpoint(args.init_checkpoint,map_location=args.device)
        info=payload['training_info']
        assert payload['source_info']['source_commit']==packet['source_commit']
        assert info['parent_init_seed']==args.parent and info['encoding']==args.encoding and info['task']==args.selector_task
        assert info['phase']=='main' and info['steps'] in (10000,20000)
        assert config.segment_len==128 and sum(p.numel() for p in model.parameters())==839248
        return model,config,dict(checkpoint_sha256=common.sha256(args.init_checkpoint),state_sha256=state_sha(model),training_info=info)
    model,config,parent=parent_model(args);return model,config,parent


def save_snapshot(out,step,model,config,optimizer,args,packet,initial):
    path=out/'checkpoints'/str(step);path.mkdir(parents=True,exist_ok=False)
    info=dict(task=args.task,selector_task=args.selector_task,parent_init_seed=args.parent,encoding=args.encoding,
        phase=args.phase,steps=step,planned_steps=args.steps,init_state_sha256=initial,optimizer_reset=True,
        init_checkpoint_sha256=common.sha256(args.init_checkpoint) if args.init_checkpoint else None,
        schedule='zero-based even selector, odd decode' if args.task=='sd' else args.task)
    digest=save_checkpoint(path/'model.pt',model,config,source_info=dict(source_commit=packet['source_commit'],source_sha256=packet['source_sha256']),training_info=info)
    if optimizer is not None:torch.save(optimizer.state_dict(),path/'optimizer.pt')
    return dict(step=step,path=str((path/'model.pt').relative_to(out)),sha256=digest,state_sha256=state_sha(model))


def train(args):
    packet=verify_packet(args.packet);out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();deadline=start+args.max_seconds
    model,config,parent=model_from_args(args,packet);initial=state_sha(model)
    assert args.steps<=packet['steps' if args.phase=='main' else 'bridge_steps']
    tasks=(args.selector_task,'decode') if args.task=='sd' else (args.task,)
    archive_name='training.npz' if args.phase=='main' else 'bridge_training.npz'
    with np.load(args.packet/archive_name) as archive:
        inputs={t:archive[f'{t}_{args.encoding}'] for t in tasks}
        targets={t:archive[f'targets_{t}'] for t in tasks};slots=archive['slots']
    meta=dict(status='running',task=args.task,selector_task=args.selector_task,encoding=args.encoding,parent_seed=args.parent,
        phase=args.phase,steps_planned=args.steps,completed_steps=0,parameter_count=839248,segment=128,initial_state_sha256=initial,
        parent=parent,source_commit=packet['source_commit'],packet_sha256=common.sha256(args.packet/'metadata.json'),
        command=[sys.executable,*sys.argv],started_utc=datetime.now(timezone.utc).isoformat(),snapshots=[],gates=[],
        environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device=args.device,
            device_name=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'CPU',threads=args.threads,dtype='float32',TF32=False,deterministic=True))
    write(out/'metadata.json',meta)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    train_start=time.perf_counter();paused=0.;answers=0;input_bytes=0;nonzero=0
    checkpoints=sorted(set(n for n in (2000,10000,args.steps) if n<=args.steps))
    if args.phase=='bridge':checkpoints=sorted(set(n for n in (2000,args.steps) if n<=args.steps))
    with (out/'training.csv').open('w',newline='',encoding='utf8') as f:
        writer=None
        try:
            for i in range(args.steps):
                common.before_deadline(deadline);task=tasks[i%2] if args.task=='sd' else args.task
                x=torch.tensor(inputs[task][i].astype(np.int64),device=args.device);y=torch.tensor(targets[task][i].astype(np.int64),device=args.device)
                model.train();optimizer.zero_grad(set_to_none=True);result=model(x)[:,-1];loss=F.cross_entropy(result,y)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward()
                mg=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if n.startswith('layers.') and '.4.' in n and p.grad is not None]
                memory_norm=torch.stack(mg).sum().sqrt() if mg else loss.new_tensor(0.)
                if not torch.isfinite(memory_norm):raise RuntimeError('Nonfinite memory gradient')
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                common.sync(torch.device(args.device));p=result.detach().argmax(-1);correct=p==y
                position=torch.tensor(slots[i],device=args.device)
                row=dict(step=i+1,task=task,loss=float(loss.detach()),accuracy=float(correct.float().mean()),
                    both_correct=float(correct.reshape(-1,2).all(1).float().mean()),
                    prediction_changes=float((p.reshape(-1,2)[:,0]!=p.reshape(-1,2)[:,1]).float().mean()),
                    accuracy_slot0=float(correct[position==0].float().mean()),accuracy_slot1=float(correct[position==1].float().mean()),
                    gradient_norm=float(norm),memory_gradient_norm=float(memory_norm),answers=len(y),input_bytes=x.numel(),
                    elapsed_training_seconds=time.perf_counter()-train_start-paused)
                if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                writer.writerow(row);answers+=len(y);input_bytes+=x.numel();nonzero+=float(memory_norm)>0
                if i==0 or (i+1)%100==0:f.flush();print(json.dumps(row),flush=True)
                if i+1 in checkpoints:
                    f.flush();tick=time.perf_counter()
                    snapshot=save_snapshot(out,i+1,model,config,optimizer,args,packet,initial);meta['snapshots'].append(snapshot)
                    # Gates are development outputs; final outputs remain sealed.
                    previous_dataset=args.dataset;args.dataset='gate'
                    gate_tasks=tasks if args.task=='sd' else (args.task,)
                    for gate_task in gate_tasks:
                        directory=out/'gates'/str(i+1)/gate_task
                        value=evaluate_model(model,args,directory,deadline,task=gate_task,packet=packet)
                        meta['gates'].append(dict(step=i+1,task=gate_task,path=str(directory.relative_to(out)),queries=value['queries']))
                    args.dataset=previous_dataset
                    paused+=time.perf_counter()-tick
                    meta.update(completed_steps=i+1,training_seconds=row['elapsed_training_seconds'],supervised_answers=answers,
                        training_input_bytes=input_bytes,memory_nonzero_gradient_steps=nonzero)
                    write(out/'metadata.json',meta)
            meta.update(status='succeeded',final_checkpoint=meta['snapshots'][-1]['path'],final_state_sha256=state_sha(model))
            # Exact same-environment reload proof on a frozen ordinary gate prefix.
            sample=next(r for r in read(args.packet/'gate_samples.json.gz') if r['task']==tasks[0] and r['encoding']==args.encoding)
            probe=torch.tensor([list(bytes.fromhex(sample['prefix_hex']))],device=args.device);model.eval()
            with torch.no_grad():expected=model(probe)[:,-1].cpu()
            loaded,_,_=load_checkpoint(out/meta['final_checkpoint'],map_location=args.device);loaded.eval()
            with torch.no_grad():actual=loaded(probe)[:,-1].cpu()
            assert torch.equal(expected,actual);meta['reload_max_error']=float((expected-actual).abs().max());del loaded
        except BaseException as error:
            meta.update(status='failed',failure=repr(error),traceback=traceback.format_exc(),completed_steps=answers//packet['per_update_answers'])
            raise
        finally:
            meta.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,
                supervised_answers=answers,training_input_bytes=input_bytes,memory_nonzero_gradient_steps=nonzero)
            if args.device.startswith('cuda'):meta['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
            write(out/'metadata.json',meta)


def initial(args):
    packet=verify_packet(args.packet);out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    model,config,parent=parent_model(args)
    digest=save_checkpoint(out/'model.pt',model,config,source_info=dict(source_commit=packet['source_commit'],source_sha256=packet['source_sha256']),
        training_info=dict(task='initial',parent_init_seed=args.parent,steps=0,segment=128))
    write(out/'metadata.json',dict(status='succeeded',mode='initial',parent_seed=args.parent,parameter_count=839248,
        checkpoint_sha256=digest,state_sha256=state_sha(model),parent=parent,source_commit=packet['source_commit']))


def evaluation(args):
    packet=verify_packet(args.packet);guard_final(args,packet)
    model,config,payload=load_checkpoint(args.checkpoint,map_location=args.device)
    assert config.segment_len==128 and sum(p.numel() for p in model.parameters())==839248
    assert payload['source_info']['source_commit']==packet['source_commit']
    deadline=time.perf_counter()+args.max_seconds
    result=evaluate_model(model,args,args.out.resolve(),deadline,compose=args.mode=='compose',packet=packet)
    meta=read(args.out/'metadata.json');meta.update(checkpoint_sha256=common.sha256(args.checkpoint),checkpoint_training_info=payload['training_info'],
        parameter_count=839248,environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device=args.device),command=[sys.executable,*sys.argv])
    write(args.out/'metadata.json',meta)
    print(json.dumps({k:v for k,v in result.items() if k not in ('command',)}),flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','initial','train','evaluate','compose'))
    p.add_argument('--out',required=True,type=Path);p.add_argument('--packet',type=Path);p.add_argument('--parents',type=Path)
    p.add_argument('--protocol',type=Path);p.add_argument('--exclusions',type=Path);p.add_argument('--exclude-packet',action='append',type=Path,default=[])
    p.add_argument('--kind',choices=('main','engineering'),default='main');p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--bridge-steps',type=int,default=6000);p.add_argument('--contexts',type=int,default=8)
    p.add_argument('--train-seed',type=int,default=91602);p.add_argument('--bridge-seed',type=int,default=91702)
    p.add_argument('--gate-seed',type=int,default=91613);p.add_argument('--final-seed',type=int,default=91623)
    p.add_argument('--gate-n',type=int,default=64);p.add_argument('--final-n',type=int,default=256)
    p.add_argument('--parent',type=int,choices=(68101,68105),default=68101);p.add_argument('--segment',type=int,choices=(128,),default=128)
    p.add_argument('--encoding',choices=ENCODINGS,default='digits');p.add_argument('--task',choices=(*TASKS,'sd'),default='select')
    p.add_argument('--selector-task',choices=('select','select_aligned'),default='select');p.add_argument('--phase',choices=('main','bridge'),default='main')
    p.add_argument('--init-checkpoint',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--dataset',choices=('gate','final'),default='gate');p.add_argument('--decision-lock',type=Path)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2);p.add_argument('--eval-batch',type=int,default=16)
    p.add_argument('--max-seconds',type=int,default=3600)
    return p


if __name__=='__main__':
    args=parser().parse_args();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.manual_seed(91602)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {'prepare':prepare,'initial':initial,'train':train,'evaluate':evaluation,'compose':evaluation}[args.mode](args)
