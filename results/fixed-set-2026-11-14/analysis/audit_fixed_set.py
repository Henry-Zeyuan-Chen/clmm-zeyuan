"""Independent CPU-only saved-artifact audit; never imports or forwards a model.

The two frozen data modules are imported only after source/Git digest checks.
Scoring, semantic matching, inference summaries and bootstrap analysis are
implemented here independently of the experiment runner.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
import types
import numpy as np

SCRIPT = Path(__file__).resolve().parent
MAIN = ('fixed32-68101','fixed128-68101','fixed32-68105','fixed128-68105')
GROUPS = ('train','novel_values','heldout_keys','heldout_keys_templates','train_query_only')
CONDS = ('online','reset_memory')
METRICS = ('accuracy','both_correct','first_accuracy','second_accuracy','prediction_changes','nll','legal_probability_mass')

class AuditError(RuntimeError): pass

def require(value,message):
    if not value: raise AuditError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
def csvread(path):
    with Path(path).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def csvwrite(path,rows):
    require(bool(rows),f'Empty table {path}')
    with Path(path).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def near(a,b,label):
    require(math.isfinite(float(a)) and math.isclose(float(a),float(b),rel_tol=5e-6,abs_tol=5e-5),f'{label}: {a} != {b}')

def score(logits,targets):
    z=np.asarray(logits,dtype=np.float64);t=np.asarray(targets,dtype=np.int64)
    require(z.shape==(len(t),256) and np.isfinite(z).all(),'Invalid full-256 logits')
    require(np.all((t>=97)&(t<=112)),'Illegal targets')
    m=z.max(1); logsum=m+np.log(np.exp(z-m[:,None]).sum(1))
    legal=z[:,97:113];lm=legal.max(1);ls=lm+np.log(np.exp(legal-lm[:,None]).sum(1))
    p=z.argmax(1);fp=legal.argmax(1)+97
    return dict(prediction=p,forced_prediction=fp,correct=(p==t).astype(int),
        nll=logsum-z[np.arange(len(t)),t],forced_nll=ls-z[np.arange(len(t)),t],
        legal_probability_mass=np.exp(ls-logsum))

def bootstrap(values,draws,seed):
    v=np.asarray(values,dtype=float)
    require(v.ndim==1 and len(v)>0 and np.isfinite(v).all(),'Invalid bootstrap vector')
    ix=np.random.default_rng(seed).integers(0,len(v),size=(draws,len(v)))
    lo,hi=np.quantile(v[ix].mean(1),[.025,.975])
    return float(lo),float(hi)

def source_audit(folder,meta,commit,repo):
    require(meta['status']=='succeeded',f'Run incomplete/failed: {folder.name}')
    require(meta['repo_commit']==commit,f'Unexpected source revision: {folder.name}')
    manifest=meta['source_sha256']
    for required in ('clmm/pretrain/fact_streams.py','clmm/pretrain/association_probe.py','scripts/fixed_set_generalization.py'):
        require(required in manifest,f'Missing source manifest entry {required}')
    for name,digest in manifest.items():
        p=(folder/'sources'/name).resolve()
        require(p.is_relative_to((folder/'sources').resolve()),'Unsafe source path')
        require(p.is_file() and sha(p)==digest,f'Frozen source hash differs: {name}')
        if repo:
            data=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=repo)
            require(hashlib.sha256(data).hexdigest()==digest,f'Source differs from Git object: {name}')
    return manifest

def data_modules(folder):
    package=types.ModuleType('_frozen_association_audit');package.__path__=[]
    sys.modules[package.__name__]=package
    loaded=[]
    for name in ('fact_streams','association_probe'):
        full=package.__name__+'.'+name
        spec=importlib.util.spec_from_file_location(full,folder/'sources/clmm/pretrain'/f'{name}.py')
        require(spec is not None and spec.loader is not None,'Cannot load frozen data module')
        mod=importlib.util.module_from_spec(spec);sys.modules[full]=mod;spec.loader.exec_module(mod);loaded.append(mod)
    return loaded

def load_tensors(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=True)

def same_state(left,right,label):
    import torch
    require(isinstance(left,dict) and isinstance(right,dict) and set(left)==set(right),f'{label}: names differ')
    for k in left:
        a,b=left[k],right[k]
        require(isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype
            and torch.equal(a,b),f'{label}: tensor differs {k}')

def parents_at(folder,design):
    found={}
    for p in Path(folder).rglob('checkpoint.pt'):
        digest=sha(p)
        for seed,wanted in design['parents'].items():
            if digest==wanted:found[int(seed)]=(p,load_tensors(p))
    require(set(found)=={68101,68105},'Both original parents must be located by SHA-256')
    for seed,(path,payload) in found.items():
        info=payload['training_info'];c=payload['config']
        require(info.get('init_seed')==seed and info.get('task') is None and info.get('parent_checkpoint_sha256') is None,'Adapted checkpoint used as parent')
        require((c['segment_len'],c['memory_segment_len'],c['memory_batch_size'])==(32,8,32),'Original parent geometry differs')
    return found

def bank_audit(bank,data,args,streams):
    expected=data.build_bank(args['bank_seed'],args['contexts'],args['key_count'])
    require(bank==expected,'Saved bank differs from frozen reference generator')
    episodes=[streams.Episode.from_dict(e) for e in bank['episodes']]
    require(bank['key_pool']==list(streams.KEY_POOLS['train'][:args['key_count']]),'Bank key namespace differs')
    pairs=[tuple(e.key for e in ep.events) for ep in episodes]
    require(len(set(pairs))==len(pairs),'Duplicate ordered training key pairs')
    counts=Counter();assign=defaultdict(set);values=Counter()
    for ep in episodes:
        require(len(ep.events)==len(ep.queries)==2 and ep.events[0].value!=ep.events[1].value,'Bank must have two distinct-value facts')
        for event,query in zip(ep.events,ep.queries):
            require(event.template_index==query.template_index==0 and event.template_family==query.template_family=='seen','Noncanonical training wording')
            counts[event.key]+=1;assign[event.key].add(event.value);values[event.value]+=1
    require(set(counts)==set(bank['key_pool']) and set(counts.values())=={2*args['contexts']//args['key_count']},'Unequal key exposure')
    require(all(len(v)>=2 for v in assign.values()),'Persistent key-value shortcut in bank')
    require(set(values)==set(streams.VALUES) and len(set(values.values()))==1,'Answer letters are not balanced')
    # The per-key majority baseline is reported for interpretation of query-only.
    key_values=defaultdict(Counter)
    for ep in episodes:
        for event in ep.events:key_values[event.key][event.value]+=1
    return episodes,sum(max(v.values()) for v in key_values.values())/(2*len(episodes))

def training_audit(folder,args,bank,data,streams,meta):
    n,b=args['steps'],args['batch_size'];count=b//2
    raw=np.load(folder/'training_inputs.npz',allow_pickle=False)
    require(set(raw.files)=={'input_bytes','targets'},'Unexpected training array fields')
    x,y=raw['input_bytes'],raw['targets']
    require(x.dtype==np.uint8 and x.shape==(n,b,95) and y.dtype==np.uint8 and y.shape==(n,b),'Training array shape/dtype differs')
    plans,logs=read(folder/'training_plan.json'),csvread(folder/'training.csv')
    require(len(plans)==len(logs)==n,'Training rows/plans differ')
    totals=Counter();epochs=[]
    for step,(plan,log) in enumerate(zip(plans,logs)):
        require(plan['step']==int(log['step'])==step+1,'Training steps are not consecutive')
        if args['regime']=='fixed':
            batches=len(bank['episodes'])//count
            order=np.random.default_rng(np.random.SeedSequence([args['train_seed'],step//batches])).permutation(len(bank['episodes']))
            indices=order[(step%batches)*count:(step%batches+1)*count].tolist()
            require(plan==dict(step=step+1,bank_indices=indices),'Fixed batch order differs')
            episodes=[streams.Episode.from_dict(bank['episodes'][i]) for i in indices];epochs+=indices
            if (step+1)%batches==0:
                require(sorted(epochs)==list(range(len(bank['episodes']))),'Epoch misses/repeats a fixed context');epochs=[]
        else:
            episodes=data.fresh_training(bank,args['train_seed'],step,count)
            require(plan==dict(step=step+1,episodes=[e.to_dict() for e in episodes]),'Fresh training semantics differ')
        prefixes=[];labels=[]
        for ep in episodes:
            require(set(e.key for e in ep.events)<=set(bank['key_pool']),'Fresh training escaped fixed train keys')
            for q in ep.queries:
                prefixes.append(b''.join(e.encoded for e in ep.events)+streams.render_query(q));labels.append(q.answer_byte)
        ex=np.stack([np.frombuffer(p,dtype=np.uint8) for p in prefixes]);ey=np.asarray(labels,dtype=np.uint8)
        require(np.array_equal(x[step],ex) and np.array_equal(y[step],ey),f'Training input/label reconstruction failed step {step+1}')
        require(int(log['supervised_answers'])==b and int(log['input_bytes'])==b*95,'Training budget row differs')
        require(all(math.isfinite(float(log[k])) for k in ('loss','accuracy','gradient_norm','elapsed_seconds')),'Nonfinite training log')
        require(float(log['loss'])>=0 and 0<=float(log['accuracy'])<=1 and float(log['gradient_norm'])>=0,'Invalid training metric')
        totals['input_bytes']+=b*95;totals['supervised_answers']+=b
    require(meta['completed_steps']==n and all(meta[k]==v for k,v in totals.items()),'Run budget metadata differs')
    return x,y,dict(totals)

def group_data(bank,args,data,streams):
    train=[streams.Episode.from_dict(e) for e in bank['episodes']]
    return {('train' if args['regime']=='fixed' else 'reference_bank'):train,
        'novel_values':data.fresh_same(bank,args['dev_seed'],args['eval_episodes']),
        'heldout_keys':data.fresh_heldout(args['dev_seed']+1,args['eval_episodes']),
        'heldout_keys_templates':data.fresh_heldout(args['dev_seed']+1,args['eval_episodes'],'heldout'),'train_query_only':train}

def episode_semantic(ep):
    return (ep.seed,tuple((e.key,e.value,e.event_id) for e in ep.events),tuple((q.key,q.target) for q in ep.queries))

def evaluation_audit(folder,args,bank,data,streams,segment):
    groups=group_data(bank,args,data,streams)
    with gzip.open(folder/'episodes.json.gz','rt',encoding='utf8') as f:saved=json.load(f)
    require(saved==[dict(group=g,episodes=[e.to_dict() for e in eps]) for g,eps in groups.items()],'Saved evaluation populations differ')
    for i,ep in enumerate(groups['novel_values']):
        old=streams.Episode.from_dict(bank['episodes'][i%len(bank['episodes'])])
        require(tuple(e.key for e in ep.events)==tuple(e.key for e in old.events),'Novel evaluation changed ordered key pair')
        require(not set(e.value for e in ep.events)&set(e.value for e in old.events),'Novel evaluation reused an original pair value')
    require([episode_semantic(e) for e in groups['heldout_keys']]==[episode_semantic(e) for e in groups['heldout_keys_templates']],'Template families are not semantically paired')
    rows=csvread(folder/'predictions.csv');z=np.load(folder/'answer_logits.npy',allow_pickle=False)
    require(z.dtype==np.float32 and z.shape==(len(rows),256),'Saved logit array shape/dtype differs')
    expected=[]
    for group,episodes in groups.items():
        for cond in CONDS:
            for i,ep in enumerate(episodes):
                require(all(e.key in streams.KEY_POOLS['dev' if group.startswith('heldout') else 'train'] for e in ep.events),'Evaluation key namespace violation')
                for rank,q in enumerate(ep.queries):
                    prefix=streams.render_query(q) if group=='train_query_only' else b''.join(e.encoded for e in ep.events)+streams.render_query(q)
                    expected.append(dict(group=group,episode=i,base_pair=i%len(bank['episodes']) if group=='novel_values' else i,
                        rank=rank,condition=cond,key=q.key,target=q.answer_byte,prefix=prefix.decode('ascii')))
    require(len(expected)==len(rows),'Unexpected evaluation row count')
    require(len({(r['group'],r['condition'],r['episode'],r['rank']) for r in rows})==len(rows),'Duplicate evaluation identities')
    computed=score(z,[r['target'] for r in expected]);indexed={};pairrows=defaultdict(dict)
    for i,(actual,wanted) in enumerate(zip(rows,expected)):
        for k,v in wanted.items():require(str(actual[k])==str(v),f'Evaluation identity/label/prefix differs row {i}, {k}')
        for k in ('prediction','forced_prediction','correct'):require(int(actual[k])==int(computed[k][i]),f'Integer logit score differs row {i}, {k}')
        for k in ('nll','forced_nll','legal_probability_mass'):near(actual[k],computed[k][i],f'row {i}, {k}')
        identity=(actual['group'],actual['condition'],int(actual['episode']))
        pairrows[identity][int(actual['rank'])]=(actual,i)
        indexed[(actual['group'],actual['condition'],int(actual['episode']),int(actual['rank']))]=i
    vectors={};summaries=[]
    for group,episodes in groups.items():
        for cond in CONDS:
            pairs=[]
            for i,ep in enumerate(episodes):
                pair=pairrows[(group,cond,i)];require(set(pair)=={0,1},'Incomplete question pair')
                a,ia=pair[0];b,ib=pair[1]
                require(a['key']!=b['key'] and a['target']!=b['target'],'Strict pair requires distinct keys and labels')
                pa,pb=a['prefix'].encode(),b['prefix'].encode()
                require(len(pa)==len(pb)==(31 if group=='train_query_only' else 95),'Incorrect prefix length')
                require(pa[:-31]==pb[:-31] and pa[-31:].replace(a['key'].encode(),b'###')==pb[-31:].replace(b['key'].encode(),b'###'),'Queries differ beyond requested key')
                ca,cb=int(computed['correct'][ia]),int(computed['correct'][ib])
                pairs.append(dict(identity=episode_semantic(ep),base_pair=i%len(bank['episodes']) if group=='novel_values' else i,
                    accuracy=(ca+cb)/2,both_correct=int(ca and cb),first_accuracy=ca,second_accuracy=cb,
                    prediction_changes=int(computed['prediction'][ia]!=computed['prediction'][ib]),
                    nll=float((computed['nll'][ia]+computed['nll'][ib])/2),
                    legal_probability_mass=float((computed['legal_probability_mass'][ia]+computed['legal_probability_mass'][ib])/2)))
                if group=='novel_values':
                    original=streams.Episode.from_dict(bank['episodes'][i%len(bank['episodes'])])
                    oa=int(computed['prediction'][ia]==original.queries[0].answer_byte);ob=int(computed['prediction'][ib]==original.queries[1].answer_byte)
                    pairs[-1].update(old_assignment_accuracy=(oa+ob)/2,both_old_assignment=int(oa and ob))
            vectors[(group,cond)]=pairs
            summaries.append(dict(group=group,condition=cond,episodes=len(pairs),queries=2*len(pairs),
                **{k:float(np.mean([p[k] for p in pairs])) for k in METRICS}))
        if segment==128 or group=='train_query_only':
            for i,ep in enumerate(episodes):
                for rank in (0,1):require(np.array_equal(z[indexed[(group,'online',i,rank)]],z[indexed[(group,'reset_memory',i,rank)]]),'Reset should be an exact logit no-op')
    saved_summary=read(folder/'summary.json')
    require(len(saved_summary)==len(summaries),'Evaluation summary row count differs')
    for actual,wanted in zip(saved_summary,summaries):
        for k,v in wanted.items():
            if k in METRICS:near(actual[k],v,f'Summary {wanted["group"]}/{wanted["condition"]}/{k}')
            else:require(actual[k]==v,f'Summary identity differs: {k}')
    return vectors,summaries

def stage_audit(folder,meta,args,parent,design,bank,data,streams):
    import torch
    initial=load_tensors(folder/'initial_state.pt');same_state(initial,parent[1]['model_state'],f'{folder.name} original initialization')
    require(meta['checkpoint_sha256']==design['parents'][str(meta['parent_init_seed'])],'Parent checkpoint hash differs')
    stages=(0,args['steps']//10,args['steps']//2,args['steps'])
    records=meta['checkpoints'];require([r['step'] for r in records]==list(stages),'Checkpoint schedule differs')
    result={};all_summaries=[]
    for step,record in zip(stages,records):
        stage=folder/f'step-{step:04d}';payload=load_tensors(stage/'checkpoint.pt');digest=sha(stage/'checkpoint.pt')
        sm=read(stage/'metadata.json')
        require(digest==record['checkpoint_sha256']==sm['checkpoint_sha256'] and sm['step']==step,'Checkpoint digest/step differs')
        require(payload['config']==dict(parent[1]['config'],segment_len=args['segment_len']),'Checkpoint geometry differs')
        require(payload['format_version']==parent[1]['format_version'] and payload['architecture']==parent[1]['architecture'],'Checkpoint format differs')
        state=payload['model_state'];require(set(state)==set(initial),'Checkpoint state keys differ')
        for key,tensor in state.items():require(isinstance(tensor,torch.Tensor) and torch.isfinite(tensor).all().item() and tensor.shape==initial[key].shape and tensor.dtype==initial[key].dtype,f'Invalid checkpoint tensor {key}')
        if step==0:same_state(state,initial,f'{folder.name} step0')
        info=dict(task='fixed-set-generalization',parent_init_seed=meta['parent_init_seed'],parent_checkpoint_sha256=meta['checkpoint_sha256'],steps=step,
            regime=args['regime'],bank_seed=args['bank_seed'],contexts=args['contexts'],key_count=args['key_count'],train_seed=args['train_seed'],lr=args['lr'],fixed_final_checkpoint=step==args['steps'])
        require(payload['training_info']==info,'Checkpoint training metadata differs')
        prov=payload['source_info'];require(prov['repo_commit']==meta['repo_commit'] and prov['source_sha256']==meta['source_sha256'],'Checkpoint source provenance differs')
        opt=load_tensors(stage/'optimizer.pt');ids=[]
        for group in opt['param_groups']:
            require(group['lr']==args['lr'] and group['weight_decay']==.01 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8,'Optimizer settings differ')
            ids+=group['params']
        require(len(ids)==len(set(ids)),'Duplicate optimizer parameter IDs')
        if step==0:require(not opt['state'],'Step0 evaluation advanced optimizer')
        else:require(bool(opt['state']),'Missing optimizer state')
        for item in opt['state'].values():
            require(int(item['step'])==step,'Optimizer counter differs')
            for k in ('exp_avg','exp_avg_sq'):require(torch.isfinite(item[k]).all().item(),'Nonfinite optimizer state')
        vec,summaries=evaluation_audit(stage/'evaluation',args,bank,data,streams,args['segment_len'])
        require(record['summary']==sm['summary']==read(stage/'evaluation/summary.json'),'Checkpoint summary metadata copies differ')
        for actual,wanted in zip(record['summary'],summaries):
            for k in METRICS:near(actual[k],wanted[k],f'Checkpoint summary metadata {k}')
        result[step]=vec
        all_summaries.extend(dict(run=folder.name,parent=meta['parent_init_seed'],segment=args['segment_len'],regime=args['regime'],step=step,**s) for s in summaries)
    return result,all_summaries

def run_audit(root,name,parents,design,commit,repo,manifest=None):
    folder=root/name;meta=read(folder/'metadata.json');args=meta['args'];tiny=name.startswith('tiny');fresh=name.startswith('fresh')
    seed=int(name.rsplit('-',1)[1]);segment=128 if '128' in name else 32
    expected=dict(regime='fresh' if fresh else 'fixed',segment_len=segment,contexts=8 if tiny else 32,key_count=4 if tiny else 16,
        steps=1600 if tiny else 2000,batch_size=16,bank_seed=90201,train_seed=90202,dev_seed=90203,eval_episodes=128,eval_batch_size=16,lr=.0003,max_seconds=600)
    for k,v in expected.items():require(args[k]==v,f'Run argument differs {name}/{k}: {args[k]} != {v}')
    require(args['mode']==meta['mode']=='fixed_set_generalization' and meta['parent_init_seed']==seed,'Run mode/parent differs')
    require(args['device'].startswith('cuda') and meta['environment']['device'].startswith('cuda'),'Main run must use allocated CUDA device')
    require(meta['environment']['deterministic_algorithms'] is True and meta['environment']['tf32'] is False and meta['environment']['dtype']=='float32','Numerical execution settings differ')
    require(meta['elapsed_seconds']<=600,'Per-run wall time exceeded')
    require(meta['design_sha256']==sha(root/'design.json')==sha(folder/'design.json'),'Run design differs')
    current=source_audit(folder,meta,commit,repo)
    if manifest is not None:require(current==manifest,'Runs used different source manifests')
    streams,data=data_modules(folder)
    require(meta['key_pool_sha256']==streams.KEY_POOL_HASHES,'Key pool hash metadata differs')
    bank=read(folder/'bank.json');_,ceiling=bank_audit(bank,data,args,streams)
    x,y,budgets=training_audit(folder,args,bank,data,streams,meta)
    vectors,summaries=stage_audit(folder,meta,args,parents[seed],design,bank,data,streams)
    return dict(meta=meta,args=args,bank=bank,vectors=vectors,summaries=summaries,source=current,x=x,y=y,budgets=budgets,query_only_majority_ceiling=ceiling)

def unit_vector(pairs,metric,group):
    if group=='novel_values':
        blocks=defaultdict(list)
        for p in pairs:blocks[p['base_pair']].append(p[metric])
        require(len({len(v) for v in blocks.values()})==1,'Unbalanced novel-value clusters')
        return np.array([np.mean(blocks[k]) for k in sorted(blocks)],dtype=float)
    return np.array([p[metric] for p in pairs],dtype=float)

def compare(left,right,metric,group,draws,seed):
    require(len(left)==len(right) and all(a['identity']==b['identity'] and a['base_pair']==b['base_pair'] for a,b in zip(left,right)),'Comparison semantics are not paired')
    a,b=unit_vector(left,metric,group),unit_vector(right,metric,group)
    lo,hi=bootstrap(a-b,draws,seed)
    return dict(units=len(a),left=float(a.mean()),right=float(b.mean()),difference=float((a-b).mean()),lower=lo,upper=hi)

def reconstruct_disjoint(bank,data,streams,excluded):
    require(len(bank['episodes'])==32,'Disjoint selection requires 32 base pairs')
    originals=[streams.Episode.from_dict(d) for d in bank['episodes']]
    accepted=[[] for _ in range(32)];seen=[set() for _ in range(32)]
    counters=dict(visited=0,rejected_training=0,rejected_duplicate=0,skipped_completed_base_pair=0)
    for source_index,ep in enumerate(data.fresh_same(bank,90205,8192)):
        counters['visited']+=1;base=source_index%32
        if len(accepted[base])==4:counters['skipped_completed_base_pair']+=1;continue
        original=originals[base]
        require(tuple(e.key for e in ep.events)==tuple(e.key for e in original.events),'Disjoint candidate changed ordered keys')
        values={e.value for e in ep.events};old={e.value for e in original.events}
        require(len(values)==2 and not values&old,'Disjoint candidate reused old values')
        context=b''.join(e.encoded for e in ep.events)
        if context in excluded:counters['rejected_training']+=1;continue
        if context in seen[base]:counters['rejected_duplicate']+=1;continue
        seen[base].add(context)
        accepted[base].append(dict(base_pair=base,variant=len(accepted[base]),source_candidate_index=source_index,
            context_sha256=hashlib.sha256(context).hexdigest(),episode=ep.to_dict()))
        if all(len(a)==4 for a in accepted):break
    require(all(len(a)==4 for a in accepted),'Independent disjoint candidate selection exhausted')
    selected=[e for base in accepted for e in base]
    contexts=[b''.join(e.encoded for e in streams.Episode.from_dict(r['episode']).events) for r in selected]
    require(len(set(contexts))==128 and not set(contexts)&excluded,'Disjoint evaluation overlaps training or duplicates contexts')
    counters.update(accepted=128,candidates_generated=8192,excluded_unique_contexts=len(excluded),accepted_unique_contexts=128)
    return selected,counters,contexts

def check_disjoint_selection(saved,selection,selected,counters):
    require(saved['episodes']==selected and saved['selection']==selection,'Disjoint saved selection differs')
    for k,v in dict(seed=90205,candidate_cap=8192,variants_per_base_pair=4,base_pairs=32,**counters).items():
        require(selection[k]==v,f'Disjoint selection counter differs: {k}')

def disjoint_audit(root,runs,manifest,commit,draws,bootstrap_seed):
    """Independent candidate rejection and scoring; never import the probe script."""
    folder=root/'disjoint-transfer';meta=read(folder/'metadata.json')
    names=('fixed32-68101','fixed128-68101','fresh32-68101','fresh128-68101')
    amendment=read(root/'transfer-amendment.json');spec=amendment['new_probe']
    require(amendment['amendment_id']=='fresh-assignment-disjoint-probe-v1' and spec['seed']==90205 and spec['base_pairs']==32
        and spec['assignments_per_pair']==4 and spec['total_episodes']==128 and spec['candidate_limit']==8192 and tuple(spec['models'])==names,'Disjoint amendment differs')
    require(meta['status']=='succeeded' and meta['mode']=='training-disjoint-novel-values-amendment' and meta['repo_commit']==commit,'Disjoint run incomplete or source revision differs')
    require(meta['args']['device']=='cuda' and 0<meta['args']['max_seconds']<=300 and meta['elapsed_seconds']<=300,'Disjoint resource limit differs')
    require(meta['branch_sha256']==sha(root/'branch.json'),'Disjoint branch provenance differs')
    require(meta['evaluation']==dict(runs=list(names),parent_init_seed=68101,final_step=2000,conditions=list(CONDS),batch_size=16,
        episodes=128,queries_per_run_condition=256,output_vocabulary=256),'Disjoint evaluation declaration differs')
    hashes=meta['source_sha256'];require(hashes['repo']==manifest,'Disjoint model source hashes differ')
    for name,digest in manifest.items():require(sha(folder/'sources/repo'/name)==digest,f'Disjoint frozen source copy differs {name}')
    require(sha(folder/'sources/disjoint_probe.py')==hashes['disjoint_probe']=='4415b084752467f6f3a13b4ab8fad6bc9e3f1dcda339621049f31e678ae178c6','Disjoint own frozen source hash differs')
    require(meta['fresh_training_arrays_identical'] is True and meta['banks_identical'] is True,'Disjoint input invariant record differs')
    require(set(meta['run_artifacts'])==set(names) and set(meta['evaluation_checks'])==set(names),'Disjoint evaluated run set differs')
    for name in names:
        artifact=meta['run_artifacts'][name];base=root/name
        for key,path in (('checkpoint_sha256',base/'step-2000/checkpoint.pt'),('training_npz_sha256',base/'training_inputs.npz'),('bank_sha256',base/'bank.json'),('metadata_sha256',base/'metadata.json')):
            require(artifact[key]==sha(path),f'Disjoint input hash differs {name}/{key}')
        require(artifact['parent_checkpoint_sha256']==runs[name]['meta']['checkpoint_sha256'] and artifact['source_sha256']==manifest,'Disjoint parent/source artifact differs')
        require(artifact['segment_len']==runs[name]['args']['segment_len'] and artifact['regime']==runs[name]['args']['regime'],'Disjoint model geometry/regime declaration differs')
        require(all(meta['evaluation_checks'][name].get(k) is True for k in ('outer_checkpoint_tensors_unchanged','training_rng_unchanged','checkpoint_file_unchanged')),'Disjoint execution invariant failed')
        require(runs[name]['bank']==runs[names[0]]['bank'],'Disjoint endpoints use different banks')
    streams,data=data_modules(root/names[0]);bank=runs[names[0]]['bank']
    excluded={bytes(context[:64]) for name in ('fresh32-68101','fresh128-68101') for batch in runs[name]['x'] for context in batch}
    selected,counters,contexts=reconstruct_disjoint(bank,data,streams,excluded)
    saved=read(folder/'episodes.json');check_disjoint_selection(saved,meta['selection'],selected,counters)
    require(meta['episodes_sha256']==sha(folder/'episodes.json'),'Disjoint episode artifact hash differs')
    expected=[]
    for name in names:
        for condition in CONDS:
            for index,record in enumerate(selected):
                ep=streams.Episode.from_dict(record['episode'])
                for rank,q in enumerate(ep.queries):
                    expected.append(dict(run=name,condition=condition,episode=index,base_pair=record['base_pair'],variant=record['variant'],
                        source_candidate_index=record['source_candidate_index'],rank=rank,key=q.key,prefix=(contexts[index]+streams.render_query(q)).decode('ascii'),target=q.answer_byte))
    rows=csvread(folder/'predictions.csv');z=np.load(folder/'answer_logits.npy',allow_pickle=False)
    require(meta['evaluation_rows']==len(rows)==len(expected)==2048 and z.dtype==np.float32 and z.shape==(2048,256),'Disjoint logit/row shape differs')
    require(meta['answer_logits_sha256']==sha(folder/'answer_logits.npy') and meta['predictions_sha256']==sha(folder/'predictions.csv'),'Disjoint output hashes differ')
    computed=score(z,[r['target'] for r in expected]);computed['forced_correct']=(computed['forced_prediction']==np.array([r['target'] for r in expected])).astype(int)
    index={};pairs=defaultdict(dict)
    for i,(actual,wanted) in enumerate(zip(rows,expected)):
        for k,v in wanted.items():require(str(actual[k])==str(v),f'Disjoint identity/input/target differs {i}/{k}')
        for k in ('prediction','forced_prediction','correct','forced_correct'):require(int(actual[k])==int(computed[k][i]),f'Disjoint score differs {i}/{k}')
        for k in ('nll','forced_nll','legal_probability_mass'):near(actual[k],computed[k][i],f'Disjoint score {i}/{k}')
        key=(actual['run'],actual['condition'],int(actual['episode']),int(actual['rank']))
        require(key not in index,'Duplicate disjoint prediction identity');index[key]=i
        pairs[key[:3]][key[3]]=i
    vectors={};summaries=[]
    for name in names:
        for cond in CONDS:
            values=[]
            for i,record in enumerate(selected):
                pair=pairs[(name,cond,i)];require(set(pair)=={0,1},'Incomplete disjoint question pair');a,b=pair[0],pair[1]
                ca,cb=int(computed['correct'][a]),int(computed['correct'][b]);fa,fb=int(computed['forced_correct'][a]),int(computed['forced_correct'][b])
                ep=streams.Episode.from_dict(record['episode'])
                values.append(dict(identity=episode_semantic(ep),base_pair=record['base_pair'],accuracy=(ca+cb)/2,both_correct=int(ca and cb),
                    forced_accuracy=(fa+fb)/2,forced_both_correct=int(fa and fb),first_accuracy=ca,second_accuracy=cb,
                    prediction_changes=int(computed['prediction'][a]!=computed['prediction'][b]),nll=float((computed['nll'][a]+computed['nll'][b])/2),
                    forced_nll=float((computed['forced_nll'][a]+computed['forced_nll'][b])/2),
                    legal_probability_mass=float((computed['legal_probability_mass'][a]+computed['legal_probability_mass'][b])/2)))
                original=streams.Episode.from_dict(bank['episodes'][record['base_pair']])
                oa=int(computed['prediction'][a]==original.queries[0].answer_byte);ob=int(computed['prediction'][b]==original.queries[1].answer_byte)
                values[-1].update(old_assignment_accuracy=(oa+ob)/2,both_old_assignment=int(oa and ob))
            vectors[(name,cond)]=values
            summary=dict(run=name,condition=cond,episodes=128,queries=256,**{k:float(np.mean([p[k] for p in values])) for k in (*METRICS,'forced_accuracy','forced_both_correct','forced_nll')})
            lo,hi=bootstrap(unit_vector(values,'both_correct','novel_values'),draws,bootstrap_seed)
            summary.update(both_count=sum(p['both_correct'] for p in values),bootstrap_units=32,both_lower=lo,both_upper=hi)
            summary.update(posthoc_old_assignment_accuracy=float(np.mean([p['old_assignment_accuracy'] for p in values])),posthoc_both_old_assignment=float(np.mean([p['both_old_assignment'] for p in values])))
            summaries.append(summary)
        if '128-' in name:
            for i in range(128):
                for rank in (0,1):require(np.array_equal(z[index[(name,'online',i,rank)]],z[index[(name,'reset_memory',i,rank)]]),'Disjoint 128 reset is not exact no-op')
    original_summary=read(folder/'summary.json');require(original_summary==meta['summary'],'Disjoint summary metadata copy differs')
    require(len(original_summary)==8,'Disjoint summary length differs')
    csv_summary=csvread(folder/'summary.csv');require(len(csv_summary)==8,'Disjoint CSV summary length differs')
    for original,wanted,csvrow in zip(original_summary,summaries,csv_summary):
        require(set(original)==({'run','condition','episodes','queries'} | (set(METRICS)-{'prediction_changes'}) | {'forced_accuracy','forced_both_correct','forced_nll'}),'Disjoint reported metric fields differ')
        for k,v in original.items():
            if k in ('run','condition','episodes','queries'):require(v==wanted[k] and str(csvrow[k])==str(v),'Disjoint summary identity differs')
            else:near(v,wanted[k],f'Disjoint summary {k}');near(csvrow[k],wanted[k],f'Disjoint CSV summary {k}')
    differences=[]
    for left,right in (('fresh32-68101','fixed32-68101'),('fresh128-68101','fixed128-68101'),('fixed128-68101','fixed32-68101'),('fresh128-68101','fresh32-68101')):
        for cond in CONDS:
            d=compare(vectors[(left,cond)],vectors[(right,cond)],'both_correct','novel_values',draws,bootstrap_seed)
            differences.append(dict(left_run=left,right_run=right,condition=cond,metric='both_correct',**d))
    for name in names:
        d=compare(vectors[(name,'online')],vectors[(name,'reset_memory')],'both_correct','novel_values',draws,bootstrap_seed)
        differences.append(dict(left_run=name,right_run=name,condition='online-minus-reset',metric='both_correct',**d))
    return dict(summaries=summaries,differences=differences,selection=meta['selection'],
        provenance=dict(amendment_sha256=sha(root/'transfer-amendment.json'),probe_source_sha256=hashes['disjoint_probe'],metadata_sha256=sha(folder/'metadata.json'),evaluated_rows=2048))

def summary_markdown(summaries,decisions,budgets,overlaps,trigger,disjoint,verification):
    def row(name,group):return next(r for r in summaries if r['run']==name and r['group']==group and r['condition']=='online' and r['step']==2000)
    def count(r):return f"{r['both_count']}/{r['episodes']} ({100*r['both_correct']:.2f}%)"
    def interval(r):return f"[{100*r['both_lower']:.2f}, {100*r['both_upper']:.2f}]"
    lines=['# Audited fixed-set fitting and assignment transfer','',
        f"Saved-artifact audit: **verified**. Source `{verification['source_commit']}`. This recomputes persisted evidence without reproducing training or model inference.",'',
        'All four fixed runs retain their original parent, data, optimizer budget and final step-2000 endpoint. Both queries are scored against all 256 byte logits. Earlier stages remain descriptive.','',
        '| Fixed run | Exact train both correct | Novel assignments both correct | Novel 95% interval | Heldout keys both correct | Heldout keys + wording both correct | Query-only accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in MAIN:
        train,novel,held,word,query=(row(name,g) for g in GROUPS)
        lines.append(f"| {name} | {count(train)} | {count(novel)} | {interval(novel)} | {count(held)} | {count(word)} | {100*query['accuracy']:.2f}% |")
    lines+=['',f"The frozen primary trigger is `fixed128-68101/step-2000`: {trigger['train_both_count']}/32 exact-bank pairs and {trigger['novel_both_count']}/128 novel-assignment pairs. It selects the **{trigger['branch']}** branch using the declared 31/32 fitting and 103/128 transfer counts.",'',
        'The novel assignments use the same ordered key pairs while excluding both original letters. This is a deliberate value shift. The widened window exposes all 95 input bytes to attention and retains the neural-memory path. Joint heldout wording does not isolate fact wording from query wording.','']
    if disjoint:
        lines+=['## Exploratory fresh-training branch','',
            'Fresh training restarts both geometries from the original primary parent and changes the assignment distribution. Its reference bank is an evaluation population, not its exact fixed training set. The amended probe contains 128 distinct contexts absent from both complete fresh training histories and uses identical examples for all four endpoints.','',
            '| Training-disjoint final endpoint | Both correct | Cluster 95% interval | First-key accuracy | Second-key accuracy | Posthoc original-bank answer rate |',
            '|---|---:|---:|---:|---:|---:|']
        for r in disjoint['summaries']:
            if r['condition']!='online':continue
            lines.append(f"| {r['run']} | {count(r)} | {interval(r)} | {100*r['first_accuracy']:.2f}% | {100*r['second_accuracy']:.2f}% | {100*r['posthoc_old_assignment_accuracy']:.2f}% |")
        lines+=['','Paired fresh-minus-fixed effects on these training-disjoint examples:']
        for d in disjoint['differences']:
            if d['condition']=='online' and d['left_run'].startswith('fresh') and d['right_run'].startswith('fixed'):
                lines.append(f"- {d['left_run']} minus {d['right_run']}: {100*d['difference']:+.2f} percentage points [{100*d['lower']:.2f}, {100*d['upper']:.2f}].")
        lines+=['','The original fresh-regime `novel_values` population is retained descriptively because exact training-context overlap can occur:']
        for r in overlaps:
            if r['run'].startswith('fresh') and r['group']=='novel_values':lines.append(f"- {r['run']}: {r['exact_training_context_seen']}/{r['episodes']} contexts seen in training; {r['unseen_episodes']} unseen contexts remain.")
        lines+=['','The original-bank answer rates are posthoc descriptions of incorrect old labels, since both old letters are absent from each new context. They do not establish an internal retrieval or memorization mechanism.','']
    elif trigger['branch']=='tiny':
        r=next(d for d in decisions if d['run']=='tiny128-68101')
        lines+=['## Tiny fitting control','',f"The eight-context, four-key control reaches {r['bank_both_count']}/{r['bank_episodes']} both-correct pairs at its fixed step-1600 endpoint. Every key appears four times with conflicting assigned values. This is a finite-budget optimization/complexity check.",'']
    lines+=['## Evidence limits','',
        'Bootstrap intervals use 5,000 draws with seed 90204. Novel values resample complete base-pair clusters and retain their assignment replicates; other populations use complete two-question episodes. Parent checkpoints are reported separately. Train-bank intervals are descriptive and zero-event bootstrap intervals do not imply zero population error.','',
        'A successful exact-bank fit demonstrates finite-set trainability under this recipe. Failed transfer establishes a limitation of the tested training procedure, not architecture impossibility. Exact-set fitting or same-key transfer alone does not qualify the project retention baseline; revision, causal-state/reset and session requirements remain unresolved.','',
        f"The audit checks {verification['checked_training_answers']:,} supervised training answers, {verification['checked_training_input_bytes']:,} training input bytes and {verification['checked_evaluation_answers']:,} saved evaluation logit vectors across {len(verification['runs'])} runs. All original initial states, declared checkpoints/optimizer counters, source hashes, matching budgets, and required 128-window exact reset no-ops pass.",'']
    return '\n'.join(lines)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--parents',type=Path,required=True)
    p.add_argument('--expected-commit',required=True);p.add_argument('--repo',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv);root=args.root.resolve();out=args.out.resolve()
    require(out.is_relative_to((root/'analysis').resolve()) and not out.exists(),'Output must be new under experiment/analysis')
    design=read(root/'design.json');require(tuple(r['name'] for r in design['main_runs'])==MAIN,'Main run declaration differs')
    parents=parents_at(args.parents,design);runs={};manifest=None
    for name in MAIN:
        runs[name]=run_audit(root,name,parents,design,args.expected_commit,args.repo,manifest)
        if manifest is None:manifest=runs[name]['source']
    for name in MAIN[1:]:
        require(runs[name]['bank']==runs[MAIN[0]]['bank'],'Main banks differ')
        require(np.array_equal(runs[name]['x'],runs[MAIN[0]]['x']) and np.array_equal(runs[name]['y'],runs[MAIN[0]]['y']),'Main training bytes/targets not matched')
    primary=runs['fixed128-68101']['vectors'][2000]
    fit=sum(p['both_correct'] for p in primary[('train','online')]);novel=sum(p['both_correct'] for p in primary[('novel_values','online')])
    branch='tiny' if fit<31 else ('fresh' if novel<103 else 'none')
    extra=('tiny128-68101',) if branch=='tiny' else (('fresh32-68101','fresh128-68101') if branch=='fresh' else ())
    for name in extra:runs[name]=run_audit(root,name,parents,design,args.expected_commit,args.repo,manifest)
    for name,result in runs.items():
        require(all(result['meta']['environment'][k]==runs[MAIN[0]]['meta']['environment'][k] for k in ('python','torch','numpy','device_name','dtype','deterministic_algorithms','tf32')),'Run environment differs')
        if not name.startswith('tiny'):require(result['bank']==runs[MAIN[0]]['bank'],'Main/fresh fixed reference banks differ')
    for name in ('fresh32-68101','fresh128-68101','tiny128-68101'):
        require(name in extra or not (root/name).exists(),f'Unexpected conditional run directory {name}')
    if branch=='fresh':require(np.array_equal(runs[extra[0]]['x'],runs[extra[1]]['x']) and np.array_equal(runs[extra[0]]['y'],runs[extra[1]]['y']),'Fresh 32/128 training bytes are not matched')
    require((root/'branch.json').is_file(),'Missing pipeline branch decision')
    branch_record=read(root/'branch.json')
    require(branch_record['selected_from']=='fixed128-68101/step-2000' and branch_record['fit_threshold']==.95 and branch_record['transfer_threshold']==.8,'Pipeline branch source/threshold differs')
    near(branch_record['fit_both_correct'],fit/32,'Pipeline fitting trigger')
    near(branch_record['novel_values_both_correct'],novel/128,'Pipeline transfer trigger')
    require(branch_record['choice']==branch and bool(branch_record['recorded_utc']),'Pipeline branch decision differs')
    require((root/'pipeline.json').is_file(),'Missing pipeline record')
    pipeline=read(root/'pipeline.json')
    require(pipeline['branch']==branch_record,'Pipeline and branch.json decision copies differ')
    require(pipeline['status']=='succeeded' and [r['name'] for r in pipeline['runs']]==list(MAIN+extra),'Pipeline status/run set differs')
    require(all(r['status']=='succeeded' and r['returncode']==0 and r['elapsed_seconds']<=660 for r in pipeline['runs']),'Pipeline reports failed/over-budget run')
    require((datetime.fromisoformat(pipeline['ended_utc'])-datetime.fromisoformat(pipeline['started_utc'])).total_seconds()<=3300,'Pipeline wall time exceeded')
    # Keep the original record alongside the independently reconstructed trigger.
    trigger=dict(primary='fixed128-68101',step=2000,train_both_count=fit,train_episodes=32,novel_both_count=novel,novel_episodes=128,branch=branch,expected_extra_runs=list(extra),pipeline_record=branch_record)
    summaries=[];differences=[];decisions=[];budgets=[];overlaps=[]
    draws=design['analysis']['bootstrap_draws'];seed=design['analysis']['bootstrap_seed']
    for name,result in runs.items():
        for row in result['summaries']:
            pairs=result['vectors'][row['step']][(row['group'],row['condition'])]
            v=unit_vector(pairs,'both_correct',row['group']);lo,hi=bootstrap(v,draws,seed)
            # Train-set intervals are descriptive empirical resampling, never a generalization estimate.
            summaries.append(dict(**row,both_count=int(sum(p['both_correct'] for p in pairs)),bootstrap_units=len(v),both_lower=lo,both_upper=hi,
                posthoc_old_assignment_accuracy=(float(np.mean([p['old_assignment_accuracy'] for p in pairs])) if row['group']=='novel_values' else None),
                posthoc_both_old_assignment=(float(np.mean([p['both_old_assignment'] for p in pairs])) if row['group']=='novel_values' else None)))
        final=result['vectors'][result['args']['steps']]
        bank_group='train' if result['args']['regime']=='fixed' else 'reference_bank'
        bank_pairs=final[(bank_group,'online')]
        decisions.append(dict(run=name,parent=result['meta']['parent_init_seed'],segment=result['args']['segment_len'],
            regime=result['args']['regime'],bank_group=bank_group,bank_episodes=len(bank_pairs),bank_both_count=sum(p['both_correct'] for p in bank_pairs),
            bank_both_correct=float(np.mean([p['both_correct'] for p in bank_pairs])),
            fixed_fit_gate_at_0_95=(bool(np.mean([p['both_correct'] for p in bank_pairs])>=.95) if bank_group=='train' else None),
            novel_both_count=sum(p['both_correct'] for p in final[('novel_values','online')]),
            novel_both_correct=float(np.mean([p['both_correct'] for p in final[('novel_values','online')]])),
            assignment_transfer_at_0_8=bool(np.mean([p['both_correct'] for p in final[('novel_values','online')]])>=.8),
            query_only_per_key_majority_ceiling=result['query_only_majority_ceiling']))
        process=next(r for r in pipeline['runs'] if r['name']==name)
        budgets.append(dict(run=name,steps=result['args']['steps'],**result['budgets'],runner_elapsed_seconds=result['meta']['elapsed_seconds'],process_elapsed_seconds=process['elapsed_seconds']))
        # Model-visible context overlap ignores episode seeds, which are metadata.
        trained_contexts={bytes(context[:64]) for batch in result['x'] for context in batch[::2]}
        for group,cond in final:
            if cond!='online':continue
            pairs=final[(group,cond)]
            seen=[]
            for pair in pairs:
                events=pair['identity'][1]
                # Canonical wording is reconstructed independently of the seed.
                context=b''.join((f'fact {key} has value {value}.\n'.encode('ascii').rjust(32,b' ')) for key,value,event_id in events)
                seen.append(context in trained_contexts)
            correct=np.array([p['both_correct'] for p in pairs],dtype=float);seen=np.array(seen,dtype=bool)
            overlaps.append(dict(run=name,group=group,episodes=len(pairs),exact_training_context_seen=int(seen.sum()),
                seen_both_correct=(float(correct[seen].mean()) if seen.any() else None),
                unseen_episodes=int((~seen).sum()),unseen_both_correct=(float(correct[~seen].mean()) if (~seen).any() else None)))
        if result['args']['segment_len']==32:
            for step,vectors in result['vectors'].items():
                for group,cond in vectors:
                    if cond!='online':continue
                    d=compare(vectors[(group,'online')],vectors[(group,'reset_memory')],'both_correct',group,draws,seed)
                    differences.append(dict(comparison='online-minus-reset',left_run=name,right_run=name,step=step,group=group,condition='paired',metric='both_correct',**d))
    for parent in (68101,68105):
        left,right=f'fixed128-{parent}',f'fixed32-{parent}'
        for step in (0,200,1000,2000):
            for group in GROUPS:
                for cond in CONDS:
                    d=compare(runs[left]['vectors'][step][(group,cond)],runs[right]['vectors'][step][(group,cond)],'both_correct',group,draws,seed)
                    differences.append(dict(comparison='128-minus-32',left_run=left,right_run=right,step=step,group=group,condition=cond,metric='both_correct',**d))
    if branch=='fresh':
        for segment in (32,128):
            left,right=f'fresh{segment}-68101',f'fixed{segment}-68101'
            for group in GROUPS:
                for cond in CONDS:
                    left_group='reference_bank' if group=='train' else group
                    d=compare(runs[left]['vectors'][2000][(left_group,cond)],runs[right]['vectors'][2000][(group,cond)],'both_correct',group,draws,seed)
                    differences.append(dict(comparison='fresh-minus-fixed',left_run=left,right_run=right,step=2000,group=left_group,condition=cond,metric='both_correct',**d))
    disjoint=disjoint_audit(root,runs,manifest,args.expected_commit,draws,seed) if branch=='fresh' else None
    out.mkdir(parents=True)
    csvwrite(out/'metrics.csv',summaries);csvwrite(out/'paired_differences.csv',differences);csvwrite(out/'decisions.csv',decisions);csvwrite(out/'budgets.csv',budgets)
    csvwrite(out/'training_context_overlap.csv',overlaps)
    if disjoint:
        csvwrite(out/'disjoint_metrics.csv',disjoint['summaries']);csvwrite(out/'disjoint_differences.csv',disjoint['differences'])
        write(out/'disjoint_selection.json',dict(selection=disjoint['selection'],provenance=disjoint['provenance']))
    write(out/'branch_reconstructed.json',trigger)
    verification=dict(status='verified',scope='Saved-artifact integrity and independently recomputed scoring; does not execute model inference or reproduce training.',
        experiment=design['experiment_id'],source_commit=args.expected_commit,design_sha256=sha(root/'design.json'),runs=list(runs),
        checked_training_answers=sum(b['supervised_answers'] for b in budgets),checked_training_input_bytes=sum(b['input_bytes'] for b in budgets),
        checked_evaluation_answers=sum(row['queries'] for row in summaries)+(2048 if disjoint else 0),bootstrap_draws=draws,bootstrap_seed=seed,
        novel_value_unit='Base ordered key pairs, retaining all four assignment replicates (tiny control has sixteen replicates per base pair).',
        branch=branch,source_sha256=manifest,auditor_sha256=sha(__file__),disjoint_probe=(disjoint['provenance'] if disjoint else None),
        posthoc_metrics=dict(original_bank_target_prediction='Requested after main fitting/transfer outcomes; descriptive per-query and both-old rates, not a primary gate or internal-mechanism inference.'),
        parents={str(k):dict(path=str(v[0]),sha256=sha(v[0])) for k,v in parents.items()},
        limitations=['Fixed-checkpoint development evidence; no training-seed variability inference or reserved test access.',
        'Zero-event empirical bootstrap intervals do not establish zero population error or equivalence.',
        'Exact-set fitting and same-key transfer alone do not qualify the retention baseline.'])
    write(out/'verification.json',verification)
    (out/'summary.md').write_text(summary_markdown(summaries,decisions,budgets,overlaps,trigger,disjoint,verification),encoding='utf8')
    print(json.dumps(dict(status='verified',out=str(out),runs=list(runs),branch=branch,decisions=decisions),indent=2))

if __name__=='__main__':main()
