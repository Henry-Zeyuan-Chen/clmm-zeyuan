"""Frozen-weight fact/query format factorial; no optimizer or parameter training."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,os,shutil,subprocess,sys,time,traceback
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
import torch
from clmm.pretrain import load_checkpoint,build_model
from clmm.pretrain.format_factorial import TASKS,CELLS,make_codec,fresh_worlds,samples
from clmm.pretrain.record_selection import collect_variant_signatures
from scripts.selection_experiment import score_logits
from scripts.sequence_freshness_experiment import state_sha


def read(p):
    with (gzip.open(p,'rt',encoding='utf8') if p.suffix=='.gz' else p.open(encoding='utf8')) as f:return json.load(f)
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(p,'wt',encoding='utf8') if p.suffix=='.gz' else p.open('w',encoding='utf8')) as f:json.dump(v,f,allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def csvfile(p,rows):
    with p.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False);o=a.out;codec=make_codec()
    prior=set(read(a.exclusions));assert len(prior)==789300
    engineering=fresh_worlds(prior,seed=91813,n_train=8,n_dev=8,codec=codec)
    engineering_ids=collect_variant_signatures(engineering)
    main=fresh_worlds(prior|engineering_ids,seed=91823,n_train=32,n_dev=128,codec=codec)
    assert len(collect_variant_signatures(main))==640
    write(o/'engineering_worlds.json',engineering);write(o/'main_worlds.json',main)
    for name,worlds in [('engineering',engineering),('main',main)]:
        rows=samples(worlds,codec)
        for i,row in enumerate(rows):row['packet_row_index']=i
        write(o/f'{name}_samples.json.gz',rows)
    write(o/'key_codec.json',codec.to_dict());shutil.copy2(a.exclusions,o/'prior_exclusions.json.gz');shutil.copy2(a.protocol,o/'protocol.json')
    registry=read(a.registry);cases=[];model_files=[];(o/'weights').mkdir()
    for m in registry['models']:
        allowed=(m['phase']=='main' and m['trained_task'] in ('select','decode')) or (m['phase']=='bridge' and m['trained_task']=='value')
        if not allowed:continue
        source=Path(m['local_checkpoint']);digest=sha(source);assert digest==m['checkpoint_sha256']
        target=o/'weights'/f'{digest}.pt';shutil.copy2(source,target);model_files.append(target)
        cases.append(dict(id=f"trained-{m['parent_seed']}-{m['encoding']}-{m['trained_task']}",stage='trained',parent=m['parent_seed'],encoding=m['encoding'],task=m['trained_task'],
            checkpoint=target.relative_to(o).as_posix(),checkpoint_sha256=digest,state_sha256=m['state_sha256'],training_updates=m['updates'],
            inherited_selector_updates=m['inherited_selector_updates'],original_name=m['name'],model_source_commit=registry['source_commit']))
    assert len(cases)==12
    for parent in [68101,68105]:
        source=a.parents/f'lm-{parent}/checkpoint.pt';target=o/'weights'/f'initial-{parent}.pt';shutil.copy2(source,target);model_files.append(target)
        model,cfg,payload=load_checkpoint(target,map_location='cpu');assert sum(p.numel() for p in model.parameters())==839248
        assert payload['training_info']['init_seed']==parent and payload['training_info'].get('task') is None
        digest=sha(target);state=state_sha(model)
        for encoding in ['digits','atomic']:
            for task in TASKS:cases.append(dict(id=f'initial-{parent}-{encoding}-{task}',stage='initial',parent=parent,encoding=encoding,task=task,
                checkpoint=target.relative_to(o).as_posix(),checkpoint_sha256=digest,state_sha256=state,training_updates=0,original_language_updates=400,model_source_commit=payload['source_info'].get('source_commit')))
    write(o/'cases.json',cases)
    oldmeta=read(a.prior_packet_metadata);write(o/'original_model_source.json',oldmeta['source_sha256'])
    source_files=[*sorted((REPO/'clmm').rglob('*.py')),*sorted((REPO/'titans_pytorch').glob('*.py')),*sorted((REPO/'scripts').glob('*.py')),REPO/'pyproject.toml',REPO/'requirements-lock.txt']
    source_hashes={}
    for p in source_files:
        relative=p.relative_to(REPO);target=o/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);source_hashes[relative.as_posix()]=sha(p)
    for relative,digest in oldmeta['source_sha256'].items():assert sha(REPO/relative)==digest,relative
    hashes={p.relative_to(o).as_posix():sha(p) for p in o.rglob('*') if p.is_file() and 'sources' not in p.relative_to(o).parts}
    meta=dict(status='prepared',created_utc=utc(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_sha256=source_hashes,files_sha256=hashes,cases=24,distinct_checkpoints=len(model_files),main_packet_rows=34560,main_vectors=138240,
        prior_histories=len(prior),engineering_histories=len(engineering_ids),main_histories=640,command=[sys.executable,*sys.argv],environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__))
    write(o/'metadata.json',meta);print(json.dumps(meta|{'source_sha256':'saved','files_sha256':'saved'}),flush=True)


def verify_packet(packet):
    m=read(packet/'metadata.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==m['source_commit']
    for n,d in m['source_sha256'].items():assert sha(REPO/n)==d,n
    for n,d in m['files_sha256'].items():assert sha(packet/n)==d,n
    return m


@torch.no_grad()
def evaluate_case(a,case,dataset,out,deadline):
    started=utc();tick=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    checkpoint=a.packet/case['checkpoint'];assert sha(checkpoint)==case['checkpoint_sha256']
    model,cfg,payload=load_checkpoint(checkpoint,map_location='cpu');assert state_sha(model)==case['state_sha256']
    if cfg.segment_len!=128:
        assert case['stage']=='initial' and cfg.segment_len==32
        cfg=replace(cfg,segment_len=128)
        with torch.random.fork_rng(devices=[]):changed=build_model(cfg)
        changed.load_state_dict(model.state_dict(),strict=True);model=changed
    assert sum(p.numel() for p in model.parameters())==839248 and state_sha(model)==case['state_sha256']
    model.eval();before=state_sha(model);rng=torch.random.get_rng_state().clone()
    rows=[r for r in read(a.packet/f'{dataset}_samples.json.gz') if r['task']==case['task'] and r['encoding']==case['encoding']]
    for i,r in enumerate(rows):r.update(score_index=i,packet_row_index=r.pop('packet_row_index'),case=case['id'])
    logits=np.empty((len(rows),256),np.float32);resources=[];buckets={}
    for i,r in enumerate(rows):buckets.setdefault((r['condition'],r['cell'],r['group'],r['variant']),[]).append(i)
    for key,indices in buckets.items():
        for offset in range(0,len(indices),16):
            if time.perf_counter()>deadline:raise TimeoutError('Bounded CPU study deadline')
            ix=indices[offset:offset+16];x=torch.tensor([list(bytes.fromhex(rows[i]['prefix_hex'])) for i in ix]);t=time.perf_counter()
            y=model(x)[:,-1];elapsed=time.perf_counter()-t
            target=torch.tensor([rows[i]['target'] for i in ix]);scores={k:v.cpu().tolist() for k,v in score_logits(y,target,case['task']).items()}
            logits[ix]=y.cpu().numpy()
            for j,i in enumerate(ix):rows[i].update(**{k:v[j] for k,v in scores.items()},correct=int(scores['prediction'][j]==rows[i]['target']))
            resources.append(dict(condition=key[0],cell=key[1],group=key[2],variant=key[3],row_indices=json.dumps(ix),queries=len(ix),forward_seconds=elapsed))
    assert before==state_sha(model) and torch.equal(rng,torch.random.get_rng_state())
    erased={}
    for i,r in enumerate(rows):
        if r['condition']=='cue_erased':erased.setdefault((r['group'],r['episode']),[]).append(i)
    for ix in erased.values():assert len(ix)==2 and rows[ix[0]]['prefix_hex']==rows[ix[1]]['prefix_hex'] and np.array_equal(logits[ix[0]],logits[ix[1]])
    np.save(out/'logits.npy',logits);csvfile(out/'predictions.csv',rows);csvfile(out/'resources.csv',resources)
    meta=dict(status='succeeded',case=case,dataset=dataset,started_utc=started,ended_utc=utc(),queries=len(rows),registered_parameters=839248,
        model_state_sha256=before,weights_and_RNG_pure=True,forward_seconds=sum(r['forward_seconds'] for r in resources),
        elapsed_seconds=time.perf_counter()-tick,logits_sha256=sha(out/'logits.npy'),predictions_sha256=sha(out/'predictions.csv'),
        packet_metadata_sha256=sha(a.packet/'metadata.json'),environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device='cpu',threads=2))
    write(out/'metadata.json',meta);print(json.dumps(dict(case=case['id'],dataset=dataset,status='succeeded',queries=len(rows),seconds=meta['elapsed_seconds'])),flush=True)
    return meta


def run(a):
    packet=verify_packet(a.packet);a.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();deadline=start+2700
    audit=read(a.data_audit);assert audit['status']=='passed' and audit['packet_metadata_sha256']==sha(a.packet/'metadata.json')
    cases=read(a.packet/'cases.json');write(a.out/'run_metadata.json',dict(status='running',started_utc=utc(),source_commit=packet['source_commit'],
        packet_metadata_sha256=sha(a.packet/'metadata.json'),data_audit_sha256=sha(a.data_audit),command=[sys.executable,*sys.argv]))
    try:
        engineering=[]
        for task in ['select','value']:
            c=next(r for r in cases if r['stage']=='trained' and r['parent']==68101 and r['encoding']=='digits' and r['task']==task)
            engineering.append(evaluate_case(a,c,'engineering',a.out/'engineering'/c['id'],deadline))
        rate=max(r['forward_seconds']/r['queries'] for r in engineering);fixed=max(r['elapsed_seconds']-r['forward_seconds'] for r in engineering)
        projection=1.25*(138240*rate+24*fixed+180)
        decision=dict(created_utc=utc(),before_main_outputs=True,forward_seconds_per_query=rate,fixed_case_seconds=fixed,
            projected_seconds=projection,remaining_seconds=deadline-time.perf_counter(),proceed=projection<deadline-time.perf_counter(),padding=1.25)
        write(a.out/'runtime_decision.json',decision)
        if not decision['proceed']:raise RuntimeError('All24 fixed main cases exceed declared45minute CPU budget; no main inference started')
        for case in cases:evaluate_case(a,case,'main',a.out/'main'/case['id'],deadline)
        write(a.out/'run_metadata.json',dict(status='succeeded',started_utc=read(a.out/'run_metadata.json')['started_utc'],ended_utc=utc(),
            elapsed_seconds=time.perf_counter()-start,source_commit=packet['source_commit'],packet_metadata_sha256=sha(a.packet/'metadata.json'),
            data_audit_sha256=sha(a.data_audit),main_cases=24,main_vectors=138240,command=[sys.executable,*sys.argv]))
    except BaseException:
        meta=read(a.out/'run_metadata.json');meta.update(status='failed',ended_utc=utc(),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start)
        write(a.out/'run_metadata.json',meta);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run']);p.add_argument('--out',required=True,type=Path)
    for name in ['packet','protocol','exclusions','registry','parents','prior-packet-metadata','data-audit']:p.add_argument('--'+name,type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(91802);torch.use_deterministic_algorithms(True)
    {'prepare':prepare,'run':run}[a.mode](a)
