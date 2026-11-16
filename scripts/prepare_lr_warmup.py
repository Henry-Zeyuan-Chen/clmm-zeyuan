"""Freeze fresh identical mixed-format data for constant versus warm-up LR."""
import argparse,shutil,subprocess,sys,json
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import numpy as np
import torch
from scripts.format_training_experiment import read,write,sha,utc,load_checkpoint,state_sha
from clmm.pretrain.format_training import fresh_training,training_arrays,evaluation_samples,templates,WARMUP_CELLS
from clmm.pretrain.format_factorial import fresh_worlds
from clmm.pretrain.record_selection import make_codec,collect_variant_signatures

P=argparse.ArgumentParser();P.add_argument('--out',type=Path,required=True);P.add_argument('--prior-packet',type=Path,required=True);P.add_argument('--exclusions',type=Path,required=True);P.add_argument('--protocol',type=Path,required=True)
A=P.parse_args();A.out.mkdir(parents=True,exist_ok=False);o=A.out;torch.set_num_threads(2)
protocol=read(A.protocol);assert protocol['arms']==['constant','warmup'] and protocol['reserved_fact_templates']==templates(True)['facts'][3] and protocol['reserved_query_templates']==templates(True)['queries'][3]
prior=set(read(A.exclusions));assert len(prior)==985652;codec=make_codec()
write(o/'protocol.json',protocol);write(o/'templates.json',templates(True));write(o/'key_codec.json',codec.to_dict());shutil.copy2(A.exclusions,o/'prior_exclusions.json.gz')
eng,seen=fresh_training(prior,steps=64,seed=92302,codec=codec)
eng_eval=fresh_worlds(seen,seed=92313,n_train=8,n_dev=8,codec=codec);seen.update(collect_variant_signatures(eng_eval))
main,seen=fresh_training(seen,steps=6000,seed=92202,codec=codec)
confirmation=fresh_worlds(seen,seed=92223,n_train=256,n_dev=256,codec=codec)
for name,plan in [('engineering',eng),('main',main)]:
    write(o/f'{name}_training_plan.json.gz',plan);arrays=training_arrays(plan,codec)
    np.savez_compressed(o/f'{name}_training.npz',**{k:v for k,v in arrays.items() if k in ('mixed_digits','mixed_atomic','targets')})
for name,worlds in [('engineering',eng_eval),('confirmation',confirmation)]:
    write(o/f'{name}_worlds.json',worlds)
    write(o/f'{name}_samples.json.gz',evaluation_samples(worlds,codec,confirmation=name=='confirmation',cells=WARMUP_CELLS if name=='confirmation' else None))
seen.update(collect_variant_signatures(confirmation));assert len(seen)==1181812
write(o/'all_consulted_histories.json.gz',sorted(seen));write(o/'history_counts.json',dict(prior=985652,engineering_training=2048,engineering_evaluation=64,main_training=192000,confirmation=2048,total=len(seen)))
inits=read(A.prior_packet/'initializers.json');assert len(inits)==4;(o/'initializers').mkdir()
for init in inits:
    dest=o/init['checkpoint'];shutil.copy2(A.prior_packet/init['checkpoint'],dest);assert sha(dest)==init['checkpoint_sha256']
    model,cfg,payload=load_checkpoint(dest);assert state_sha(model)==init['state_sha256'] and cfg.segment_len==128 and sum(p.numel() for p in model.parameters())==839248
    assert payload['training_info']['task']=='select' and payload['training_info']['steps']==20000
write(o/'initializers.json',inits)
sources={}
for p in [*sorted((REPO/'clmm').rglob('*.py')),*sorted((REPO/'titans_pytorch').glob('*.py')),*sorted((REPO/'scripts').glob('*.py')),REPO/'pyproject.toml',REPO/'requirements-lock.txt']:
    relative=p.relative_to(REPO);dest=o/'sources'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);sources[relative.as_posix()]=sha(p)
old=read(A.prior_packet/'metadata.json')['original_model_source']
for n,d in old.items():assert sha(REPO/n)==d,n
hashes={p.relative_to(o).as_posix():sha(p) for p in o.rglob('*') if p.is_file() and 'sources' not in p.relative_to(o).parts}
meta=dict(status='prepared',created_utc=utc(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),source_sha256=sources,original_model_source=old,files_sha256=hashes,
    command=[sys.executable,*sys.argv],confirmation_packet_rows=77824,confirmation_vectors=466944,environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__))
write(o/'metadata.json',meta);print(json.dumps(dict(status='prepared',source=meta['source_commit'],histories=len(seen))),flush=True)
