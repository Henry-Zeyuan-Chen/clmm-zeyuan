"""Saved-vector audit and prespecified statistics; no model imports or execution."""
from __future__ import annotations
import argparse, csv, hashlib, json, pathlib
import numpy as np

NORMAL_GROUPS=('acquire_train_seen','acquire_train_swapped','acquire_dev_seen','acquire_dev_swapped','acquire_dev_heldout','acquire_dev_heldout_swapped')
PRIMARY=(('train_seen','acquire_train_seen','acquire_train_swapped'),('dev_heldout','acquire_dev_heldout','acquire_dev_heldout_swapped'))

def read(path): return json.loads(path.read_text(encoding='utf8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def csvrows(path):
    with path.open(newline='',encoding='utf8') as f: return list(csv.DictReader(f))
def logsumexp(x):
    m=x.max(axis=-1);return m+np.log(np.exp(x-m[:,None]).sum(axis=-1))
def close(a,b,label):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    assert np.allclose(a,b,rtol=2e-6,atol=2e-6), (label,float(np.max(np.abs(a-b))))
    return float(np.max(np.abs(a-b))) if a.size else 0.

def audit_case(path, packet, pmeta, sources, n):
    meta=read(path/'metadata.json');assert meta['status']=='succeeded'
    assert meta['encoding'] in ('digits','atomic') and meta['parent_seed'] in (68101,68105)
    assert meta['segment']==128 and meta['parameter_count']==839248
    assert meta['data_sha256']==pmeta['files_sha256'] and meta['source_commit']==pmeta['source_commit']
    assert sha(path/'checkpoint.pt')==meta['checkpoint_sha256']
    assert meta['initial_state_sha256']==meta['parent']['state_sha256']
    is_initial=meta['mode']=='initial'
    assert meta['completed_steps']==(0 if is_initial else pmeta['steps'])
    if is_initial: assert meta['final_state_sha256']==meta['initial_state_sha256']
    else:
        assert meta['supervised_answers']==pmeta['training_answers'] and meta['training_input_bytes']==pmeta['training_input_bytes']
        logs=csvrows(path/'training.csv');assert len(logs)==pmeta['steps']
        assert [int(r['step']) for r in logs]==list(range(1,pmeta['steps']+1))
        assert all(np.isfinite(float(r[k])) for r in logs for k in ('loss','accuracy','first_accuracy','second_accuracy','prediction_changes','gradient_norm','memory_gradient_norm'))
        assert all(int(r['answers'])==pmeta['contexts']*2 for r in logs)
        assert sum(int(r['input_bytes']) for r in logs)==pmeta['training_input_bytes']
        assert sum(float(r['memory_gradient_norm'])>0 for r in logs)==meta['memory_nonzero_gradient_steps']
        assert all(r['stage']==('k1' if i<int(np.ceil(.128*pmeta['steps'])) else 'k2') for i,r in enumerate(logs))
        assert (path/'optimizer.pt').is_file()
        if pmeta['steps']>=1000: assert (path/'midpoint.pt').is_file() and (path/'midpoint-optimizer.pt').is_file()
    ep=path/'evaluation';raw=np.load(ep/'answer_logits.npy');rows=csvrows(ep/'predictions.csv')
    chosen=[r for r in sources if r['encoding']==meta['encoding']]
    assert raw.dtype==np.float32 and raw.shape==(n*40,256) and np.isfinite(raw).all()
    assert len(rows)==len(chosen)==len(raw)==meta['evaluation']['queries']
    assert sha(ep/'answer_logits.npy')==meta['evaluation']['logits_sha256']
    x=raw.astype(np.float64);norm=logsumexp(x);pred=raw.argmax(-1);forced=raw[:,97:113].argmax(-1)+97
    y=np.array([r['target'] for r in chosen]);nll=norm-x[np.arange(len(y)),y]
    legal=np.exp(x[:,97:113]-norm[:,None]).sum(-1);fnll=logsumexp(x[:,97:113])-x[np.arange(len(y)),y]
    errors={}
    for i,(r,w) in enumerate(zip(rows,chosen)):
        assert int(r['score_index'])==i
        assert all(r[k]==str(v) for k,v in w.items()),(str(path),i,'sample mismatch')
        assert int(r['prediction'])==pred[i] and int(r['forced_prediction'])==forced[i]
        assert int(r['correct'])==int(pred[i]==y[i])
    for key,expected in (('nll',nll),('forced_nll',fnll),('legal_probability_mass',legal)):
        errors[key]=close([float(r[key]) for r in rows],expected,key)
    buckets={}
    for i,r in enumerate(chosen): buckets.setdefault((r['group'],r['condition']),{}).setdefault(int(r['episode']),{})[int(r['rank'])]=i
    metrics=[];correct={};predchange={}
    for (group,condition),episodes in buckets.items():
        assert sorted(episodes)==list(range(n)) and all(set(ranks)=={0,1} for ranks in episodes.values())
        ix=np.array([[episodes[i][rank] for rank in (0,1)] for i in range(n)])
        good=pred[ix]==y[ix];both=good.all(axis=1);changes=pred[ix[:,0]]!=pred[ix[:,1]]
        correct[group,condition]=both;predchange[group,condition]=changes
        metrics.append(dict(group=group,condition=condition,episodes=n,first_correct=int(good[:,0].sum()),second_correct=int(good[:,1].sum()),both_correct_count=int(both.sum()),both_correct=float(both.mean()),accuracy=float(good.mean()),prediction_changes=float(changes.mean()),same_prediction_count=int((~changes).sum()),nll=float(nll[ix].mean()),legal_probability_mass=float(legal[ix].mean()),present_value_answers=int(np.sum((pred[ix]==y[ix][:,:1])|(pred[ix]==y[ix][:,1:])))))
    summaries=csvrows(ep/'summary.csv');assert len(summaries)==len(metrics)
    for computed,recorded in zip(metrics,summaries):
        assert computed['group']==recorded['group'] and computed['condition']==recorded['condition'] and int(recorded['episodes'])==n
        for key in ('accuracy','both_correct','prediction_changes','nll','legal_probability_mass'): close([recorded[key]],[computed[key]],key)
    for group in NORMAL_GROUPS:
        a=buckets[group,'normal'];b=buckets[group,'reset_memory']
        for i in range(n):
            for rank in (0,1): assert np.array_equal(raw[a[i][rank]],raw[b[i][rank]])
    resources=csvrows(ep/'resources.csv');covered=[]
    for r in resources:
        ix=json.loads(r['row_indices']);assert ix and len(ix)<=16
        assert all(chosen[i]['condition']==r['condition'] and chosen[i]['input_bytes']==int(r['input_length']) for i in ix)
        covered.extend(ix)
    assert sorted(covered)==list(range(len(chosen))) and len(covered)==len(set(covered))
    assert meta['evaluation']['pure_checkpoint'] and meta['evaluation']['RNG_unchanged'] and meta['evaluation']['normal_reset_parity'] and meta['reload_max_error']==0.
    joint={label:correct[g,'normal']&correct[swap,'normal'] for label,g,swap in PRIMARY}
    result=dict(run=str(path),mode=meta['mode'],parent=meta['parent_seed'],encoding=meta['encoding'],vectors=len(raw),raw_score_errors=errors,raw_hash_verified=True,predictions_exact=True,normal_reset_exact=True,resources_partition_exact=True,steps=meta['completed_steps'],metrics=metrics,joint_four={k:int(v.sum()) for k,v in joint.items()},admission_pass=all(correct[g,'normal'].sum()>=math_threshold(n) for g in NORMAL_GROUPS),admission_threshold=math_threshold(n))
    return result,joint

def math_threshold(n): return int(np.ceil(.8*n))

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=pathlib.Path,required=True);p.add_argument('--packet',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
    pm=read(a.packet/'metadata.json');sources=read(a.packet/'samples.json');n=pm['episodes'];cases=[];joint={};paths=[]
    for path in sorted(a.runs.rglob('metadata.json')):
        m=read(path)
        if m.get('mode') in ('initial','train') and 'encoding' in m and m.get('data_sha256')==pm['files_sha256']: paths.append(path.parent)
    assert len(paths)==8,('expected4initial+4terminal',len(paths))
    for path in paths:
        result,values=audit_case(path,a.packet,pm,sources,n);cases.append(result)
        key=(result['mode'],result['parent'],result['encoding']);assert key not in joint;joint[key]=values
    assert set(joint)=={(mode,parent,enc) for mode in ('initial','train') for parent in (68101,68105) for enc in ('digits','atomic')}
    draws=np.random.Generator(np.random.PCG64(91404)).integers(0,n,size=(5000,n));effects=[]
    for label,_,_ in PRIMARY:
        parent_effect=np.stack([joint['train',parent,'atomic'][label].astype(float)-joint['train',parent,'digits'][label].astype(float) for parent in (68101,68105)])
        perworld=parent_effect.mean(axis=0);boots=perworld[draws].mean(axis=1);ci=np.quantile(boots,[.0125,.9875])
        effects.append(dict(endpoint=label,atomic_minus_digits=float(perworld.mean()),ci97_5=list(map(float,ci)),worlds=n,fixed_parents=2,bootstrap_draws=5000,seed=91404,parent_effects=list(map(float,parent_effect.mean(axis=1)))))
    result=dict(status='passed',kind='independent_saved_output_audit_no_model_execution',packet=str(a.packet),run_root=str(a.runs),source_commit=pm['source_commit'],cases=cases,co_primary=effects,vectors=sum(c['vectors'] for c in cases),admission_cells_pass=sum(m['both_correct_count']>=math_threshold(n) for c in cases if c['mode']=='train' for m in c['metrics'] if m['condition']=='normal' and m['group'] in NORMAL_GROUPS),admission_cells_total=24)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))

if __name__=='__main__': main()

