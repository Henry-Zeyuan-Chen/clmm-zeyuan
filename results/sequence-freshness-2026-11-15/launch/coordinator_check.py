"""Coordinator's raw-CSV primary recomputation, separate from delegated analysis."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]/'outputs/clmm-sequence-freshness-2026-11-15'

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_rows(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def main():
    out=ROOT/'validation';out.mkdir(exist_ok=True)
    pipe=read(ROOT/'gpu-output/result/pipeline.json');assert pipe['status']=='succeeded'
    folders=sorted((ROOT/'gpu-output/result/runs').iterdir());assert len(folders)==8
    cases={};totals=dict(updates=0,answers=0,input_bytes=0,score_vectors=0,memory_nonzero_gradient_steps=0)
    expected={(parent,segment,arm) for parent in (68101,68105) for segment in (32,128) for arm in ('periodic','periodic_bounded')}
    for folder in folders:
        meta=read(folder/'metadata.json');key=(meta['parent_seed'],meta['segment'],meta['training_arm'])
        assert key not in cases and key in expected and meta['status']=='succeeded'
        trace=csv_rows(folder/'training.csv');assert len(trace)==2000
        assert [int(r['step']) for r in trace]==list(range(1,2001))
        assert sum(int(r['answers']) for r in trace)==32000
        assert sum(int(r['input_bytes']) for r in trace)==3646208
        assert sha(folder/'checkpoint.pt')==meta['checkpoint_sha256']
        rows=csv_rows(folder/'evaluation/predictions.csv');assert len(rows)==4352
        pair={}
        for r in rows:
            # Use complete256-head prediction, never forced legal-value decoding.
            correct=int(r['prediction'])==int(r['target'])
            assert correct==bool(int(r['correct']))
            pair.setdefault((r['group'],r['arm'],r['condition'],int(r['episode'])),[]).append(correct)
        assert all(len(v)==2 for v in pair.values())
        cases[key]={k:int(all(v)) for k,v in pair.items()}
        totals['updates']+=len(trace);totals['answers']+=sum(int(r['answers']) for r in trace)
        totals['input_bytes']+=sum(int(r['input_bytes']) for r in trace);totals['score_vectors']+=len(rows)
        totals['memory_nonzero_gradient_steps']+=sum(float(r['memory_gradient_norm'])>0 for r in trace)
    assert set(cases)==expected
    for folder in (ROOT/'gpu-output/result/initial').iterdir():
        meta=read(folder/'metadata.json');assert meta['completed_steps']==0
        totals['score_vectors']+=len(csv_rows(folder/'evaluation/predictions.csv'))
    assert totals==dict(updates=16000,answers=256000,input_bytes=29169664,score_vectors=52224,
                       memory_nonzero_gradient_steps=totals['memory_nonzero_gradient_steps'])
    primary=[];independent=csv_rows(ROOT/'analysis/audited/primary_estimates.csv')
    for group in ('acquire_train_seen','acquire_dev_heldout'):
        values=np.array([np.mean([cases[(p,s,'periodic_bounded')][(group,'online','normal',i)]-
              cases[(p,s,'periodic')][(group,'online','normal',i)]
              for p in (68101,68105) for s in (32,128)]) for i in range(128)])
        rng=np.random.default_rng(91304);indices=rng.integers(0,128,size=(5000,128))
        boot=values[indices].mean(1);lower,upper=np.quantile(boot,[.0125,.9875])
        entry=dict(endpoint=group,estimate=float(values.mean()),lower=float(lower),upper=float(upper))
        other=next(r for r in independent if r['endpoint']==group)
        assert all(abs(entry[k]-float(other[k]))<1e-12 for k in ('estimate','lower','upper'))
        primary.append(entry)
    gates=[]
    for key,pair in sorted(cases.items()):
        for group in ('acquire_train_seen','acquire_dev_seen','acquire_dev_heldout'):
            count=sum(pair[(group,'online','normal',i)] for i in range(128))
            gates.append(dict(parent=key[0],segment=key[1],training_arm=key[2],group=group,
                              correct_worlds=count,worlds=128,passed=count/128>=.8))
    result=dict(status='passed',source_commit=pipe['source_commit'],totals=totals,primary=primary,
                gates=gates,passed_gate_cells=sum(r['passed'] for r in gates),
                primary_agrees_with_independent_review=True,script_sha256=sha(Path(__file__)))
    (out/'coordinator-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
