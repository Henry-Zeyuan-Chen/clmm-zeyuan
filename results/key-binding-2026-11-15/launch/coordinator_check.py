"""Coordinator raw prediction/trace recomputation independent of delegated code."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-key-binding-2026-11-15'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def main():
    out=ROOT/'validation';out.mkdir(exist_ok=True)
    pipe=read(ROOT/'gpu-main/result/pipeline.json');assert pipe['status']=='succeeded'
    cases={};totals=dict(updates=0,forks=0,input_bytes=0,score_vectors=0,memory_nonzero_gradient_steps=0)
    expected={(p,e) for p in (68101,68105) for e in ('digits','atomic')}
    for folder in sorted((ROOT/'gpu-main/result/runs').iterdir()):
        meta=read(folder/'metadata.json');key=(meta['parent_seed'],meta['encoding']);assert key in expected and key not in cases
        assert sha(folder/'checkpoint.pt')==meta['checkpoint_sha256']
        trace=rows(folder/'training.csv');assert [int(r['step']) for r in trace]==list(range(1,2001))
        assert sum(int(r['answers']) for r in trace)==32000 and sum(int(r['input_bytes']) for r in trace)==2908928
        totals['updates']+=len(trace);totals['forks']+=32000;totals['input_bytes']+=2908928
        totals['memory_nonzero_gradient_steps']+=sum(float(r['memory_gradient_norm'])>0 for r in trace)
        pairs={};saved=rows(folder/'evaluation/predictions.csv');assert len(saved)==10240
        for r in saved:
            actual=int(r['prediction'])==int(r['target']);assert actual==bool(int(r['correct']))
            pairs.setdefault((r['group'],r['condition'],int(r['episode'])),[]).append(actual)
        assert all(len(p)==2 for p in pairs.values())
        cases[key]={k:int(all(v)) for k,v in pairs.items()};totals['score_vectors']+=len(saved)
    assert set(cases)==expected
    for folder in (ROOT/'gpu-main/result/initial').iterdir():
        assert read(folder/'metadata.json')['completed_steps']==0
        totals['score_vectors']+=len(rows(folder/'evaluation/predictions.csv'))
    assert totals['score_vectors']==81920 and totals['updates']==8000
    primaries=[]
    for group,swap in (('acquire_train_seen','acquire_train_swapped'),('acquire_dev_heldout','acquire_dev_heldout_swapped')):
        values=[]
        for i in range(256):
            contrasts=[]
            for p in (68101,68105):
                joint={e:cases[(p,e)][(group,'normal',i)]*cases[(p,e)][(swap,'normal',i)] for e in ('digits','atomic')}
                contrasts.append(joint['atomic']-joint['digits'])
            values.append(np.mean(contrasts))
        a=np.asarray(values);rng=np.random.default_rng(91404);boot=a[rng.integers(0,256,size=(5000,256))].mean(1)
        lower,upper=np.quantile(boot,[.0125,.9875]);primaries.append(dict(group=group,estimate=float(a.mean()),lower=float(lower),upper=float(upper)))
    gates=[]
    for (parent,encoding),case in sorted(cases.items()):
        for group in ('acquire_train_seen','acquire_train_swapped','acquire_dev_seen','acquire_dev_swapped','acquire_dev_heldout','acquire_dev_heldout_swapped'):
            count=sum(case[(group,'normal',i)] for i in range(256))
            gates.append(dict(parent=parent,encoding=encoding,group=group,both_correct=count,episodes=256,passed=count>=205))
    result=dict(status='passed',totals=totals,primary=primaries,gates=gates,passed_gate_cells=sum(g['passed'] for g in gates),script_sha256=sha(Path(__file__)))
    (out/'coordinator-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
