"""Selected actual CPU re-inference of frozen GPU outputs, with fixed selection."""
import csv,hashlib,json,os,sys,time
from contextlib import nullcontext
from pathlib import Path
WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]/'outputs/clmm-sequence-freshness-2026-11-15'
SOURCE=WORK/'source-checkout'
sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
from clmm.pretrain import load_checkpoint,reset_memory_at_segments
from scripts.sequence_freshness_experiment import state_sha

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')

def main():
    out=ROOT/'cpu-recheck';out.mkdir(exist_ok=False)
    # Set before reading model scores; episodes0/1 in every group/arm/condition.
    plan=dict(selection='episode in [0,1], every initial/final model, group, arm and condition',
              absolute_logit_tolerance=.002,maximum_probability_error=.0002,
              maximum_seconds=300,scope='selected inference, not full training reproduction',
              threads=2,batch_size=16)
    write(out/'plan.json',plan);started=time.monotonic()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.manual_seed(91302)
    packet=read(ROOT/'packet-final/metadata.json')
    assert all(sha(SOURCE/n)==h for n,h in packet['source_sha256'].items())
    folders=sorted((ROOT/'gpu-output/result/initial').iterdir())+sorted((ROOT/'gpu-output/result/runs').iterdir())
    assert len(folders)==12
    report=[];saved=[];cpu_scores=[];expected_scores=[];global_peak=0.
    for folder in folders:
        if time.monotonic()-started>plan['maximum_seconds']:raise TimeoutError('CPU proof300s limit')
        meta=read(folder/'metadata.json');assert meta['status']=='succeeded'
        with (folder/'evaluation/predictions.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
        reference=np.load(folder/'evaluation/answer_logits.npy')
        selected=[r for r in rows if int(r['episode']) in (0,1)]
        model,config,payload=load_checkpoint(folder/'checkpoint.pt',map_location='cpu');model.eval()
        assert sum(p.numel() for p in model.parameters())==839248
        assert sha(folder/'checkpoint.pt')==meta['checkpoint_sha256']
        before=state_sha(model);rng=torch.random.get_rng_state().clone();buckets={}
        for row in selected:buckets.setdefault((row['condition'],int(row['input_bytes'])),[]).append(row)
        case_peak=0.;prob_peak=0.;differences=0;near_ties=[];count=0
        for (condition,length),group in buckets.items():
            with (reset_memory_at_segments(model) if condition=='reset_memory' else nullcontext()):
                for offset in range(0,len(group),16):
                    if time.monotonic()-started>plan['maximum_seconds']:raise TimeoutError('CPU proof300s limit')
                    batch=group[offset:offset+16]
                    x=torch.tensor([list(bytes.fromhex(r['prefix_hex'])) for r in batch],dtype=torch.long)
                    with torch.no_grad():actual=model(x)[:,-1].numpy()
                    expected=reference[[int(r['score_index']) for r in batch]]
                    error=np.max(np.abs(actual-expected),axis=1);case_peak=max(case_peak,float(error.max()))
                    probabilities=[]
                    for values in (actual,expected):
                        values=values.astype(np.float64);values-=values.max(1,keepdims=True)
                        values=np.exp(values);probabilities.append(values/values.sum(1,keepdims=True))
                    prob_peak=max(prob_peak,float(np.abs(probabilities[0]-probabilities[1]).max()))
                    mismatch=actual.argmax(1)!=expected.argmax(1);differences+=int(mismatch.sum())
                    for j,r in enumerate(batch):
                        if mismatch[j]:
                            ordered=np.sort(expected[j]);margin=float(ordered[-1]-ordered[-2])
                            near_ties.append(dict(score_index=int(r['score_index']),margin=margin,error=float(error[j]),covered_by_perturbation=margin<=2*float(error[j])))
                        saved.append(dict(case=folder.name,**r));cpu_scores.append(actual[j]);expected_scores.append(expected[j])
                    count+=len(batch)
        assert before==state_sha(model) and torch.equal(rng,torch.random.get_rng_state())
        entry=dict(case=folder.name,checkpoint_sha256=sha(folder/'checkpoint.pt'),queries=count,
                   max_absolute_logit_error=case_peak,max_probability_error=prob_peak,
                   argmax_differences=differences,near_ties=near_ties,checkpoint_pure=True,RNG_unchanged=True)
        report.append(entry);global_peak=max(global_peak,case_peak)
        write(out/'progress.json',report);print(json.dumps(entry),flush=True)
        del model
    np.save(out/'cpu_logits.npy',np.asarray(cpu_scores));np.save(out/'gpu_logits.npy',np.asarray(expected_scores))
    write(out/'selected_rows.json',saved)
    result=dict(status='passed' if all(r['max_absolute_logit_error']<=.002 and r['max_probability_error']<=.0002 and r['argmax_differences']==0 for r in report) else 'numerical_review_required',
                models=len(report),queries=sum(r['queries'] for r in report),max_absolute_error=global_peak,
                argmax_differences=sum(r['argmax_differences'] for r in report),cases=report,
                source_commit=packet['source_commit'],script_sha256=sha(Path(__file__)),
                environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device='CPU'),
                elapsed_seconds=time.monotonic()-started,plan=plan)
    write(out/'verification.json',result);print(json.dumps(result),flush=True)
    assert result['status']=='passed'
if __name__=='__main__':main()
