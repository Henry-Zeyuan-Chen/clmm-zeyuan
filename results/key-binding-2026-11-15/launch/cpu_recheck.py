"""Actual selected CPU inference, using exact saved GPU batch membership."""
import csv,hashlib,json,sys,time
from contextlib import nullcontext
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-key-binding-2026-11-15'
sys.path.insert(0,str(WORK/'source-frozen'))
import numpy as np
import torch
from clmm.pretrain import load_checkpoint,reset_memory_at_segments,zero_memory_output
from scripts.sequence_freshness_experiment import state_sha
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def prob(a):
    a=a.astype(np.float64);a=np.exp(a-a.max(-1,keepdims=True));return a/a.sum(-1,keepdims=True)

def main():
    out=ROOT/'cpu-recheck';out.mkdir(exist_ok=False)
    plan=dict(selection='First saved GPU resource batch for every group/condition in every initial/terminal case',
        maximum_seconds=240,max_logit_error=.002,max_probability_error=.0002,prediction_mismatches_allowed=0,
        threads=2,scope='Selected exact-batch inference, not full training reproduction')
    write(out/'plan.json',plan);start=time.monotonic()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(91402)
    packet=read(ROOT/'packet/metadata.json')
    assert all(sha(WORK/'source-frozen'/n)==h for n,h in packet['source_sha256'].items())
    folders=sorted((ROOT/'gpu-main/result/initial').iterdir())+sorted((ROOT/'gpu-main/result/runs').iterdir())
    assert len(folders)==8
    records=[];CPU=[];GPU=[];selected=[]
    for folder in folders:
        meta=read(folder/'metadata.json');assert meta['status']=='succeeded' and sha(folder/'checkpoint.pt')==meta['checkpoint_sha256']
        model,config,_=load_checkpoint(folder/'checkpoint.pt',map_location='cpu');model.eval()
        assert config.segment_len==128 and sum(p.numel() for p in model.parameters())==839248
        before=state_sha(model);rng=torch.random.get_rng_state().clone()
        saved=rows(folder/'evaluation/predictions.csv');reference=np.load(folder/'evaluation/answer_logits.npy')
        chosen=set();checks=[]
        for batch_id,resource in enumerate(rows(folder/'evaluation/resources.csv')):
            indices=json.loads(resource['row_indices']);keys={(saved[i]['group'],saved[i]['condition']) for i in indices}
            assert len(keys)==1
            key=next(iter(keys))
            if key in chosen:continue
            chosen.add(key)
            if time.monotonic()-start>240:raise TimeoutError('Selected CPU240s bound')
            x=torch.tensor([list(bytes.fromhex(saved[i]['prefix_hex'])) for i in indices],dtype=torch.long)
            condition=resource['condition']
            scope=reset_memory_at_segments(model) if condition=='reset_memory' else zero_memory_output(model) if condition=='retrieval_zero' else nullcontext()
            with scope:
                with torch.no_grad():actual=model(x)[:,-1].numpy()
            expected=reference[indices];error=np.max(np.abs(actual-expected),1)
            probability_error=np.max(np.abs(prob(actual)-prob(expected)),1)
            mismatches=actual.argmax(1)!=expected.argmax(1)
            checks.append(dict(group=key[0],condition=key[1],saved_batch=batch_id,indices=indices,queries=len(indices),
                max_logit_error=float(error.max()),max_probability_error=float(probability_error.max()),prediction_mismatches=int(mismatches.sum()),
                failed_rows=[dict(score_index=indices[j],logit_error=float(error[j]),probability_error=float(probability_error[j]),prediction_changed=bool(mismatches[j]))
                    for j in range(len(indices)) if error[j]>.002 or probability_error[j]>.0002 or mismatches[j]]))
            CPU.extend(actual);GPU.extend(expected);selected.extend(dict(case=folder.name,**saved[i]) for i in indices)
        assert len(chosen)==20 and before==state_sha(model) and torch.equal(rng,torch.random.get_rng_state())
        record=dict(case=folder.name,checkpoint_sha256=sha(folder/'checkpoint.pt'),queries=sum(c['queries'] for c in checks),
            max_logit_error=max(c['max_logit_error'] for c in checks),max_probability_error=max(c['max_probability_error'] for c in checks),
            prediction_mismatches=sum(c['prediction_mismatches'] for c in checks),checks=checks,checkpoint_and_RNG_pure=True)
        records.append(record);write(out/'progress.json',records);print(json.dumps({k:v for k,v in record.items() if k!='checks'}),flush=True)
    np.save(out/'cpu_logits.npy',np.asarray(CPU));np.save(out/'gpu_logits.npy',np.asarray(GPU));write(out/'selected_rows.json',selected)
    passed=all(r['max_logit_error']<=.002 and r['max_probability_error']<=.0002 and r['prediction_mismatches']==0 for r in records)
    result=dict(status='passed' if passed else 'numerical_review_required',models=8,queries=len(selected),cases=records,
        max_logit_error=max(r['max_logit_error'] for r in records),max_probability_error=max(r['max_probability_error'] for r in records),
        prediction_mismatches=sum(r['prediction_mismatches'] for r in records),script_sha256=sha(Path(__file__)),plan=plan,
        environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device='CPU'),elapsed_seconds=time.monotonic()-start)
    write(out/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    assert passed
if __name__=='__main__':main()
