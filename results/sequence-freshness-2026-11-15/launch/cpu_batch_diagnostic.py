"""Preserved post-failure diagnosis, never a relaxed replacement acceptance test."""
import csv,hashlib,json,sys,time
from contextlib import nullcontext
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-sequence-freshness-2026-11-15'
sys.path.insert(0,str(WORK/'source-checkout'))
import numpy as np
import torch
from clmm.pretrain import load_checkpoint,reset_memory_at_segments
from scripts.sequence_freshness_experiment import state_sha
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def probabilities(a):
    a=a.astype(np.float64);a=np.exp(a-a.max(-1,keepdims=True));return a/a.sum(-1,keepdims=True)

start=time.monotonic();out=ROOT/'cpu-recheck/batch-diagnostic';out.mkdir(exist_ok=False)
write(out/'plan.json',dict(stage='post-failure diagnostic',selection='All original GPU resource batches containing selected rows over original logit/probability thresholds',
      maximum_seconds=120,CPUs=2,original_acceptance_thresholds_unchanged=True,no_GPU_or_training=True))
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
selected=read(ROOT/'cpu-recheck/selected_rows.json');a=np.load(ROOT/'cpu-recheck/cpu_logits.npy');b=np.load(ROOT/'cpu-recheck/gpu_logits.npy')
error=np.max(np.abs(a-b),1);p_error=np.max(np.abs(probabilities(a)-probabilities(b)),1)
failures=[i for i in range(len(selected)) if error[i]>.002 or p_error[i]>.0002]
case_names=sorted({selected[i]['case'] for i in failures});report=[]
for name in case_names:
    folder=ROOT/'gpu-output/result/runs'/name;model,_,_=load_checkpoint(folder/'checkpoint.pt',map_location='cpu');model.eval()
    before=state_sha(model);rng=torch.random.get_rng_state().clone();saved=rows(folder/'evaluation/predictions.csv');gpu=np.load(folder/'evaluation/answer_logits.npy')
    bad={int(selected[i]['score_index']):i for i in failures if selected[i]['case']==name}
    for index,resource in enumerate(rows(folder/'evaluation/resources.csv')):
        indices=json.loads(resource['row_indices']);wanted=sorted(set(indices)&set(bad))
        if not wanted:continue
        assert time.monotonic()-start<120
        x=torch.tensor([list(bytes.fromhex(saved[i]['prefix_hex'])) for i in indices],dtype=torch.long)
        with (reset_memory_at_segments(model) if resource['condition']=='reset_memory' else nullcontext()):
            with torch.no_grad():actual=model(x)[:,-1].numpy()
        np.save(out/f'{name}-batch{index}.npy',actual)
        for i in wanted:
            position=indices.index(i);original=a[bad[i]]
            with (reset_memory_at_segments(model) if resource['condition']=='reset_memory' else nullcontext()):
                with torch.no_grad():single=model(x[position:position+1])[:,-1].numpy()[0]
            entry=dict(case=name,score_index=i,group=saved[i]['group'],arm=saved[i]['arm'],
                exact_GPU_batch_indices=indices,original_subset_GPU_max_error=float(np.max(np.abs(original-gpu[i]))),
                exact_batch_GPU_max_error=float(np.max(np.abs(actual[position]-gpu[i]))),
                exact_batch_GPU_max_probability_error=float(np.max(np.abs(probabilities(actual[position:position+1])-probabilities(gpu[i:i+1])))),
                exact_batch_original_subset_CPU_error=float(np.max(np.abs(actual[position]-original))),
                exact_batch_single_CPU_error=float(np.max(np.abs(actual[position]-single))),
                single_GPU_error=float(np.max(np.abs(single-gpu[i]))),
                predicted_bytes=dict(GPU=int(gpu[i].argmax()),CPU_exact_batch=int(actual[position].argmax()),CPU_single=int(single.argmax()),CPU_original_subset=int(original.argmax())))
            report.append(entry)
    assert before==state_sha(model) and torch.equal(rng,torch.random.get_rng_state())
result=dict(status='completed',scope='post-failure diagnosis, original numerical acceptance remains unmet',
    failures=len(failures),affected_models=len(case_names),rows=report,checkpoint_and_RNG_pure=True,
    exact_batch_all_within_original_tolerance=all(r['exact_batch_GPU_max_error']<=.002 and r['exact_batch_GPU_max_probability_error']<=.0002 for r in report),
    all_predicted_bytes_equal=all(len(set(r['predicted_bytes'].values()))==1 for r in report),
    elapsed_seconds=time.monotonic()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
write(out/'review.json',result);print(json.dumps(result,indent=2))
