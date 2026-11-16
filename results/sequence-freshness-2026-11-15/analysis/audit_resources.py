"""Check pipeline resources, logged budgets and recovery-only midpoint artifacts."""
import collections,csv,json
from pathlib import Path
import numpy as np
import torch
from audit_packet import read_json,sha
from audit_provenance import check_finite,tensor_state_sha
from audit_results import load_csv
ROOT=Path(__file__).resolve().parents[1];result_root=ROOT/'gpu-output/result';pipeline=read_json(result_root/'pipeline.json')
assert pipeline['status']=='succeeded' and pipeline['source_commit']=='95da59531ccf5d5f3fe2729553e6d980da598352'
assert pipeline['workers']==2 and pipeline['steps']==2000 and pipeline['slurm_job']=='13922356'
expected_initial={f'{parent}-w{segment}' for parent in [68101,68105] for segment in [32,128]}
expected_runs={f'{parent}-w{segment}-{arm}' for parent in [68101,68105] for segment in [32,128] for arm in ['periodic','periodic_bounded']}
assert {r['name'] for r in pipeline['initial']}==expected_initial
assert {r['name'] for r in pipeline['cases']}==expected_runs
assert {r['name'] for r in pipeline['preflight']}=={'w32','w128'}
train=max(r['training_seconds'] for r in pipeline['preflight']);evaluation=max(r['evaluation_seconds'] for r in pipeline['preflight'])
projection=1.25*(8*100*train+12*8*evaluation)/2
assert np.isclose(projection,pipeline['projected_main_seconds'],rtol=0,atol=1e-9)
assert pipeline['runtime_gate']==(projection<=4200) and pipeline['main_elapsed_seconds']<=4200 and pipeline['elapsed_seconds']<=5000
assert min(r['elapsed_seconds'] for r in pipeline['preflight'])>=max(r['training_seconds'] for r in pipeline['preflight'])
main_steps=main_answers=main_bytes=main_gradient_steps=0;eval_rows=eval_bytes=eval_batches=0;stage_rows=[];midpoints=[]
for section,directory in [('initial','initial'),('cases','runs'),('preflight','engineering')]:
    for record in pipeline[section]:
        folder=result_root/directory/record['name'];meta=read_json(folder/'metadata.json')
        assert meta['status']=='succeeded' and record['checkpoint_sha256']==sha(folder/'checkpoint.pt')==meta['checkpoint_sha256']
        assert meta['parent_seed']==record['parent'] and meta['segment']==record['segment'] and meta['training_arm']==record['arm']
        assert np.isclose(record['training_seconds'],meta['training_elapsed_seconds'],rtol=0,atol=1e-9)
        assert np.isclose(record['evaluation_seconds'],meta['evaluation']['elapsed_seconds'],rtol=0,atol=1e-9)
        if section in ['initial','cases']:
            resources=load_csv(folder/'evaluation/resources.csv');eval_batches+=len(resources)
            for row in resources:
                indices=json.loads(row['row_indices']);eval_rows+=len(indices);eval_bytes+=int(row['input_length'])*len(indices)
        if section=='cases':
            trace=load_csv(folder/'training.csv');main_steps+=len(trace);main_answers+=sum(int(r['answers']) for r in trace);main_bytes+=sum(int(r['input_bytes']) for r in trace)
            main_gradient_steps+=sum(float(r['memory_gradient_norm'])>0 for r in trace)
            for stage,indices in [('k1',range(256)),('k2',list(range(256,1280))+[s for s in range(1280,2000) if (s-1280)%4!=3]),('revision',[s for s in range(1280,2000) if (s-1280)%4==3])]:
                selected=[trace[i] for i in indices]
                stage_rows.append(dict(case=folder.name,stage=stage,updates=len(selected),mean_accuracy=float(np.mean([float(r['accuracy']) for r in selected])),mean_loss=float(np.mean([float(r['loss']) for r in selected]))))
            mid=torch.load(folder/'midpoint.pt',map_location='cpu',weights_only=True);check_finite(mid['model_state'])
            assert mid['training_info']['steps']==1000 and mid['training_info']['selection']=='Recovery only; terminal2000 primary'
            assert sum(t.numel() for t in mid['model_state'].values())==839248
            optimizer=torch.load(folder/'midpoint-optimizer.pt',map_location='cpu',weights_only=True);check_finite(optimizer)
            assert {int(state['step'].item()) for state in optimizer['state'].values()}=={1000}
            midpoints.append(dict(case=folder.name,steps=1000,selection=mid['training_info']['selection'],checkpoint_sha256=sha(folder/'midpoint.pt'),optimizer_sha256=sha(folder/'midpoint-optimizer.pt'),all_finite=True))
assert (main_steps,main_answers,main_bytes,main_gradient_steps)==(16000,256000,29169664,16000)
assert (eval_rows,eval_bytes,eval_batches)==(52224,10269696,3264)
with (ROOT/'analysis/audited/training_stage_summary.csv').open('w',newline='',encoding='utf8') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(stage_rows[0]));writer.writeheader();writer.writerows(stage_rows)
result=dict(status='passed',scope='saved pipeline/CSV/tensor inspection only; no model execution',
    training_steps=main_steps,training_answer_forks=main_answers,training_input_bytes=main_bytes,memory_nonzero_gradient_steps=main_gradient_steps,
    main_evaluation_vectors=eval_rows,main_evaluation_input_bytes=eval_bytes,main_evaluation_resource_batches=eval_batches,
    initial_checkpoints=4,terminal_checkpoints=8,recovery_midpoints=len(midpoints),midpoints=midpoints,
    runtime_projection_seconds=projection,main_elapsed_seconds=pipeline['main_elapsed_seconds'],pipeline_elapsed_seconds=pipeline['elapsed_seconds'],pipeline_hard_seconds=5000,
    allocation_release_verified_by_this_script=False,allocation_note='Coordinator owns scheduler accounting/release; pipeline completion is separately audited.',
    input_sha256={'pipeline.json':sha(result_root/'pipeline.json')})
(ROOT/'analysis/resources_and_midpoints_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='midpoints'}))
