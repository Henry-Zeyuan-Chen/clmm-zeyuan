"""Persistent tensor storage and observed budgets for both revision phases."""
import argparse,csv,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo))
import torch
from scripts.revision_local_experiment import load_checkpoint
torch.set_num_threads(1);rows=[];trained=[]
for phase in ('frozen','trained'):
 meta=json.loads((a.root/phase/'metadata.json').read_text());assert meta['status']=='succeeded'
 for case in meta['cases']:
  folder=a.root/phase/'runs'/case['name'];model,_=load_checkpoint(folder/'checkpoint.pt',torch.device('cpu'))
  slow=sum(t.numel()*t.element_size() for t in model.parameters());fixed=sum(t.numel()*t.element_size() for t in model.buffers());fast=model.initial_state(1).weights['trace'].untyped_storage().nbytes()
  assert slow==327684 and fixed==22016 and fast==65536
  for stage in (('evaluation',) if phase=='frozen' else ('initial-evaluation','evaluation')):
   with (folder/stage/'resources.csv').open() as h:r=list(csv.DictReader(h))
   writes=sum(int(x['actual_writes'])*int(x['batch']) for x in r);updates=sum(int(x['active_module_updates']) for x in r);decay=sum(int(x['decay_applications']) for x in r);projection=sum(int(x['projection_applications']) for x in r)
   assert writes==updates==34688 and decay==writes*int(model.config.active_bulk_decay) and projection==writes*int(model.config.key_local_erasure)
   rows.append(dict(phase=phase,stage=stage,case=case['name'],rule=case['revision_rule'],parameters=81921,slow_parameter_bytes=slow,fixed_buffer_bytes=fixed,table_bytes=512,persistent_fast_bytes_per_stream=fast,observed_writes=writes,retained_module_updates=updates,bulk_decay_applications=decay,projection_applications=projection,instrumented_eval_seconds=sum(float(x['elapsed_seconds']) for x in r),eval_device=meta['environment']['eval_device']))
  if phase=='trained':
   optimizer=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
   storage={}
   for state in optimizer['state'].values():
    for t in state.values():
     if isinstance(t,torch.Tensor):storage[(str(t.device),t.untyped_storage().data_ptr())]=t.untyped_storage().nbytes()
   trained.append(dict(case=case['name'],rule=case['revision_rule'],optimizer_steps=case['training']['step'],training_observations=case['training']['observations'],training_answers=case['training']['answers'],optimizer_tensor_storage_bytes=sum(storage.values()),training_seconds=case['training_seconds']))
def write(path,records):
 with path.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=records[0]);w.writeheader();w.writerows(records)
write(a.root/'analysis/resource_accounting.csv',rows);write(a.root/'analysis/training_accounting.csv',trained)
report=dict(evaluation_weightsets=len(rows),observed_evaluation_writes=sum(r['observed_writes'] for r in rows),trained_cases=len(trained),actual_optimizer_steps=sum(r['optimizer_steps'] for r in trained),actual_training_observations=sum(r['training_observations'] for r in trained),actual_training_answers=sum(r['training_answers'] for r in trained),scope='Persistent tensor storage and counted updates; excludes Python overhead/transient buffers. Candidate adds projection arithmetic; not FLOPmatching or barelatency benchmark. Initial controls are not additional training.')
(a.root/'analysis/resource_accounting.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
