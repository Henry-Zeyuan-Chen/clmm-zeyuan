"""Measured state/budget accounting; candidate arithmetic remains dense."""
import argparse,csv,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.repo))
import torch
from scripts.routing_decay_experiment import load_checkpoint
meta=json.loads((a.root/'main/metadata.json').read_text());rows=[]
for case in meta['cases']:
 folder=a.root/'main/runs'/case['name'];model,_=load_checkpoint(folder/'checkpoint.pt',torch.device('cpu'))
 with (folder/'evaluation/resources.csv').open() as h:resources=list(csv.DictReader(h))
 slow=sum(t.numel()*t.element_size() for t in model.parameters());fixed=sum(t.numel()*t.element_size() for t in model.buffers())
 state=model.initial_state(1);fast=state.weights['trace'].untyped_storage().nbytes()
 row=dict(case=case['name'],registered_parameters=sum(t.numel() for t in model.parameters()),optimized_parameters=0,new_optimizer_steps=0,new_training_labels=0,slow_parameter_bytes=slow,fixed_buffer_bytes=fixed,routing_table_bytes=model.route_table.untyped_storage().nbytes(),fast_bytes_per_stream=fast,
 observed_writes=sum(int(r['actual_writes'])*int(r['batch']) for r in resources),retained_association_updates=sum(int(r['active_module_updates']) for r in resources),decay_applications=sum(int(r['decay_applications']) for r in resources),
 instrumented_evaluation_seconds=sum(float(r['elapsed_seconds']) for r in resources),case_seconds=case['elapsed_seconds'],main_device=meta['environment']['device'],candidate_modules_computed_per_write=4,effective_modules_retained_per_write=1)
 assert slow==327684 and fixed==22016 and fast==65536 and row['routing_table_bytes']==512
 assert row['observed_writes']==row['retained_association_updates']==34688
 assert row['decay_applications']==row['observed_writes']*(1 if case['inactive_decay']=='protected' else 4)
 rows.append(row)
with (a.root/'analysis/resource_accounting.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
report=dict(cases=len(rows),new_training_steps=0,new_training_labels=0,observed_evaluation_writes=sum(r['observed_writes'] for r in rows),main_elapsed_seconds=meta['elapsed_seconds'],main_device=meta['environment']['device'],environment=meta['environment'],scope='Instrumented inference cost, not a bare latency/FLOP benchmark. Same dense candidate arithmetic for content/random. Inactive global decay adds3multiplications intentionally; allocated GPU time also includes CPU execution.')
(a.root/'analysis/resource_accounting.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
