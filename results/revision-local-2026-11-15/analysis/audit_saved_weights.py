"""Independent actual checkpoint/optimizer tensor audit; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import torch

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--parents',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(1)
meta=json.loads((a.root/'metadata.json').read_text())
mode=meta['args']['mode']

def digest(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        if name=='route_table':
            continue
        t=tensor.detach().cpu().contiguous()
        h.update(name.encode()+b'\0'+str(tuple(t.shape)).encode()+b'\0'+t.numpy().tobytes())
    return h.hexdigest()

checks=[]
initial_by_seed={}
for case in meta['cases']:
    folder=a.root/'runs'/case['name']
    parent_path=a.parents/case['parent']
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==case['parent_sha256']
    assert parent['format']=='clmm.adapter.reference.v1'
    assert parent['training']['step']==(1200 if mode=='frozen' else 0)
    original=parent['model_state']
    assert len(original)==9
    assert digest(original)==case['initial_slow_sha256']
    for stage,filename in [('final','checkpoint.pt')]+([('initial','initial_checkpoint.pt')] if mode=='train' else []):
        path=folder/filename
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['format']=='clmm.revision.local.v1'
        assert payload['config']==case['config']
        assert payload['parent_sha256']==case['parent_sha256']
        assert payload['source_commit']==case['source_commit']==meta['source_commit']
        state=payload['model_state']
        assert set(state)==set(original)|{'route_table'}
        assert state['route_table'].dtype==torch.int64
        assert state['route_table'].tolist()==case['route_table']
        for name,tensor in state.items():
            assert tensor.device.type=='cpu'
            assert tensor.layout==torch.strided and torch.isfinite(tensor).all()
            if name!='route_table':
                assert tensor.dtype==torch.float32 and tensor.shape==original[name].shape
        slow=digest(state)
        assert slow==payload['slow_sha256']==case['initial_slow_sha256' if stage=='initial' else 'slow_sha256']
        unchanged=[name for name in original if torch.equal(state[name],original[name])]
        if mode=='frozen' or stage=='initial':
            assert len(unchanged)==9
            assert payload['training']['step']==0
            assert payload['training']['observations']==payload['training']['answers']==0
        else:
            assert payload['training']==case['training']
            assert payload['training']['step']==1200
            assert len(unchanged)<9,'No learned tensor changed across 1200 steps'
        for name in ('codebook.keys','codebook.values','router_prototypes'):
            assert torch.equal(state[name],original[name])
        checks.append(dict(case=case['name'],stage=stage,slow_sha256=slow,
                           parent_tensors_bitwise_unchanged=unchanged,
                           training=payload['training'],route_table_bytes=state['route_table'].numel()*8))
        if stage=='initial':
            seed=case['parent_seed']
            if seed in initial_by_seed:
                assert all(torch.equal(state[k],v) for k,v in initial_by_seed[seed].items())
            else:
                initial_by_seed[seed]={k:v.clone() for k,v in state.items() if k!='route_table'}
    if mode=='train':
        optimizer=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
        groups=optimizer['param_groups']
        assert len(groups)==1
        group=groups[0]
        assert group['lr']==.001 and group['weight_decay']==0 and group['betas']==(.9,.999)
        assert group['eps']==1e-8 and len(group['params'])==6
        assert len(optimizer['state'])==6
        for item in optimizer['state'].values():
            assert float(item['step'])==1200
            assert item['exp_avg'].shape==item['exp_avg_sq'].shape
            assert torch.isfinite(item['exp_avg']).all() and torch.isfinite(item['exp_avg_sq']).all()
            assert (item['exp_avg_sq']>=0).all()
report=dict(status='passed',mode=mode,source_commit=meta['source_commit'],cases=len(meta['cases']),
            checked_checkpoint_payloads=len(checks),checkpoints=checks,
            no_fast_state_in_checkpoint=True,optimizer_steps_verified=1200 if mode=='train' else 0,
            scope='Actual CPU-loaded saved checkpoint/optimizer tensors; no inference/training rerun.')
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='checkpoints'},indent=2))
