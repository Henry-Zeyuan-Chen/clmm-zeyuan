"""Read checkpoint/optimizer tensors and source snapshots without building a model."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from audit_packet import read_json,sha


def tensor_state_sha(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        digest.update(name.encode());digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def check_finite(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for child in value.values():check_finite(child)
    elif isinstance(value,(list,tuple)):
        for child in value:check_finite(child)

def audit(packet,folders,repo=None):
    metadata=read_json(packet/'metadata.json');sources=[]
    for name,digest in metadata['source_sha256'].items():
        assert sha(packet/'sources'/name)==digest
        row=dict(path=name,sha256=digest,snapshot_matches=True,git_blob_matches=None)
        if repo:
            result=subprocess.run(['git','show',f"{metadata['source_commit']}:{name}"],cwd=repo,capture_output=True)
            if result.returncode==0:row['git_blob_matches']=hashlib.sha256(result.stdout).hexdigest()==digest
            row['current_checkout_matches']=sha(repo/name)==digest
        sources.append(row)
    for name,digest in metadata['files_sha256'].items():assert sha(packet/name)==digest
    parents={};parent_rows=[]
    for parent in read_json(packet/'parents.json'):
        path=packet/parent['path'];assert sha(path)==parent['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=True);check_finite(payload['model_state'])
        assert payload['training_info']['init_seed']==parent['seed'] and payload['training_info'].get('task') is None
        assert payload['config']==parent['config']
        state=payload['model_state'];assert tensor_state_sha(state)==parent['state_sha256']
        assert sum(t.numel() for t in state.values())==839248
        parents[parent['seed']]=state
        parent_rows.append(dict(seed=parent['seed'],sha256=parent['sha256'],state_sha256=parent['state_sha256'],tensor_elements=839248,all_finite=True))
    cases=[]
    for folder in folders:
        meta=read_json(folder/'metadata.json');assert meta['status']=='succeeded'
        assert sha(folder/'checkpoint.pt')==meta['checkpoint_sha256']
        payload=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        state=payload['model_state'];check_finite(state);assert tensor_state_sha(state)==meta['final_state_sha256']
        parent_state=parents[meta['parent_seed']]
        assert set(state)==set(parent_state)
        assert all(state[name].shape==parent_state[name].shape and state[name].dtype==parent_state[name].dtype for name in state)
        expected_config=dict(meta['parent']['config']);expected_config['segment_len']=meta['segment'];assert payload['config']==expected_config
        assert sum(t.numel() for t in state.values())==839248
        assert payload['training_info']['parent_checkpoint_sha256']==meta['parent']['sha256']
        assert payload['training_info']['steps']==meta['completed_steps']
        changes=sum(not torch.equal(state[name],parent_state[name]) for name in state)
        optimizer_steps=[];optimizer_elements=0;no_state_names=[]
        if meta['mode']=='initial':assert changes==0 and meta['completed_steps']==0
        else:
            optimizer=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True);check_finite(optimizer)
            optimizer_steps=sorted({int(s['step'].item()) for s in optimizer['state'].values()})
            assert optimizer_steps==[meta['completed_steps']]
            optimizer_elements=sum(s['exp_avg'].numel() for s in optimizer['state'].values())
            assert 0<optimizer_elements<=839248
            parameter_ids=[i for group in optimizer['param_groups'] for i in group['params']]
            assert parameter_ids==list(range(len(state)))
            names=list(state)
            assert all(optimizer['state'][i]['exp_avg'].shape==state[names[i]].shape for i in optimizer['state'])
            no_state_names=[names[i] for i in parameter_ids if i not in optimizer['state']]
            assert all(s['exp_avg'].shape==s['exp_avg_sq'].shape for s in optimizer['state'].values())
        cases.append(dict(case=folder.name,parent=meta['parent_seed'],segment=meta['segment'],mode=meta['mode'],
            unchanged_parameter_shapes=True,tensor_elements=839248,all_finite=True,changed_tensors=changes,optimizer_steps=optimizer_steps,
            checkpoint_sha256=meta['checkpoint_sha256'],optimizer_state_elements=optimizer_elements,no_optimizer_state_tensors=no_state_names))
    return dict(status='passed',scope='CPU tensor and source inspection only; no model built or executed',
                environment=dict(torch=str(torch.__version__),numpy=np.__version__),source_commit=metadata['source_commit'],
                source_files=sources,parents=parent_rows,cases=cases)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--case',nargs='*',type=Path,default=[])
    parser.add_argument('--repo',type=Path);parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'provenance_audit.json')
    args=parser.parse_args();result=audit(args.packet,args.case,args.repo)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:value for key,value in result.items() if key!='source_files'}))


