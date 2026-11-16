"""Tensor-only model/optimizer/checkpoint provenance audit; never builds a model."""
import argparse,collections,hashlib,json,pathlib
import torch

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf8'))
def state_sha(state):
 h=hashlib.sha256()
 for name,t in sorted(state.items()):
  h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def load(path):return torch.load(path,map_location='cpu',weights_only=True)
def check_state(path,config,parent_state):
 payload=load(path);state=payload['model_state'];assert payload['config']==config
 assert list(state)==list(parent_state) and all(state[k].shape==parent_state[k].shape and state[k].dtype==parent_state[k].dtype for k in state)
 assert sum(t.numel() for t in state.values())==839248
 assert all(torch.isfinite(t).all() for t in state.values())
 for k in state:
  if '.memory_model.' in k or k.endswith('.rotary_emb.freqs'):assert torch.equal(state[k],parent_state[k]),k
 return payload,state_sha(state)
def optimizer(path,step,state):
 opt=load(path);assert set(opt)=={'state','param_groups'} and len(opt['param_groups'])==1
 group=opt['param_groups'][0];ids=group['params'];assert len(ids)==len(set(ids))==len(state) and set(opt['state'])<=set(ids)
 assert ids==list(range(len(state)))
 names=list(state);inactive={i for i,k in enumerate(names) if '.memory_model.' in k or k.endswith('.rotary_emb.freqs')}
 assert set(ids)-set(opt['state'])==inactive and len(inactive)==8
 assert group['lr']==.0003 and group['weight_decay']==.01 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and not group['amsgrad']
 shapes=collections.Counter(tuple(v.shape) for i,v in enumerate(state.values()) if i not in inactive);actual=collections.Counter()
 for i in sorted(opt['state']):
  entry=opt['state'][i];assert set(entry)=={'step','exp_avg','exp_avg_sq'} and float(entry['step'])==step
  a,b=entry['exp_avg'],entry['exp_avg_sq'];assert a.shape==b.shape==state[names[i]].shape and a.dtype==b.dtype==torch.float32
  assert torch.isfinite(a).all() and torch.isfinite(b).all() and (b>=0).all();actual[tuple(a.shape)]+=1
 assert shapes==actual
 return dict(path=str(path),sha256=sha(path),steps_exact=step,states=len(opt['state']),registered_optimizer_ids=len(ids),inactive_names=[names[i] for i in sorted(inactive)],tensor_elements=sum(v['exp_avg'].numel() for v in opt['state'].values()),finite=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--packet',type=pathlib.Path,required=True);p.add_argument('--runs',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();torch.set_num_threads(1)
 pm=read(a.packet/'metadata.json');parents={}
 for row in read(a.packet/'parents.json'):
  payload=load(a.packet/row['path']);assert sha(a.packet/row['path'])==row['sha256'] and state_sha(payload['model_state'])==row['state_sha256'];parents[row['seed']]=(row,payload)
 records=[]
 for f in sorted(a.runs.rglob('metadata.json')):
  m=read(f)
  if m.get('mode') not in ('initial','train') or m.get('data_sha256')!=pm['files_sha256']:continue
  path=f.parent;row,parent=parents[m['parent_seed']];config=dict(parent['config']);config['segment_len']=128
  payload,h=check_state(path/'checkpoint.pt',config,parent['model_state']);assert h==m['final_state_sha256'] and sha(path/'checkpoint.pt')==m['checkpoint_sha256']
  assert payload['source_info']['source_commit']==pm['source_commit'] and payload['source_info']['source_sha256']==pm['source_sha256']
  info=payload['training_info'];assert info['parent_init_seed']==m['parent_seed'] and info['parent_checkpoint_sha256']==row['sha256'] and info['encoding']==m['encoding'] and info['segment']==128 and info['fixed_terminal']
  record=dict(run=str(path),mode=m['mode'],parent=m['parent_seed'],encoding=m['encoding'],parameters=839248,state_sha256=h,checkpoint_sha256=sha(path/'checkpoint.pt'),finite=True,optimizers=[])
  if m['mode']=='initial':assert h==row['state_sha256'] and info['steps']==0 and all(torch.equal(payload['model_state'][k],v) for k,v in parent['model_state'].items())
  else:
   assert info['steps']==pm['steps']==2000 and h!=row['state_sha256']
   record['optimizers'].append(optimizer(path/'optimizer.pt',2000,payload['model_state']))
   midpoint,mh=check_state(path/'midpoint.pt',config,parent['model_state']);assert midpoint['source_info']['source_commit']==pm['source_commit'] and midpoint['training_info']['steps']==1000
   assert midpoint['training_info']['selection']=='Recovery only, fixed terminal2000 primary'
   record['midpoint_sha256']=sha(path/'midpoint.pt');record['midpoint_state_sha256']=mh
   record['optimizers'].append(optimizer(path/'midpoint-optimizer.pt',1000,midpoint['model_state']))
  records.append(record)
 assert len(records)==8 and sum(r['mode']=='train' for r in records)==4
 result=dict(status='passed',kind='tensor_only_checkpoint_optimizer_audit',models_constructed=False,model_inference=False,source_commit=pm['source_commit'],initial_checkpoints=4,terminal_checkpoints=4,recovery_midpoints=4,optimizer_states=8,records=records)
 a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
