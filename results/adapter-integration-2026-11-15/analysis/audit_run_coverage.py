"""Verify the complete declared invocation/file set after archive transport."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    pipeline=read(args.root/'pipeline.json');freeze=read(args.root/'analysis/prospective_freeze_review.json')
    assert pipeline['status']=='succeeded' and not pipeline['skipped']
    assert sha(args.root/'design.json')==freeze['design_sha256']
    expected=['throughput-gpu']+[f'neural-anchor-{seed}' for seed in (90511,90515)]
    expected += [f'{rule}-{routing}-{seed}' for seed in (90511,90515) for rule,routing in
                 (('gradient','dense'),('gradient','top1'),('hebbian','dense'),('hebbian','top1'))]
    expected += [f'fork-{rule}-{routing}-{seed}' for seed in (90511,90515) for rule,routing in
                 (('gradient','dense'),('gradient','top1'),('hebbian','dense'),('hebbian','top1'))]
    assert [r['name'] for r in pipeline['runs']]==expected
    required={};runs=[];optimizer_checks=[];session_checks=[]
    def require(path):
        assert path.is_file() and path.stat().st_size>0,path
        required[path.relative_to(args.root).as_posix()]=sha(path)
    for invocation in pipeline['runs']:
        name=invocation['name'];run=args.root/name;require(run/'metadata.json');meta=read(run/'metadata.json')
        assert invocation['status']=='succeeded' and invocation['exit_code']==0 and meta['status']=='succeeded'
        command=list(invocation['command'])
        assert command[1]=='-B';command.pop(1)
        assert command==meta['command']
        assert meta['source_commit']==pipeline['source_commit'] and meta['source_commit'].startswith(freeze['source_commit'])
        assert meta['design_sha256']==freeze['design_sha256']
        for file in ('design.json','training_data.npz','training_plan.json'):require(run/file)
        for file in meta['source_sha256']:require(run/'sources'/file)
        margs=meta['args'];probe=name=='throughput-gpu'
        assert margs['steps']==(10 if probe else 1200) and margs['batch_size']==32
        assert margs['eval_episodes']==(16 if probe else 128) and margs['pressure_episodes']==(16 if probe else 64)
        assert margs['eval_batch_size']==16 and margs['dev_seed']==(90603 if probe else 90623) and margs['device']=='cuda'
        stage_names=['evaluation']
        if meta['offline_training']:
            stage_names.append('initial-evaluation')
            for file in ('initial_checkpoint.pt','checkpoint.pt','optimizer.pt','training.csv'):require(run/file)
            checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
            optimizer=torch.load(run/'optimizer.pt',map_location='cpu',weights_only=True)
            assert len(optimizer['param_groups'])==1 and len(optimizer['state'])==6
            assert optimizer['param_groups'][0]['lr']==.001 and optimizer['param_groups'][0]['weight_decay']==0
            names=('key_encoder','value_encoder','decoder','W0','alpha','raw_eta')
            for index,name2 in enumerate(names):
                state=optimizer['state'][index]
                assert float(state['step'])==margs['steps']
                for moment in ('exp_avg','exp_avg_sq'):
                    tensor=state[moment]
                    assert tensor.shape==checkpoint['model_state'][name2].shape and torch.isfinite(tensor).all()
            optimizer_checks.append(dict(run=name,step=margs['steps'],parameter_states=6,finite=True))
        for stage in stage_names:
            for file in ('episodes.json','plans.json','predictions.csv','resources.csv','scores.npy','summary.csv'):require(run/stage/file)
        if margs['mode']!='fork':
            first=read(run/'evaluation/episodes.json')['pressure_k8_d16'][:16]
            fingerprint=hashlib.sha256(json.dumps(meta['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            for point,delay in (('prefill',0),('midblock',1)):
                file=run/f'evaluation/example_{point}_session.pt';require(file)
                saved=torch.load(file,map_location='cpu',weights_only=True)
                size=len(first)
                assert saved['batch_size']==size and saved['arm']=='periodic' and saved['prime_count']==10
                assert saved['delay_count']==delay and saved['replay_count']==0 and not saved['closed']
                assert saved['episode_seeds']==[ep['seed'] for ep in first]
                assert saved['model_method']==meta['method'] and saved['model_config_fingerprint']==fingerprint
                assert saved['model_state']['config']==meta['config'] and saved['model_state']['config_fingerprint']==fingerprint
                assert saved['model_state']['write_count']==10+delay
                assert len(saved['policies'])==size
                for ep,policy in zip(first,saved['policies']):
                    assert policy['config']['seed']==ep['seed'] and policy['next_source']==10+delay
                    assert len(policy['buffer'])==8 and len(policy['snapshot'])==(4 if delay else 0)
                session_checks.append(dict(run=name,point=point,batch=size,write_count=10+delay,metadata_checkpoint_binding=True))
        runs.append(dict(run=name,mode=margs['mode'],stages=stage_names,source_files=len(meta['source_sha256'])))
    result=dict(status='passed',scope='Complete prospective invocation/file coverage, optimizer payloads and saved session bundles; not an inference rerun',
        source_commit=pipeline['source_commit'],invocations=len(runs),main_invocations=len(runs)-1,
        required_files=len(required),required_sha256=required,runs=runs,optimizer_checks=optimizer_checks,session_checks=session_checks,
        transport_note='Required contents independently checked despite the reported tar directory-mtime warning; transport compressed-file SHA verified by coordinator')
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('required_sha256','runs','optimizer_checks','session_checks')},indent=2))

if __name__=='__main__':main()
