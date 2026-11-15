"""Independent equation, state and bounded-session checks; no training."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.repo))
    import torch
    from clmm.hebbian import AssociativeAdapter,AssociativeConfig
    from clmm.pretrain.kv_control import KVState,physical_state_bytes
    from clmm.pretrain.kv_correlated_portable import PortableCorrelatedConfig,load_portable_checkpoint
    from clmm.pretrain.kv_streams import episode,replay_plan
    from clmm.sleep import SleepSession
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources={name:sha(args.repo/name) for name in ('clmm/hebbian/adapter.py','clmm/sleep/adapter.py',
             'clmm/pretrain/kv_scheduler.py','clmm/pretrain/kv_correlated_portable.py')}
    start=time.perf_counter()
    torch.set_num_threads(1)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    code_config=PortableCorrelatedConfig(**payload['config'])
    keys,values=payload['model_state']['codebook.keys'],payload['model_state']['codebook.values']
    reports=[]
    initial_by_seed={}
    for seed in (90511,90515):
        for rule in ('gradient','hebbian'):
            for routing in ('dense','top1'):
                torch.manual_seed(seed)
                model=AssociativeAdapter(AssociativeConfig(code_config,update_rule=rule,routing=routing),keys,values)
                slow={name:t.detach().clone() for name,t in model.state_dict().items()}
                if seed not in initial_by_seed:initial_by_seed[seed]=slow
                assert all(torch.equal(t,initial_by_seed[seed][name]) for name,t in slow.items())
                # Crafted nonzero state and nontrivial alpha make the analytic
                # derivative test discriminate Hebb from residual correction.
                with torch.no_grad():
                    model.W0.copy_(torch.randn_like(model.W0)*.01)
                    model.alpha.copy_(.7+torch.randn_like(model.alpha)*.02)
                trace=(torch.randn(2,4,64,64)*.03).requires_grad_()
                state=KVState({'trace':trace},None,7,model.method,model.config.fingerprint)
                observed_keys=torch.tensor([5,29])
                observed_values=torch.tensor([3,11])
                p=torch.tanh(torch.einsum('bd,mdh->bmh',keys[observed_keys],model.key_encoder))
                q=torch.tanh(torch.einsum('bd,mdh->bmh',values[observed_values],model.value_encoder))
                effective=model.W0.unsqueeze(0)+model.alpha.unsqueeze(0)*trace
                predicted=torch.einsum('bmi,bmij->bmj',p,effective)
                if rule=='gradient':
                    derivative=torch.autograd.grad(.5*(predicted-q).square().sum(),trace)[0]
                    candidate=((1-model.eta)*trace-model.eta*derivative).clamp(-1,1)
                else:
                    candidate=((1-model.eta)*trace+model.eta*p.unsqueeze(-1)*q.unsqueeze(-2)).clamp(-1,1)
                prototypes=keys[[0,16,32,48]]
                similarity=torch.nn.functional.normalize(keys[observed_keys],dim=-1)@torch.nn.functional.normalize(prototypes,dim=-1).T
                route=(torch.ones(2,4,dtype=torch.bool) if routing=='dense' else torch.nn.functional.one_hot(similarity.argmax(-1),4).bool())
                expected=torch.where(route[:,:,None,None],candidate,trace)
                slow={name:t.detach().clone() for name,t in model.state_dict().items()}
                rng=torch.random.get_rng_state().clone()
                with patch('torch.autograd.grad',side_effect=AssertionError('Online engine called autograd.grad')):
                    actual=model.write(observed_keys,observed_values,state)
                error=float((actual.weights['trace']-expected).abs().max().detach())
                assert error<=2e-7
                assert torch.equal(actual.weights['trace'][~route],trace[~route])
                assert torch.equal(model.route(observed_keys),route)
                assert actual.write_count==8
                exported=model.export_state(actual)
                query=torch.tensor([[5,8],[29,17]])
                before=model.read(query,actual)
                raw=io.BytesIO();torch.save(exported,raw);raw.seek(0)
                restored=model.import_state(torch.load(raw,weights_only=True))
                assert torch.equal(before,model.read(query,restored))
                nxt=model.write(torch.tensor([8,17]),torch.tensor([4,9]),actual)
                resumed=model.write(torch.tensor([8,17]),torch.tensor([4,9]),restored)
                assert torch.equal(nxt.weights['trace'],resumed.weights['trace'])
                assert all(torch.equal(t,model.state_dict()[name]) for name,t in slow.items())
                assert torch.equal(rng,torch.random.get_rng_state())
                assert physical_state_bytes(actual)==2*4*64*64*4
                reports.append(dict(seed=seed,rule=rule,routing=routing,max_equation_error=error,
                    per_stream_fast_tensor_bytes=physical_state_bytes(actual)//2,
                    actual_active_modules=route.sum(-1).tolist(),outer_parameters_unchanged=True,
                    online_autograd_engine_calls=0,continuation_exact=True))
    model,_=load_portable_checkpoint(args.checkpoint,torch.device('cpu'))
    model.eval().requires_grad_(False)
    sessions=[]
    with torch.no_grad():
        for arm in ('online','repeat','continuous','periodic'):
            eps=[episode(seed,8,3,2,pressure=True) for seed in (90601,90603)]
            session=SleepSession(model,2,arm,[ep['seed'] for ep in eps])
            emitted=[[],[]]
            for index in range(len(eps[0]['keys'])):
                ki=torch.tensor([ep['keys'][index] for ep in eps])
                vi=torch.tensor([ep['values'][index] for ep in eps])
                rows=(session.prime if index<eps[0]['boundary'] else session.observe)(ki,vi)
                for target,events in zip(emitted,rows):target.extend(events)
                if index==eps[0]['boundary']:
                    bundle=session.export_state()
                    bundle['policies']=json.loads(json.dumps(bundle['policies']))
                    raw=io.BytesIO();torch.save(bundle,raw);raw.seek(0)
                    session=SleepSession.from_state(model,torch.load(raw,weights_only=True))
            for target,events in zip(emitted,session.finish()):target.extend(events)
            assert emitted==[replay_plan(ep,arm)['events'] for ep in eps]
            query=torch.tensor([[q['key'] for q in ep['queries']] for ep in eps])
            saved=session.model.export_state(session.state)
            assert torch.equal(session.read(query),session.model.read(query,session.state))
            assert all(torch.equal(t,saved['weights'][name]) for name,t in session.state.weights.items())
            sessions.append(dict(arm=arm,terminal_partial_delay=3,total_writes=session.state.write_count,
                                 exact_frozen_event_parity=True,midblock_resume=True))
    assert time.perf_counter()-start<60
    assert sources=={name:sha(args.repo/name) for name in sources},'Sources changed during proof'
    result=dict(status='passed',scope='Independent bounded CPU equations/persistence/session proof, no training or efficacy',
        elapsed_seconds=time.perf_counter()-start,source_sha256=sources,checkpoint_sha256=sha(args.checkpoint),
        torch=str(torch.__version__),equation_checks=reports,session_checks=sessions,
        identical_initial_parameters_across_four_variants=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('equation_checks','session_checks')},indent=2))


if __name__=='__main__':main()
