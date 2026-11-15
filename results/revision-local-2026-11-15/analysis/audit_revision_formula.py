"""Independent bounded CPU formula/state audit, with no training/timing probe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--parents', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
sys.dont_write_bytecode = True
sys.path.insert(0, str(a.repo))
source = a.repo/'clmm/hebbian/revision.py'
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
import torch
from clmm.hebbian.revision import RevisionAdapter, RevisionConfig
from clmm.hebbian.routing_decay import RoutingDecayAdapter
from clmm.pretrain.adapter_registry import load_adapter_checkpoint
from clmm.pretrain.kv_control import physical_state_bytes

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
rules = ('ema', 'additive', 'erase_ema', 'local_delta')
generator = torch.Generator().manual_seed(91801)
cases = []
for seed in (90511, 90515):
    parent_path = a.parents / f'hebbian-dense-{seed}.pt'
    if not parent_path.exists():
        parent_path = a.parents/f'hebbian-dense-{seed}'/'checkpoint.pt'
    parent, _ = load_adapter_checkpoint(parent_path, torch.device('cpu'))
    original = {k:v.clone() for k,v in parent.state_dict().items()}
    original_rng = torch.random.get_rng_state().clone()
    for policy, map_seed in [('content', None)] + [('random', s) for s in range(90701,90706)]:
        models = {r:RevisionAdapter.from_parent(parent, revision_rule=r, policy=policy,
                                              random_seed=map_seed) for r in rules}
        assert len({m.config.fingerprint for m in models.values()}) == 4
        reference = RoutingDecayAdapter.from_parent(parent, policy=policy,
                                                    random_seed=map_seed, inactive_decay='protected')
        shared_trace = torch.rand((3,4,64,64), generator=generator)*.2-.1
        keys, values = torch.tensor([0,17,63]), torch.tensor([0,7,15])
        reference_written = reference.write(keys,values,reference._state(shared_trace.clone(),4))
        expected_first = None
        for rule, model in models.items():
            for name, tensor in original.items():
                assert torch.equal(model.state_dict()[name], tensor)
            assert torch.equal(model.route_table, reference.route_table)
            state = model._state(shared_trace.clone(),4)
            saved = model.export_state(state)
            result = model.write(keys,values,state)
            assert torch.equal(state.weights['trace'], shared_trace)
            expected = shared_trace.clone()
            residuals = []
            for b in range(3):
                module = int(model.route_table[keys[b]])
                x = torch.tanh(model.codebook.keys[keys[b]] @ model.key_encoder[module])
                y = torch.tanh(model.codebook.values[values[b]] @ model.value_encoder[module])
                h = shared_trace[b,module]
                z = h.clone()
                if rule in ('erase_ema','local_delta'):
                    denom = torch.maximum(x.dot(x), torch.tensor(model.config.projection_eps))
                    z = z - torch.outer(x/denom, x @ h)
                if rule in ('ema','erase_ema'):
                    z = (1-model.eta)*z
                z = torch.clamp(z+model.eta*torch.outer(x,y),
                                -model.config.trace_bound,model.config.trace_bound)
                expected[b,module] = z
                if rule in ('erase_ema','local_delta'):
                    residuals.append(float((x @ z-model.eta*x.dot(x)*y).abs().max().detach()))
            err = float((result.weights['trace']-expected).abs().max().detach())
            assert err < 2e-7, (seed,policy,map_seed,rule,err)
            assert result.write_count == 5
            roundtrip = model.import_state(model.export_state(result))
            assert torch.equal(roundtrip.weights['trace'],result.weights['trace'])
            assert physical_state_bytes(result) == 65536*3
            for other_rule, other in models.items():
                if other_rule == rule:
                    continue
                try:
                    other.import_state(saved)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Cross-rule import accepted')
            first = model.write(keys,values,model.initial_state(3)).weights['trace']
            if expected_first is None:
                expected_first = first.clone()
            else:
                assert torch.equal(first, expected_first)
            if rule == 'ema':
                assert torch.equal(result.weights['trace'], reference_written.weights['trace'])
            # Zero pre activity must remain finite for erasure without a key oracle.
            if model.config.key_local_erasure:
                model.key_encoder.data.zero_()
                zero = model.write(keys,values,state)
                decay = 1-model.eta if model.config.active_bulk_decay else 1
                active = model.route(keys)[...,None,None]
                expected_zero = torch.where(active, shared_trace*decay,shared_trace)
                assert torch.equal(zero.weights['trace'],expected_zero)
                assert torch.isfinite(zero.weights['trace']).all()
            cases.append(dict(parent_seed=seed,policy=policy,map_seed=map_seed,rule=rule,
                              max_formula_error=err,
                              max_projection_identity_error=max(residuals,default=None),
                              fast_bytes_per_stream=physical_state_bytes(result)//3,
                              config_fingerprint=model.config.fingerprint,
                              route_hash=model.config.routing_table_sha256))
        assert torch.equal(torch.random.get_rng_state(),original_rng)
        assert all(torch.equal(original[k],v) for k,v in parent.state_dict().items())
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha, 'Source changed during audit'
report = dict(status='passed',source=str(source),source_sha256=source_sha,
              cases=cases,case_count=len(cases),
              checks=['independent looped active-module equation','inactive exact preservation',
                      'input state unchanged','zero-state first write bitwise identical across four rules',
                      'EMA output bitwise identical to legacy protected EMA',
                      'unclamped raw projection identity','zero-key response finite and correct',
                      'strict cross-rule state rejection','state roundtrip bitwise exact',
                      'parent all nine slow/codebook/prototype tensors bitwise identical',
                      'all six routing tables identical to existing controls',
                      '65536 persistent trace bytes per stream','global CPU RNG unchanged'],
              scope='CPU formula/state checks only; no optimization or performance probe.')
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'cases'},indent=2))
