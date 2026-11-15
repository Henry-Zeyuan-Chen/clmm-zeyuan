"""Independent CPU equation/state checks on the routing/decay intervention."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
sys.dont_write_bytecode = True


def unchanged(before, model):
    assert set(before) == set(model.state_dict())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--parents', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo))
    from clmm.pretrain.adapter_registry import load_adapter_checkpoint, fork_associative
    from clmm.pretrain.kv_control import physical_state_bytes
    from clmm.hebbian.routing_decay import RoutingDecayAdapter
    results = []
    policies = [('content', None)] + [('random', seed) for seed in range(90701, 90706)]
    batch = 8
    query = torch.arange(32).reshape(batch, 4)
    generator = torch.Generator().manual_seed(90887)
    max_formula_error = 0.
    for rule in ('gradient', 'hebbian'):
        for seed in (90511, 90515):
            name = f'{rule}-dense-{seed}'
            parent, _ = load_adapter_checkpoint(args.parents/name/'checkpoint.pt', torch.device('cpu'))
            parent.eval().requires_grad_(False)
            parent_before = {n: t.clone() for n, t in parent.state_dict().items()}
            old_top1 = fork_associative(parent, update_rule=rule, routing='top1')
            content_routes = old_top1.route(torch.arange(64)).long().argmax(-1)
            fingerprints = set()
            prior_model = None
            for policy, map_seed in policies:
                siblings = []
                for decay in ('protected', 'global'):
                    cpu_rng = torch.random.get_rng_state().clone()
                    np_rng = pickle.dumps(np.random.get_state())
                    model = RoutingDecayAdapter.from_parent(parent, policy=policy, random_seed=map_seed, inactive_decay=decay)
                    assert torch.equal(cpu_rng, torch.random.get_rng_state())
                    assert np_rng == pickle.dumps(np.random.get_state())
                    assert all(torch.equal(t, model.state_dict()[n]) for n, t in parent_before.items())
                    assert all(not v.requires_grad for v in model.parameters())
                    assert torch.equal(torch.bincount(model.route_table, minlength=4), torch.bincount(content_routes, minlength=4))
                    expected_table = content_routes if policy == 'content' else torch.from_numpy(np.random.Generator(np.random.PCG64(map_seed)).permutation(content_routes.numpy()))
                    assert torch.equal(expected_table, model.route_table)
                    assert model.config.fingerprint not in fingerprints
                    fingerprints.add(model.config.fingerprint)
                    initial = model.initial_state(batch)
                    assert initial.write_count == 0 and torch.count_nonzero(initial.weights['trace']) == 0
                    assert physical_state_bytes(initial) == batch * 65536
                    trace = torch.randn(batch, 4, 64, 64, generator=generator) * .04
                    state = model._state(trace.clone(), 17)
                    before_trace = trace.clone()
                    slow_before = {n: t.clone() for n, t in model.state_dict().items()}
                    keys = torch.tensor([0, 1, 7, 16, 23, 32, 48, 63])
                    values = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14])
                    with torch.no_grad():
                        actual = model.write(keys, values, state)
                        pre = torch.tanh(torch.einsum('bd,mdh->bmh', model.codebook.keys[keys], model.key_encoder))
                        post = torch.tanh(torch.einsum('bd,mdh->bmh', model.codebook.values[values], model.value_encoder))
                        # Module-by-module analytic evaluation, independently of adapter.write.
                        expected = trace.clone()
                        for b in range(batch):
                            for module in range(4):
                                active = int(model.route_table[keys[b]]) == module
                                if active:
                                    if rule == 'hebbian':
                                        change = torch.outer(pre[b, module], post[b, module])
                                    else:
                                        effective = model.W0[module] + model.alpha[module]*trace[b, module]
                                        residual = pre[b, module] @ effective - post[b, module]
                                        change = -model.alpha[module] * torch.outer(pre[b, module], residual)
                                    expected[b, module] = ((1-model.eta)*trace[b, module] + model.eta*change).clamp(-model.config.trace_bound, model.config.trace_bound)
                                elif decay == 'global':
                                    expected[b, module] = (1-model.eta)*trace[b, module]
                        error = float((actual.weights['trace']-expected).abs().max())
                        max_formula_error = max(max_formula_error, error)
                        assert torch.allclose(actual.weights['trace'], expected, rtol=1e-6, atol=1e-7), error
                        assert actual.write_count == 18
                        assert torch.equal(state.weights['trace'], before_trace)
                        mask = model.route(keys)
                        assert mask.dtype == torch.bool and mask.shape == (batch, 4) and mask.sum().item() == batch
                        assert model.decay_mask(keys).sum().item() == batch*(1 if decay == 'protected' else 4)
                        for b in range(batch):
                            inactive = ~mask[b]
                            target = trace[b, inactive] if decay == 'protected' else (1-model.eta)*trace[b, inactive]
                            assert torch.equal(actual.weights['trace'][b, inactive], target)
                        payload = model.export_state(actual)
                        buffer = io.BytesIO()
                        torch.save(payload, buffer)
                        buffer.seek(0)
                        restored = model.import_state(torch.load(buffer, map_location='cpu', weights_only=True))
                        assert torch.equal(restored.weights['trace'], actual.weights['trace'])
                        before_read = restored.weights['trace'].clone()
                        score = model.read(query, restored)
                        assert torch.equal(before_read, restored.weights['trace'])
                        assert torch.equal(score, model.read(query, actual)) and torch.isfinite(score).all()
                        if prior_model is not None:
                            try:
                                prior_model.import_state(payload)
                            except ValueError:
                                pass
                            else:
                                raise AssertionError('Cross-policy state import accepted')
                        if policy == 'content' and decay == 'protected':
                            new_state = model.initial_state(batch)
                            old_state = old_top1.initial_state(batch)
                            for offset in range(12):
                                observed_keys = (keys+offset*5) % 64
                                observed_values = (values+offset*3) % 16
                                new_state = model.write(observed_keys, observed_values, new_state)
                                old_state = old_top1.write(observed_keys, observed_values, old_state)
                                assert torch.equal(new_state.weights['trace'], old_state.weights['trace'])
                                assert torch.equal(model.read(query, new_state), old_top1.read(query, old_state))
                    unchanged(slow_before, model)
                    unchanged(parent_before, parent)
                    assert torch.equal(cpu_rng, torch.random.get_rng_state())
                    assert np_rng == pickle.dumps(np.random.get_state())
                    siblings.append((model, actual.weights['trace']))
                    prior_model = model
                    results.append({'parent': name, 'policy': policy, 'random_seed': map_seed,
                                    'inactive_decay': decay, 'max_formula_error': error,
                                    'fingerprint': model.config.fingerprint,
                                    'routing_table_sha256': model.routing_table_sha256})
                # Re-evaluate the same input H to demand bit-exact active equality.
                protected, global_model = siblings[0][0], siblings[1][0]
                shared = protected._state(before_trace.clone(), 17)
                global_shared = global_model._state(before_trace.clone(), 17)
                a = protected.write(keys, values, shared).weights['trace']
                b = global_model.write(keys, values, global_shared).weights['trace']
                active = protected.route(keys)[..., None, None].expand_as(a)
                assert torch.equal(a[active], b[active])
            assert len(fingerprints) == 12
    out = {'all_checks_passed': True, 'scope': 'independent CPU equations/state validation; not GPU or training reproduction',
           'cases': len(results), 'batch': batch, 'old_top1_content_protected_exact_steps_per_parent': 12,
           'maximum_coordinate_formula_error': max_formula_error,
           'source_sha256': hashlib.sha256((args.repo/'clmm/hebbian/routing_decay.py').read_bytes()).hexdigest(),
           'checks': ['exact parent learned/codebook/prototype tensors', 'zero initial H;64KiB perstream',
                      'private fixed tables and exact cardinalities', 'independent two-rule analytic equations',
                      'exact sibling active results and expected inactive traces', 'unchanged inputs/slow weights/RNG',
                      'pure reads', 'serialization exact', 'cross-policy imports rejected', 'all12 policy fingerprints distinct'],
           'results': results}
    args.out.write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('all_checks_passed', 'cases', 'maximum_coordinate_formula_error', 'source_sha256')}))


if __name__ == '__main__':
    main()
