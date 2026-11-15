"""Independent CPU-only parent/data audit; creates no training or model updates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
sys.dont_write_bytecode = True


def tensor_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--parents', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo))
    from clmm.pretrain.kv_streams import training_data, evaluation_data, signature
    regenerated, specs, excluded = training_data(1200, 32, 90502)
    print('regenerated exact planned training inputs', flush=True)
    result = {'audit_scope': 'CPU parent input checks; no training or model updates',
              'training_seed': 90502, 'steps': 1200, 'batch': 32,
              'planned_training_unique_histories': len(excluded), 'parents': {}}
    canonical_keys = canonical_values = None
    observed_pairs = set()
    learned_names = ('key_encoder', 'value_encoder', 'decoder', 'W0', 'alpha', 'raw_eta')
    for rule in ('gradient', 'hebbian'):
        for seed in (90511, 90515):
            name = f'{rule}-dense-{seed}'
            folder = args.parents / name
            with np.load(folder/'training_data.npz') as archive:
                arrays = {key: archive[key] for key in archive.files}
                assert set(arrays) == set(regenerated)
                exact = {key: bool(np.array_equal(arrays[key], expected))
                         for key, expected in regenerated.items()}
                assert all(exact.values()), (name, exact)
                assert arrays['keys'].shape == (1200, 32, 14)
                seen_keys = set()
                seen_values = set()
                signatures = set()
                pairs = set()
                for step in range(1200):
                    length = int(arrays['lengths'][step])
                    for batch in range(32):
                        keys = arrays['keys'][step, batch, :length].tolist()
                        values = arrays['values'][step, batch, :length].tolist()
                        signatures.add(signature(keys, values))
                        pairs.update(zip(keys, values))
                        seen_keys.update(keys)
                        seen_values.update(values)
                assert signatures == excluded
                assert seen_keys == set(range(64)) and seen_values == set(range(16))
                assert pairs == set((k, v) for k in range(64) for v in range(16))
                observed_pairs.update(pairs)
            payload = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
            assert payload['format'] == 'clmm.adapter.reference.v1'
            assert payload['config']['routing'] == 'dense'
            assert payload['config']['update_rule'] == rule
            assert payload['training']['init_seed'] == seed
            state = payload['model_state']
            assert set(learned_names).issubset(state)
            assert all(torch.isfinite(t).all() for t in state.values())
            keys = state['codebook.keys']
            values = state['codebook.values']
            if canonical_keys is None:
                canonical_keys, canonical_values = keys.clone(), values.clone()
            assert torch.equal(keys, canonical_keys) and torch.equal(values, canonical_values)
            assert torch.equal(state['router_prototypes'], keys[[0, 16, 32, 48]])
            result['parents'][name] = {
                'checkpoint_sha256': hashlib.sha256((folder/'checkpoint.pt').read_bytes()).hexdigest(),
                'training_arrays_equal_regeneration': exact,
                'training_unique_histories': len(signatures),
                'familiar_key_count': len(seen_keys), 'familiar_value_count': len(seen_values),
                'familiar_key_value_binding_count': len(pairs),
                'learned_tensor_sha256': {n: tensor_hash(state[n]) for n in learned_names},
                'eta': float(state['raw_eta'].sigmoid()),
            }
            print(f'checked {name}', flush=True)
    similarities = F.normalize(canonical_keys, dim=-1) @ F.normalize(canonical_keys[[0, 16, 32, 48]], dim=-1).T
    table = similarities.argmax(-1).cpu().numpy()
    cardinalities = np.bincount(table, minlength=4)
    result['content_table'] = table.tolist()
    result['content_key_cardinalities'] = cardinalities.tolist()
    result['prospective_random_tables'] = {}
    for seed in range(90701, 90706):
        random_table = np.random.Generator(np.random.PCG64(seed)).permutation(table)
        assert np.array_equal(np.bincount(random_table, minlength=4), cardinalities)
        result['prospective_random_tables'][str(seed)] = random_table.tolist()
    data = evaluation_data(90723, 128, 128, excluded, 'full')
    data = {k: v for k, v in data.items() if not k.startswith('pressure_k32_')}
    assert len(data) == 9
    signatures = [ep['signature'] for episodes in data.values() for ep in episodes]
    assert not excluded.intersection(signatures)
    assert all(len(episodes) == 128 for episodes in data.values())
    for k in (8, 16):
        for episode in range(128):
            cells = [data[f'pressure_k{k}_d{d}'][episode] for d in (0, 16, 64)]
            assert len({ep['seed'] for ep in cells}) == 1
            assert len({ep['boundary'] for ep in cells}) == 1
            boundary = cells[0]['boundary']
            assert all(ep['keys'][:boundary] == cells[0]['keys'] for ep in cells)
            assert all(ep['values'][:boundary] == cells[0]['values'] for ep in cells)
            assert all(ep['queries'][:4] == cells[0]['queries'] for ep in cells)
    result['fresh_data'] = {'development_seed': 90723, 'cells': list(data),
                            'histories_per_cell': 128, 'training_signature_overlap': 0,
                            'delay_prefix_pairing': True}
    result['all_checks_passed'] = True
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('all_checks_passed', 'planned_training_unique_histories', 'content_key_cardinalities', 'fresh_data')}))


if __name__ == '__main__':
    main()
