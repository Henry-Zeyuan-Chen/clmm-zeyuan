"""CPU tensor equality on saved case payloads, without constructing adapters."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slow_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name == 'route_table':
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode()+b'\0'+str(tuple(value.shape)).encode()+b'\0'+value.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parents', type=Path, required=True, help='Original per-parent folders or flattened parent .pt files')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    parents = {}
    for rule in ('gradient', 'hebbian'):
        for seed in (90511, 90515):
            path = args.parents/f'{rule}-dense-{seed}'/'checkpoint.pt'
            if not path.exists():
                path = args.parents/f'{rule}-dense-{seed}.pt'
            parents[(rule, seed)] = (torch.load(path, map_location='cpu', weights_only=True), file_sha(path))
    expected = {f'{rule}-{seed}-{route}-{decay}' for rule in ('gradient', 'hebbian') for seed in (90511, 90515)
                for route in ('content', *(f'random-{seed}' for seed in range(90701, 90706))) for decay in ('protected', 'global')}
    results = {}
    fingerprints = {}
    for folder in sorted((args.root/'runs').iterdir()):
        if not folder.is_dir():
            continue
        assert folder.name in expected
        info = json.loads((folder/'case.json').read_text(encoding='utf-8-sig'))
        if info['status'] != 'succeeded':
            continue
        parent, parent_hash = parents[(info['rule'], int(info['parent_seed']))]
        saved = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
        assert saved['format'] == 'clmm.routing_decay.v1'
        assert saved['source_commit'] == info['source_commit'] == '61212289d6029da65eeb6a3735ce5ec3894866a9'
        assert saved['parent_sha256'] == info['parent_sha256'] == parent_hash
        assert file_sha(folder/'checkpoint.pt') == info['checkpoint_sha256']
        assert saved['config'] == info['config']
        state, original = saved['model_state'], parent['model_state']
        assert set(state) == set(original) | {'route_table'}
        assert all(torch.equal(state[key], value) for key, value in original.items())
        assert all(torch.isfinite(value).all() for value in state.values())
        assert all(state[key].dtype == value.dtype for key, value in original.items())
        assert slow_sha(state) == slow_sha(original) == saved['slow_sha256'] == info['slow_sha256']
        keys = original['codebook.keys']
        content = (F.normalize(keys, dim=-1) @ F.normalize(keys[[0, 16, 32, 48]], dim=-1).T).argmax(-1)
        expected_table = content.numpy() if info['policy'] == 'content' else np.random.Generator(np.random.PCG64(info['random_seed'])).permutation(content.numpy())
        table = state['route_table']
        assert table.dtype == torch.int64 and table.shape == (64,)
        assert np.array_equal(table.numpy(), expected_table)
        header = json.dumps({'shape': [64], 'dtype': 'torch.int64'}, sort_keys=True, separators=(',', ':')).encode()
        digest = hashlib.sha256(header+b'\0'+table.numpy().astype('<i8', copy=False).tobytes(order='C')).hexdigest()
        assert digest == saved['config']['routing_table_sha256']
        assert table.tolist() == info['route_table']
        fingerprint = hashlib.sha256(json.dumps(saved['config'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        prior = fingerprints.setdefault((info['rule'], int(info['parent_seed'])), set())
        assert fingerprint not in prior
        prior.add(fingerprint)
        results[folder.name] = dict(parent_sha256=parent_hash, slow_sha256=saved['slow_sha256'],
                                    routing_table_sha256=digest, exact_original_tensor_count=len(original),
                                    checkpoint_sha256=info['checkpoint_sha256'])
    output = {'all_available_checks_passed': True, 'complete48': set(results) == expected,
              'checked_cases': len(results), 'missing_cases': sorted(expected-set(results)),
              'scope': 'Saved tensor/policy binding checks on CPU; no training or adapter construction', 'results': results}
    args.out.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: output[k] for k in ('all_available_checks_passed', 'complete48', 'checked_cases')}))


if __name__ == '__main__':
    main()
