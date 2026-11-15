"""Authenticate data/model/source packet using stdlib and NumPy only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from independent_tools import PARENTS, ROUTES, audit_dataset, saved_signatures, sha
from independent_generation import audit_generation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--seed', type=int, default=91223)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'packet_audit.json')
    args = parser.parse_args()
    packet = args.packet.resolve()
    meta = json.loads((packet/'metadata.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'succeeded' and meta['mode'] == 'prepare'
    assert meta['offline_training'] is False and meta['model_count'] == 12
    assert meta['source_commit'] == args.source_commit
    assert meta['development_seed'] == args.seed and meta['episodes'] == args.episodes
    assert sha(packet/'episodes.json') == meta['data_sha256']
    assert sha(packet/'excluded_signatures.json') == meta['exclusions_sha256']
    excluded_list = json.loads((packet/'excluded_signatures.json').read_text(encoding='utf-8'))
    assert excluded_list == sorted(set(excluded_list))
    excluded = set(excluded_list)
    manifest = json.loads((packet/'exclusion_manifest.json').read_text(encoding='utf-8'))
    assert manifest['union_histories'] == len(excluded)
    source_files = {}
    for name, expected in meta['source_sha256'].items():
        preserved = packet/'sources'/name
        assert sha(preserved) == expected
        blob = subprocess.check_output(['git', 'show', f"{meta['source_commit']}:{name}"], cwd=args.repo)
        assert hashlib.sha256(blob).hexdigest() == expected, name
        source_files[name] = expected
    required = ['scripts/sleep_multirevision_experiment.py', 'scripts/freshness_experiment.py',
                'scripts/adapter_integration_experiment.py', 'scripts/revision_local_experiment.py',
                'clmm/sleep/multirevision.py', 'clmm/sleep/adapter.py',
                'clmm/pretrain/kv_scheduler.py', 'clmm/pretrain/kv_streams.py', 'clmm/hebbian/revision.py']
    assert all(name in source_files for name in required)
    exclusions = {}
    for seed, relative in ((90502, 'trained/old_training_exclusions.npz'), (90802, 'trained/training_data.npz')):
        original = args.prior_root/relative
        ids = saved_signatures(original)
        assert ids and ids <= excluded
        preserved = packet/f'training_exclusions_{seed}.npz'
        with np.load(preserved, allow_pickle=False) as actual, np.load(original, allow_pickle=False) as old:
            assert set(actual.files) == set(old.files)
            assert all(np.array_equal(actual[name], old[name]) for name in actual.files)
        exclusions[f'training{seed}'] = dict(source=str(original.resolve()), sha256=sha(original), histories=len(ids))
    reconstructed, inputs = set(), []
    for item in manifest['inputs']:
        if item['kind'] == 'training':
            path = packet/f"training_exclusions_{item['seed']}.npz"
            ids = saved_signatures(path)
        elif item['kind'] == 'prior_evaluation':
            path = args.prior_root.parent/item['path']
            ids = saved_signatures(path)
        elif item['kind'] == 'supplied_exclusions':
            command = meta['command']
            path = Path(command[command.index('--exclusions')+1])
            ids = set(json.loads(path.read_text(encoding='utf-8')))
        else:
            raise AssertionError(item)
        assert sha(path) == item['sha256']
        if 'histories' in item:
            assert len(ids) == item['histories']
        reconstructed.update(ids)
        inputs.append(dict(path=str(path.resolve()), histories=len(ids), sha256=item['sha256']))
    assert reconstructed == excluded
    # Independently enumerate every prior numeric KV episodes.json, rather than
    # trusting the producer's manifest to enumerate its own omissions.
    prior_episode_files = []
    output_root, current_root = args.prior_root.parent.resolve(), packet.parent.resolve()
    for root in sorted(output_root.glob('clmm-*')):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('episodes.json')):
            resolved = path.resolve()
            if current_root in resolved.parents:
                # Current engineering is an exclusion input; prepared/main data
                # and their preserved copies are the outcomes under audit.
                if not any('engineer' in part.lower() for part in resolved.relative_to(current_root).parts):
                    continue
            ids = saved_signatures(path)
            assert ids <= excluded, (str(path), len(ids-excluded))
            prior_episode_files.append(dict(path=str(resolved), sha256=sha(path), histories=len(ids)))
    datasets = json.loads((packet/'episodes.json').read_text(encoding='utf-8'))
    geometry = audit_dataset(datasets, args.episodes, excluded)
    generation = audit_generation(datasets, args.seed, args.episodes, excluded)
    models = json.loads((packet/'models.json').read_text(encoding='utf-8'))
    assert len(models) == len({r['name'] for r in models}) == 12
    assert {r['name'] for r in models} == {f'{p}-{r}-local_delta' for p in PARENTS for r in ROUTES}
    inventory = []
    for record in models:
        folder = args.prior_root/'trained/runs'/record['name']
        old = json.loads((folder/'case.json').read_text(encoding='utf-8'))
        assert record['original_training'] == old['training'] and old['training']['step'] == 1200
        assert all(record[key] == old[key] for key in ('parent_seed', 'policy', 'random_seed', 'config', 'slow_sha256', 'route_table', 'checkpoint_sha256'))
        assert sha(packet/record['checkpoint']) == sha(folder/'checkpoint.pt') == record['checkpoint_sha256']
        inventory.append(dict(name=record['name'], checkpoint_sha256=record['checkpoint_sha256']))
    result = dict(status='passed', scope='full prepared inputs/source/file arithmetic; no model inference',
                  packet=str(packet), metadata_sha256=sha(packet/'metadata.json'), source_commit=meta['source_commit'],
                  episodes_sha256=meta['data_sha256'], exclusions_sha256=meta['exclusions_sha256'],
                  design_sha256=sha(packet/'design.json') if (packet/'design.json').exists() else None,
                  geometry=geometry, generation=generation, training_exclusions=exclusions,
                  reconstructed_manifest_inputs=inputs, independently_enumerated_prior_episode_files=prior_episode_files,
                  models=inventory, preserved_source_files=source_files, source_files_match_git_blobs=len(source_files))
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='passed', models=12, histories=args.episodes, excluded_histories=len(excluded), prior_files=len(prior_episode_files), out=str(args.out))))


if __name__ == '__main__':
    main()
