"""Authenticate a prepared input packet independently, without importing CLMM."""
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
    parser.add_argument('--seed', type=int, default=91123)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'packet_audit.json')
    args = parser.parse_args()
    packet = args.packet.resolve()
    meta = json.loads((packet/'metadata.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'succeeded' and meta['mode'] == 'prepare'
    assert meta['offline_training'] is False and meta['model_count'] == 12
    assert meta['source_commit'] == '06c0a35b543c25437659765128b81718e5bfa1c5'
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
    required = ['scripts/sleep_multirevision_experiment.py', 'scripts/adapter_integration_experiment.py',
                'scripts/revision_local_experiment.py', 'clmm/sleep/multirevision.py',
                'clmm/sleep/adapter.py', 'clmm/pretrain/kv_scheduler.py', 'clmm/pretrain/kv_streams.py',
                'clmm/hebbian/revision.py']
    assert all(name in source_files for name in required)
    roles = {'training90502': 'trained/old_training_exclusions.npz',
             'training90802': 'trained/training_data.npz',
             'evaluation90723': 'provenance/prior90723/episodes.json',
             'evaluation90823': 'frozen/episodes.json',
             'evaluation90923': 'trained/episodes.json',
             'evaluation91023': 'stress/episodes.json'}
    exclusions = {}
    for role, relative in roles.items():
        original = args.prior_root/relative
        signatures = saved_signatures(original)
        assert signatures and signatures <= excluded, role
        exclusions[role] = dict(source=str(original.resolve()), sha256=sha(original),
                                histories=len(signatures), all_present=True)
    for seed, relative in ((90502, 'trained/old_training_exclusions.npz'), (90802, 'trained/training_data.npz')):
        preserved = packet/f'training_exclusions_{seed}.npz'
        with np.load(preserved, allow_pickle=False) as actual, np.load(args.prior_root/relative, allow_pickle=False) as original:
            assert set(actual.files) == set(original.files)
            assert all(np.array_equal(actual[name], original[name]) for name in actual.files)
    reconstructed_exclusions = set()
    manifest_inputs = []
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
            raise AssertionError(f'Unexpected exclusion source {item}')
        assert sha(path) == item['sha256']
        if 'histories' in item:
            assert len(ids) == item['histories']
        reconstructed_exclusions.update(ids)
        manifest_inputs.append(dict(path=str(path.resolve()), histories=len(ids), sha256=item['sha256']))
    assert reconstructed_exclusions == excluded
    datasets = json.loads((packet/'episodes.json').read_text(encoding='utf-8'))
    geometry = audit_dataset(datasets, args.episodes, excluded)
    generation = audit_generation(datasets, args.seed, args.episodes, excluded)
    models = json.loads((packet/'models.json').read_text(encoding='utf-8'))
    assert len(models) == len({r['name'] for r in models}) == 12
    expected_names = {f'{p}-{r}-local_delta' for p in PARENTS for r in ROUTES}
    assert {r['name'] for r in models} == expected_names
    model_inventory = []
    for record in models:
        name = record['name']
        case_folder = args.prior_root/'trained/runs'/name
        old = json.loads((case_folder/'case.json').read_text(encoding='utf-8'))
        assert record['original_training'] == old['training'] and old['training']['step'] == 1200
        assert all(record[key] == old[key] for key in ('parent_seed', 'policy', 'random_seed', 'config', 'slow_sha256', 'route_table', 'checkpoint_sha256'))
        path = packet/record['checkpoint']
        assert sha(path) == sha(case_folder/'checkpoint.pt') == record['checkpoint_sha256']
        model_inventory.append(dict(name=name, checkpoint_sha256=record['checkpoint_sha256']))
    if (packet/'design.json').exists():
        design = json.loads((packet/'design.json').read_text(encoding='utf-8'))
        assert design['data']['seed'] == 91123 and design['data']['histories'] == 128
        design_sha = sha(packet/'design.json')
    else:
        design_sha = None
    result = dict(status='passed', scope='full prepared inputs/source/file arithmetic; no model inference',
                  packet=str(packet), metadata_sha256=sha(packet/'metadata.json'),
                  episodes_sha256=meta['data_sha256'], exclusions_sha256=meta['exclusions_sha256'],
                  design_sha256=design_sha, geometry=geometry, generation=generation,
                  expressly_required_exclusion_roles=exclusions,
                  reconstructed_all_manifest_inputs=manifest_inputs,
                  extra_exclusions_allowed=True, models=model_inventory, preserved_source_files=source_files,
                  source_files_match_git_blobs=len(source_files))
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='passed', models=12, histories=args.episodes, cells=10,
                         excluded_histories=len(excluded), source_files=len(source_files), out=str(args.out))))


if __name__ == '__main__':
    main()
