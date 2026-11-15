"""Raw-artifact audit and paired analysis; imports no evaluator implementation."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True

SEEDS = (90511, 90515)
MAPS = ('content', *(f'random-{seed}' for seed in range(90701, 90706)))
DECAYS = ('protected', 'global')
RULES = ('gradient', 'hebbian')


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def name(rule, seed, route, decay):
    return f'{rule}-{seed}-{route}-{decay}'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='Actual run root containing runs/ and episodes.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-commit', default='61212289d6029da65eeb6a3735ce5ec3894866a9')
    parser.add_argument('--saved-training', type=Path, help='Original saved training_data.npz from any audited dense parent')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.root/'metadata.json').read_text(encoding='utf-8-sig'))
    assert metadata['scope'] == 'full48 factorial' and int(metadata['expected_cases']) == 48
    assert metadata['source_commit'] == args.source_commit
    assert metadata['offline_training'] is False
    assert int(metadata['args']['dev_seed']) == 90723
    assert int(metadata['args']['eval_episodes']) == 128 and int(metadata['args']['pressure_episodes']) == 128
    assert int(metadata['args']['eval_batch_size']) == 16
    episodes = json.loads((args.root/'episodes.json').read_text(encoding='utf-8-sig'))
    expected_cells = ('acquire2', 'acquire4', 'revise2', *(f'pressure_k{k}_d{d}' for k in (8, 16) for d in (0, 16, 64)))
    assert set(episodes) == set(expected_cells)
    assert all(len(eps) == 128 for eps in episodes.values())
    for eps in episodes.values():
        for episode in eps:
            assert episode['signature'] == hashlib.sha256(bytes(episode['keys'])+b'|'+bytes(episode['values'])).hexdigest()
    exclusion_checks = {}
    with np.load(args.root/'exclusion_training_data.npz') as archive:
        exclusion = {key: archive[key] for key in archive.files}
        assert exclusion['keys'].shape == (1200, 32, 14)
        exclusions = set()
        for step in range(1200):
            length = int(exclusion['lengths'][step])
            for batch in range(32):
                keys = exclusion['keys'][step, batch, :length].tolist()
                values = exclusion['values'][step, batch, :length].tolist()
                exclusions.add(hashlib.sha256(bytes(keys)+b'|'+bytes(values)).hexdigest())
        assert len(exclusions) == int(metadata['exclusion_unique_histories'])
        assert not exclusions.intersection(ep['signature'] for eps in episodes.values() for ep in eps)
        if args.saved_training:
            with np.load(args.saved_training) as original:
                assert set(exclusion) == set(original.files)
                exclusion_checks = {key: bool(np.array_equal(exclusion[key], original[key])) for key in exclusion}
                assert all(exclusion_checks.values())
    stream_source = args.root/'sources/clmm/pretrain/kv_streams.py'
    assert sha(stream_source) == metadata['source_sha256']['clmm/pretrain/kv_streams.py']
    specification = importlib.util.spec_from_file_location('_frozen_streams_for_design_audit', stream_source)
    stream_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(stream_module)
    intended = stream_module.evaluation_data(90723, 128, 128, exclusions, 'full')
    intended = {key: value for key, value in intended.items() if not key.startswith('pressure_k32_')}
    assert intended == episodes, 'Actual main episodes differ from the declared deterministic fresh dataset'
    design_path = args.root/'design.json'
    if design_path.exists():
        design = json.loads(design_path.read_text(encoding='utf-8-sig'))
        assert design['primary']['cell'] == 'pressure_k8_d64'
    expected_names = {name(rule, seed, route, decay) for rule in RULES for seed in SEEDS for route in MAPS for decay in DECAYS}
    run_folders = {p.name: p for p in (args.root/'runs').iterdir() if p.is_dir()}
    assert set(run_folders) <= expected_names, sorted(set(run_folders)-expected_names)
    audit = {'source': 'predictions.csv/scores.npy/plans.json/resources.csv/case.json; no evaluator imports',
             'root': str(args.root.resolve()), 'metadata_status': metadata['status'], 'expected_cases': 48,
             'completed_cases': [], 'incomplete_cases': [], 'missing_cases': sorted(expected_names-set(run_folders)),
             'prediction_vectors': 0, 'causal_write_events': 0, 'resource_batches': 0,
             'serialized_batches': 0, 'serialization_boundaries': 0,
             'retained_active_association_applications': 0, 'retained_ema_decay_applications': 0,
             'source_hashes': {}, 'per_case': {}}
    audit['main_exclusion_vs_original_training_arrays'] = exclusion_checks
    audit['unique_exclusion_signatures'] = len(exclusions)
    audit['training_signature_overlap'] = 0
    audit['exact_actual_vs_intended_episode_equality'] = True
    vectors = {}
    all_correct = {}
    gates = []
    pressure_gates = []
    summaries = []
    exposures = []
    identifiers = {}
    parent_identities = defaultdict(set)
    common_tables = {}
    main_draws = np.random.Generator(np.random.PCG64(90704)).integers(0, 128, size=(5000, 128))
    block_draws = np.random.Generator(np.random.PCG64(90704)).integers(0, 8, size=(5000, 8))
    allshort = {}
    own_admission = {}
    for case in sorted(run_folders):
        folder = run_folders[case]
        info = json.loads((folder/'case.json').read_text(encoding='utf-8-sig'))
        assert info['name'] == case and info['offline_training'] is False
        required = ('predictions.csv', 'scores.npy', 'plans.json', 'resources.csv', 'episodes.json')
        if info['status'] != 'succeeded' or any(not (folder/'evaluation'/f).exists() for f in required):
            audit['incomplete_cases'].append(case)
            continue
        assert info['source_commit'] == metadata['source_commit']
        assert sha(folder/'checkpoint.pt') == info['checkpoint_sha256']
        route = 'content' if info['policy'] == 'content' else f"random-{info['random_seed']}"
        rule, seed, decay = info['rule'], int(info['parent_seed']), info['inactive_decay']
        assert case == name(rule, seed, route, decay)
        assert np.array_equal(info['route_cardinalities'], np.bincount(info['route_table'], minlength=4))
        table = np.asarray(info['route_table'], dtype=np.int64)
        assert table.shape == (64,) and table.min() >= 0 and table.max() < 4
        if route in common_tables:
            assert np.array_equal(table, common_tables[route])
        else:
            common_tables[route] = table
        parent_identities[(rule, seed)].add((info['parent_sha256'], info['slow_sha256']))
        evaluation = folder/'evaluation'
        assert json.loads((evaluation/'episodes.json').read_text(encoding='utf-8-sig')) == episodes
        with (evaluation/'predictions.csv').open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        scores = np.load(evaluation/'scores.npy')
        assert sha(evaluation/'scores.npy') == info['evaluation']['scores_sha256']
        assert info['evaluation']['pure_reads'] is True and info['evaluation']['slow_parameters_and_rng_unchanged'] is True
        assert scores.shape == (len(rows), 16) and scores.dtype == np.float32 and np.isfinite(scores).all()
        predictions = scores.argmax(-1)
        target = np.asarray([int(r['target']) for r in rows])
        assert np.array_equal(predictions, [int(r['prediction']) for r in rows])
        assert np.array_equal(predictions == target, [bool(int(r['correct'])) for r in rows])
        mse = ((scores-np.eye(16)[target])**2).mean(1)
        assert np.allclose(mse, [float(r['mse']) for r in rows], atol=1e-10, rtol=1e-8)
        groups = defaultdict(list)
        original_groups = defaultdict(list)
        query_groups = defaultdict(list)
        grouped_rows = defaultdict(list)
        row_keys = set()
        for r in rows:
            cell, arm, epidx, rank = r['cell'], r['arm'], int(r['episode']), int(r['rank'])
            key = (cell, arm, epidx, rank)
            assert key not in row_keys
            row_keys.add(key)
            assert arm in (('online',) if cell.startswith('pressure') else ('online', 'reset', 'rolled'))
            ep = episodes[cell][epidx]
            query = ep['queries'][rank]
            assert all(int(r[k]) == int(query[k]) for k in ('key', 'target', 'old'))
            assert r['category'] == query['category']
            assert int(r['obsolete']) == int(query['old'] >= 0 and int(r['prediction']) == query['old'])
            assert int(r['actual_writes']) == len(ep['keys'])
            assert int(r['state_write_count']) == (0 if arm == 'reset' else len(ep['keys']))
            batch_start = (epidx//16)*16
            expected_source = -1 if arm == 'reset' else batch_start+(epidx-batch_start-1)%16 if arm == 'rolled' else epidx
            assert int(r['state_source_episode']) == expected_source
            head = int(table[query['key']])
            last = max(i for i, k in enumerate(ep['keys']) if k == query['key'])
            following = ep['keys'][last+1:]
            same = sum(int(table[k]) == head for k in following)
            assert int(r['query_module']) == head
            assert int(r['after_last_real_target_observation']) == len(following)
            assert int(r['following_same_module_writes']) == same
            assert int(r['following_other_module_writes']) == len(following)-same
            assert int(r['target_decay_applications']) == (len(following) if decay == 'global' else same)
            groups[(cell, arm, epidx)].append(int(r['correct']))
            if r['category'] != 'recent':
                original_groups[(cell, arm, epidx)].append(int(r['correct']))
            query_groups[(cell, arm, r['category'], epidx)].append(int(r['correct']))
            grouped_rows[(cell, arm, r['category'])].append(r)
            if r['category'] != 'recent':
                grouped_rows[(cell, arm, 'targets')].append(r)
        assert len(rows) == 6656, len(rows)
        expected_row_keys = {(cell, arm, e, q) for cell, eps in episodes.items()
                             for arm in (('online',) if cell.startswith('pressure') else ('online', 'reset', 'rolled'))
                             for e, ep in enumerate(eps) for q in range(len(ep['queries']))}
        assert row_keys == expected_row_keys
        for cell in episodes:
            arms = ('online',) if cell.startswith('pressure') else ('online', 'reset', 'rolled')
            for arm in arms:
                cats = sorted({q['category'] for q in episodes[cell][0]['queries']})
                for category in cats:
                    value = np.asarray([np.mean(query_groups[(cell, arm, category, i)]) for i in range(128)])
                    vectors[(case, cell, arm, category)] = value
                    ids = (cell, arm, category)
                    history_ids = tuple(ep['seed'] for ep in episodes[cell])
                    assert identifiers.setdefault(ids, history_ids) == history_ids
                    raw = grouped_rows[(cell, arm, category)]
                    summaries.append(dict(case=case, rule=rule, parent_seed=seed, route=route, inactive_decay=decay,
                                          cell=cell, arm=arm, category=category, histories=128, queries=len(raw),
                                          accuracy=float(value.mean()), mse=float(np.mean([float(r['mse']) for r in raw])),
                                          obsolete_rate=float(np.mean([int(r['obsolete']) for r in raw]))))
                    if arm == 'online':
                        exposures.append(dict(case=case, rule=rule, parent_seed=seed, route=route, inactive_decay=decay,
                                              cell=cell, category=category,
                                              following_same_module_writes=float(np.mean([int(r['following_same_module_writes']) for r in raw])),
                                              following_other_module_writes=float(np.mean([int(r['following_other_module_writes']) for r in raw])),
                                              target_decay_applications=float(np.mean([int(r['target_decay_applications']) for r in raw]))))
                # Original four-target admission excludes the recent distractor.
                all_correct[(case, cell, arm)] = np.asarray([all(original_groups[(cell, arm, i)])
                    for i in range(128)], dtype=np.float64)
        with (evaluation/'summary.csv').open(newline='', encoding='utf-8-sig') as stream:
            saved_summary = list(csv.DictReader(stream))
        summary_ids = set()
        for saved in saved_summary:
            ident = (saved['cell'], saved['arm'], saved['category'])
            assert ident not in summary_ids
            summary_ids.add(ident)
            subset = grouped_rows[ident]
            assert subset and int(saved['queries']) == len(subset) and int(saved['episodes']) == 128
            by_history = defaultdict(list)
            for row in subset:
                by_history[int(row['episode'])].append(int(row['correct']))
            expected_summary = dict(accuracy=np.mean([int(r['correct']) for r in subset]),
                                    all_correct=np.mean([all(v) for v in by_history.values()]),
                                    mse=np.mean([float(r['mse']) for r in subset]))
            for key, value in expected_summary.items():
                assert np.isclose(float(saved[key]), value, atol=1e-12, rtol=1e-12), (case, ident, key)
            if saved['category'] == 'revised':
                assert np.isclose(float(saved['obsolete_rate']), np.mean([int(r['obsolete']) for r in subset]), atol=1e-12, rtol=1e-12)
            else:
                assert saved['obsolete_rate'] == ''
        assert summary_ids == set(grouped_rows)
        plans = json.loads((evaluation/'plans.json').read_text(encoding='utf-8-sig'))
        plan_ids = set()
        writes = 0
        for plan in plans:
            cell, arm, epidx = plan['cell'], plan['arm'], int(plan['episode'])
            ident = (cell, arm, epidx)
            assert ident not in plan_ids
            plan_ids.add(ident)
            ep = episodes[cell][epidx]
            expected = [dict(source=i, key=k, value=v, kind='online', after_online=i)
                        for i, (k, v) in enumerate(zip(ep['keys'], ep['values']))]
            assert plan['events'] == expected
            assert int(plan['boundary']) == ep['boundary']
            writes += len(expected)
        assert plan_ids == set(groups)
        with (evaluation/'resources.csv').open(newline='', encoding='utf-8-sig') as stream:
            resources = list(csv.DictReader(stream))
        resource_ids = set()
        for r in resources:
            cell, arm, start, count = r['cell'], r['arm'], int(r['batch_start']), int(r['batch'])
            assert count == 16 and start % 16 == 0
            ident = (cell, arm, start)
            assert ident not in resource_ids
            resource_ids.add(ident)
            eps = episodes[cell][start:start+count]
            occupancies = np.asarray([np.bincount(table[ep['keys']], minlength=4) for ep in eps])
            decays = occupancies if decay == 'protected' else np.asarray([[len(ep['keys'])]*4 for ep in eps])
            assert np.array_equal(occupancies, json.loads(r['per_history_module_occupancy']))
            assert np.array_equal(decays, json.loads(r['per_history_decay_applications']))
            assert np.array_equal(occupancies.sum(0), json.loads(r['module_occupancy']))
            assert int(r['actual_writes']) == len(eps[0]['keys'])
            assert int(r['active_module_updates']) == int(occupancies.sum())
            assert int(r['decay_applications']) == int(decays.sum())
            assert int(r['replay_writes']) == 0
            assert float(r['per_stream_final_bytes']) == 65536
            assert int(r['final_fast_bytes']) == count*65536 and int(r['peak_fast_bytes']) == count*65536
            assert float(r['boundary_error']) == 0 and float(r['midblock_error']) == 0
            assert r['resume_points'] == 'prefill,midblock' or (not eps[0]['delay'] and r['resume_points'] == 'prefill')
            assert int(r['query_state_write_count']) == (0 if arm == 'reset' else len(eps[0]['keys']))
            audit['serialization_boundaries'] += len(r['resume_points'].split(','))
            audit['retained_active_association_applications'] += int(r['active_module_updates'])
            audit['retained_ema_decay_applications'] += int(r['decay_applications'])
        assert len(resources) == 120 and len(resource_ids) == 120
        assert len(plans) == 1920
        short = {cell: float(all_correct[(case, cell, 'online')].mean()) for cell in ('acquire2', 'acquire4', 'revise2')}
        rolled_diff = all_correct[(case, 'acquire2', 'online')]-all_correct[(case, 'acquire2', 'rolled')]
        blocked = rolled_diff.reshape(8, 16).mean(1)[block_draws].mean(1)
        low, high = np.quantile(blocked, [.025, .975])
        shortpass = short['acquire2'] >= .9 and short['acquire4'] >= .8 and short['revise2'] >= .9 and low > 0
        allshort[case] = shortpass
        gates.append(dict(case=case, rule=rule, parent_seed=seed, route=route, inactive_decay=decay, **short,
                          acquire2_normal_minus_rolled=float(rolled_diff.mean()), rolled_lower95=float(low), rolled_upper95=float(high),
                          own_short_pass=bool(shortpass), reset_acquire2=float(all_correct[(case, 'acquire2', 'reset')].mean())))
        for k in (8, 16):
            value = float(all_correct[(case, f'pressure_k{k}_d0', 'online')].mean())
            own_admission[(case, k)] = bool(shortpass and value >= .8)
            pressure_gates.append(dict(case=case, rule=rule, parent_seed=seed, route=route, inactive_decay=decay,
                                       K=k, d0_four_target_all_correct=value, own_d0_pass=bool(value >= .8),
                                       own_short_pass=bool(shortpass), own_admission=own_admission[(case, k)]))
        audit['completed_cases'].append(case)
        audit['prediction_vectors'] += len(rows)
        audit['causal_write_events'] += writes
        audit['resource_batches'] += len(resources)
        audit['serialized_batches'] += len(resources)
        audit['per_case'][case] = {'predictions': len(rows), 'writes': writes, 'plans': len(plans), 'resources': len(resources),
                                   'predictions_sha256': sha(evaluation/'predictions.csv'), 'scores_sha256': sha(evaluation/'scores.npy'),
                                   'checkpoint_sha256': sha(folder/'checkpoint.pt'), 'summary_rows_verified': len(saved_summary)}
        print(f'audited {case}', flush=True)
    assert all(len(values) == 1 for values in parent_identities.values())
    if 'content' in common_tables:
        content = common_tables['content']
        for route, table in common_tables.items():
            assert np.array_equal(np.bincount(table, minlength=4), np.bincount(content, minlength=4))
            if route != 'content':
                expected = np.random.Generator(np.random.PCG64(int(route.split('-')[1]))).permutation(content)
                assert np.array_equal(table, expected)
    for path, expected in metadata.get('source_sha256', {}).items():
        saved = args.root/'sources'/path
        assert sha(saved) == expected, path
        audit['source_hashes'][path] = expected
    write_csv(args.out/'short_gates.csv', gates)
    write_csv(args.out/'pressure_parent_gates.csv', pressure_gates)
    write_csv(args.out/'per_parent_results.csv', summaries)
    write_csv(args.out/'exposure_descriptors.csv', exposures)
    complete = set(audit['completed_cases']) == expected_names
    audit['full_factorial_complete'] = complete
    family_gates = []
    if complete:
        for rule in RULES:
            for route in MAPS:
                for decay in DECAYS:
                    for k in (8, 16):
                        family_gates.append(dict(rule=rule, route=route, inactive_decay=decay, K=k,
                                                 both_parent_own_admission=all(own_admission[(name(rule, s, route, decay), k)] for s in SEEDS)))
        write_csv(args.out/'family_admission.csv', family_gates)
        def array(rule, route, decay, cell='pressure_k8_d64', category='stable', parent=None):
            parents = SEEDS if parent is None else (parent,)
            return np.mean([vectors[(name(rule, seed, route, decay), cell, 'online', category)] for seed in parents], axis=0)
        def random_mean(rule, decay, cell='pressure_k8_d64', category='stable', parent=None):
            return np.mean([array(rule, route, decay, cell, category, parent) for route in MAPS[1:]], axis=0)
        def qualify(rule, routes, decays, k=8):
            return all(own_admission[(name(rule, s, route, decay), k)] for s in SEEDS for route in routes for decay in decays)
        contrasts = []
        def contrast(label, rule, delta, primary=False, admitted=None):
            level = .975 if primary else .95
            boot = delta[main_draws].mean(1)
            lo, hi = np.quantile(boot, [(1-level)/2, 1-(1-level)/2])
            contrasts.append(dict(contrast=label, rule=rule, co_primary=primary, cell='pressure_k8_d64', category='stable',
                                  estimate=float(delta.mean()), interval_level=level, lower=float(lo), upper=float(hi),
                                  histories=128, fixed_parents=2, fixed_random_maps=5, bootstrap_draws=5000, bootstrap_seed=90704,
                                  admitted=admitted, interpretation='admitted retention contrast' if admitted else 'acquisition/interference diagnostic'))
        for rule in RULES:
            cp, cg = array(rule, 'content', 'protected'), array(rule, 'content', 'global')
            rp, rg = random_mean(rule, 'protected'), random_mean(rule, 'global')
            contrast('content_protected_minus_mean5random_protected', rule, cp-rp, rule == 'hebbian', qualify(rule, MAPS, ('protected',)))
            contrast('content_protected_minus_content_global', rule, cp-cg, rule == 'hebbian', qualify(rule, ('content',), DECAYS))
            contrast('mean5random_protected_minus_mean5random_global', rule, rp-rg, False, qualify(rule, MAPS[1:], DECAYS))
            contrast('routing_by_decay_interaction', rule, (cp-rp)-(cg-rg), False, qualify(rule, MAPS, DECAYS))
            contrast('content_global_minus_mean5random_global', rule, cg-rg, False, qualify(rule, MAPS, ('global',)))
        contrast('hebbian_minus_gradient_routing_effect', 'interaction',
                 (array('hebbian', 'content', 'protected')-random_mean('hebbian', 'protected'))-
                 (array('gradient', 'content', 'protected')-random_mean('gradient', 'protected')),
                 False, all(qualify(r, MAPS, ('protected',)) for r in RULES))
        write_csv(args.out/'contrasts.csv', contrasts)
        permap = []
        for rule in RULES:
            for seed in SEEDS:
                for route in MAPS:
                    for decay in DECAYS:
                        value = array(rule, route, decay, parent=seed)
                        permap.append(dict(rule=rule, parent_seed=seed, route=route, inactive_decay=decay,
                                           accuracy=float(value.mean()), own_admission=own_admission[(name(rule, seed, route, decay), 8)]))
        write_csv(args.out/'k8_d64_stable_per_map_parent.csv', permap)
        aggregate = []
        for rule in RULES:
            for cell in expected_cells:
                if not cell.startswith('pressure'):
                    continue
                for category in ('stable', 'revised'):
                    for decay in DECAYS:
                        for route in ('content', 'mean5random'):
                            value = array(rule, route, decay, cell, category) if route == 'content' else random_mean(rule, decay, cell, category)
                            boot = value[main_draws].mean(1)
                            lo, hi = np.quantile(boot, [.025, .975])
                            aggregate.append(dict(rule=rule, cell=cell, category=category, route=route, inactive_decay=decay,
                                                  accuracy=float(value.mean()), lower95=float(lo), upper95=float(hi)))
        write_csv(args.out/'aggregate_pressure.csv', aggregate)
        audit['co_primary'] = [r for r in contrasts if r['co_primary']]
    audit['all_available_artifact_checks_passed'] = True
    assert audit['retained_active_association_applications'] == audit['causal_write_events']
    audit['fully_audited_complete_main'] = bool(complete and metadata['status'] == 'succeeded')
    (args.out/'raw_artifact_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ('all_available_artifact_checks_passed', 'fully_audited_complete_main', 'prediction_vectors', 'causal_write_events', 'resource_batches')}))


if __name__ == '__main__':
    main()
