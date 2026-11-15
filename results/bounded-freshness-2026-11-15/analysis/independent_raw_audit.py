"""Independently reconstruct saved scores, schedules and resource arithmetic.

No model, evaluator, scheduler or dataset-generator imports are used. This is a
saved-output audit, not numerical inference reproduction.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from independent_tools import (ARMS, independent_plan, query_diagnostics, read_csv, sha)


def audit_evaluation(folder, datasets, route_table, batch_size=16):
    folder = Path(folder)
    table = np.asarray(route_table, dtype=np.int64)
    assert table.shape == (64,) and np.all((0 <= table) & (table < 4))
    if (folder/'episodes.json').exists():
        assert json.loads((folder/'episodes.json').read_text(encoding='utf-8')) == datasets
    rows = read_csv(folder/'predictions.csv')
    score = np.load(folder/'scores.npy', allow_pickle=False)
    assert score.shape == (len(rows), 16) and score.dtype == np.float32
    assert np.isfinite(score).all()
    predictions = score.argmax(axis=1)
    targets = np.asarray([int(row['target']) for row in rows])
    assert np.array_equal(predictions, [int(row['prediction']) for row in rows])
    assert np.array_equal(predictions == targets, [int(row['correct']) for row in rows])
    mse = np.mean((score-np.eye(16)[targets])**2, axis=1)
    assert np.allclose(mse, [float(row['mse']) for row in rows], atol=1e-10, rtol=1e-8)

    def arms(cell):
        return ARMS if cell.startswith('pressure_') else ('online', 'reset', 'rolled')

    plan_ids, plan_features, plan_occupancy, literal_pairs, plan_summary, candidate_pairs = set(), {}, {}, {}, {}, {}
    total_writes, stale_extras, extra_writes = 0, 0, 0
    plans = json.loads((folder/'plans.json').read_text(encoding='utf-8'))
    for saved in plans:
        cell, arm, i = saved['cell'], saved['arm'], int(saved['episode'])
        ident = (cell, arm, i)
        assert ident not in plan_ids
        plan_ids.add(ident)
        ep = datasets[cell][i]
        policy_arm = arm if cell.startswith('pressure_') else 'online'
        expected = independent_plan(ep, policy_arm)
        assert saved['boundary'] == ep['boundary']
        events = saved['events']
        basic = [{key: event[key] for key in ('source', 'key', 'value', 'kind', 'after_online')}
                 for event in events]
        expected_basic = [{key: event[key] for key in ('source', 'key', 'value', 'kind', 'after_online')}
                          for event in expected['events']]
        assert basic == expected_basic, ident
        for event, derived in zip(events, expected['events']):
            for key in ('sampled_source', 'sampled_value', 'refreshed', 'fallback'):
                if key in derived:
                    assert type(event[key]) is type(derived[key]) and event[key] == derived[key], (ident, key)
                else:
                    assert key not in event, (ident, key)
        world = {}
        for event, basic_event in zip(events, basic):
            key, value = event['key'], event['value']
            if event['kind'] == 'online':
                world[key] = value
            stale = event['kind'] == 'extra' and value != world[key]
            if 'stale' in event:
                assert bool(event['stale']) == stale
            if 'latest_online' in event:
                assert event['latest_online'] == world[key]
            if 'obsolete_anyquery' in event:
                query = next((q for q in ep['queries'] if q['key'] == key), None)
                versions = query.get('versions', [query['target']]) if query else []
                obsolete = bool(event['kind'] == 'extra' and query and value in versions[:-1])
                assert bool(event['obsolete_anyquery']) == obsolete
        total_writes += len(basic)
        stale_extras += expected['stale_extra_writes']
        extra_writes += expected['extra_writes']
        plan_features[ident] = {rank: query_diagnostics(ep, expected, query, table)
                               for rank, query in enumerate(ep['queries'])}
        plan_summary[ident] = dict(events=len(basic), extra_writes=expected['extra_writes'],
                                   stale_extra_writes=expected['stale_extra_writes'],
                                   final_obsolete_query_extras=sum(feature['query_final_obsolete_extra_writes']
                                                                    for feature in plan_features[ident].values()),
                                   refreshed_records=sum(bool(e.get('refreshed',False)) for e in expected['events']),
                                   changed_values=sum(e['value'] != e.get('sampled_value', e['value']) for e in expected['events']),
                                   fallback_writes=sum(bool(e.get('fallback',False)) for e in expected['events']))
        plan_occupancy[ident] = np.bincount(table[[event['key'] for event in basic]], minlength=4)
        if arm in ('continuous', 'periodic'):
            literal_pairs[ident] = [(event['source'], event['key'], event['value'])
                                   for event in basic if event['kind'] == 'extra']
        if arm.removesuffix('_bounded') in ('continuous', 'periodic'):
            candidate_pairs[ident] = [(event.get('sampled_source', event['source']), event['key'],
                                       event.get('sampled_value', event['value']))
                                      for event in events if event['kind'] == 'extra']
    del plans
    expected_plans = {(cell, arm, i) for cell, episodes in datasets.items()
                      for arm in arms(cell) for i in range(len(episodes))}
    assert plan_ids == expected_plans
    for cell, arm, i in literal_pairs:
        assert literal_pairs[(cell, 'continuous', i)] == literal_pairs[(cell, 'periodic', i)]
    for cell, arm, i in candidate_pairs:
        assert candidate_pairs[(cell, arm, i)] == candidate_pairs[(cell, 'continuous', i)]

    ids, groups, per_history, originals, diagnostics = set(), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        cell, arm, i, rank = row['cell'], row['arm'], int(row['episode']), int(row['rank'])
        ident = (cell, arm, i, rank)
        assert ident not in ids
        ids.add(ident)
        ep = datasets[cell][i]
        query = ep['queries'][rank]
        assert all(int(row[key]) == query[key] for key in ('key', 'target', 'old'))
        assert row['category'] == query['category']
        if 'group' in row:
            assert row['group'] == query.get('group', query['category'])
        if 'query_group' in row:
            assert row['query_group'] == query.get('group', query['category'])
        prediction = int(row['prediction'])
        feature = plan_features[(cell, arm, i)][rank]
        if 'version_history' in row:
            assert json.loads(row['version_history']) == feature['old_values']+[query['target']]
        if 'initial_value' in row:
            assert int(row['initial_value']) == (feature['old_values']+[query['target']])[0]
        aliases = dict(query_extra_writes='target_extra_writes',
                       query_stale_extra_writes='stale_target_extra_writes',
                       query_final_obsolete_extra_writes='query_final_obsolete_extra_writes',
                       last_target_valued_event='last_target_valued_event',
                       last_direct_write_source='last_direct_write_source')
        for saved_name, derived_name in aliases.items():
            if saved_name in row:
                assert int(row[saved_name]) == feature[derived_name], (ident, saved_name)
        if 'last_direct_write_kind' in row:
            assert row['last_direct_write_kind'] == feature['last_direct_write_kind']
        if 'stale_extra_writes' in row:
            assert int(row['stale_extra_writes']) == plan_summary[(cell, arm, i)]['stale_extra_writes']
        if 'fifo_eligible_blocks' in row:
            assert (int(row['fifo_eligible_blocks']) == feature['key_available_blocks']
                    if arm.removesuffix('_bounded') in ('continuous', 'periodic') else row['fifo_eligible_blocks'] == '')
        if 'final_fifo_has_key' in row:
            assert (int(row['final_fifo_has_key']) == feature['final_fifo_has_key']
                    if arm.removesuffix('_bounded') in ('continuous', 'periodic') else row['final_fifo_has_key'] == '')
        obsolete_any = int(prediction in feature['old_values'])
        penultimate = int(feature['penultimate'] >= 0 and prediction == feature['penultimate'])
        for name in ('obsolete_anyversion', 'obsolete_any', 'obsolete_any_version'):
            if name in row:
                assert int(row[name]) == obsolete_any
        for name in ('obsolete_penultimate', 'obsolete'):
            if name in row:
                assert int(row[name]) == penultimate
        write_count = len(ep['keys']) + (0 if arm in ('online', 'reset', 'rolled') else ep['delay'])
        assert int(row['actual_writes']) == write_count
        assert int(row['state_write_count']) == (0 if arm == 'reset' else write_count)
        start = (i//batch_size)*batch_size
        size = min(batch_size, len(datasets[cell])-start)
        source_episode = -1 if arm == 'reset' else start+(i-start-1)%size if arm == 'rolled' else i
        assert int(row['state_source_episode']) == source_episode
        # Existing fields describe exposure after the last REAL observation;
        # final-target-valued-write diagnostics below are independently derived.
        same = feature['after_real_same_module_writes']
        old_exposure = dict(query_module=int(table[query['key']]),
                            after_last_real_target_observation=feature['after_last_real_target_observation'],
                            following_same_module_writes=same,
                            following_other_module_writes=feature['after_real_other_module_writes'],
                            target_decay_applications=0, target_projection_events=same)
        for name, value in old_exposure.items():
            if name in row:
                assert int(row[name]) == value, (ident, name, row[name], value)
        for name in ('after_final_target_valued_write', 'subsequent_obsolete_same_key_writes',
                     'final_applied_value_for_key', 'target_extra_writes', 'stale_target_extra_writes',
                     'final_target_extra_writes', 'key_available_blocks', 'final_target_available_blocks'):
            if name in row:
                assert int(row[name]) == feature[name], (ident, name)
        categories = [query['category']]
        if query['category'] != 'recent':
            categories.append('targets')
            originals[(cell, arm, i)].append(int(row['correct']))
        if query.get('group') in ('early', 'eligible'):
            categories.append(query['group'])
        for category in categories:
            groups[(cell, arm, category)].append(row)
            per_history[(cell, arm, category, i)].append(int(row['correct']))
            diagnostics[(cell, arm, category)].append(dict(obsolete_any_error=obsolete_any, penultimate_error=penultimate,
                                                          **{k: v for k, v in feature.items() if isinstance(v, (int, float, np.integer, np.floating))}))
    expected_ids = {(cell, arm, i, rank) for cell, episodes in datasets.items()
                    for arm in arms(cell) for i, ep in enumerate(episodes) for rank in range(len(ep['queries']))}
    assert ids == expected_ids
    if (folder/'replay_audit.csv').exists():
        replay_rows = read_csv(folder/'replay_audit.csv')
        replay_ids = set()
        for row in replay_rows:
            ident = (row['cell'], row['arm'], int(row['episode']))
            assert ident not in replay_ids
            replay_ids.add(ident)
            assert all(int(row[key]) == value for key, value in plan_summary[ident].items())
        assert replay_ids == plan_ids
    vectors = {(cell, arm, category): np.asarray([np.mean(per_history[(cell, arm, category, i)])
                                                for i in range(len(datasets[cell]))])
               for cell, arm, category in groups}
    all_correct = {(cell, arm): np.asarray([all(originals[(cell, arm, i)]) for i in range(len(episodes))], dtype=float)
                   for cell, episodes in datasets.items() for arm in arms(cell)}
    saved_summary = read_csv(folder/'summary.csv')
    summary_ids = [(r['cell'], r['arm'], r['category']) for r in saved_summary]
    assert len(summary_ids) == len(set(summary_ids))
    required_summary = {key for key in groups if key[2] not in ('early', 'eligible')}
    assert required_summary <= set(summary_ids) <= set(groups)
    for row in saved_summary:
        cell, arm, category = row['cell'], row['arm'], row['category']
        subset = groups[(cell, arm, category)]
        assert int(row['queries']) == len(subset) and int(row['episodes']) == len(datasets[cell])
        assert np.isclose(float(row['accuracy']), np.mean([int(r['correct']) for r in subset]), atol=1e-12)
        assert np.isclose(float(row['mse']), np.mean([float(r['mse']) for r in subset]), atol=1e-12)
        grouped = defaultdict(list)
        for r in subset:
            grouped[int(r['episode'])].append(int(r['correct']))
        assert np.isclose(float(row['all_correct']), np.mean([all(r) for r in grouped.values()]), atol=1e-12)
        if category == 'revised' and 'obsolete_rate' in row:
            assert np.isclose(float(row['obsolete_rate']), np.mean([int(r['obsolete']) for r in subset]), atol=1e-12)

    resources = read_csv(folder/'resources.csv')
    resource_ids, application_writes = set(), 0
    for row in resources:
        cell, arm, start, size = row['cell'], row['arm'], int(row['batch_start']), int(row['batch'])
        ident = (cell, arm, start)
        assert ident not in resource_ids and start % batch_size == 0
        resource_ids.add(ident)
        assert size == min(batch_size, len(datasets[cell])-start)
        occupancy = np.asarray([plan_occupancy[(cell, arm, i)] for i in range(start, start+size)])
        assert np.array_equal(occupancy, json.loads(row['per_history_module_occupancy']))
        assert np.array_equal(occupancy.sum(axis=0), json.loads(row['module_occupancy']))
        assert np.array_equal(occupancy, json.loads(row['per_history_projection_applications']))
        assert np.array_equal(np.zeros_like(occupancy), json.loads(row['per_history_decay_applications']))
        assert int(row['active_module_updates']) == int(row['projection_applications']) == occupancy.sum()
        assert int(row['decay_applications']) == 0
        application_writes += int(row['active_module_updates'])
        ep = datasets[cell][start]
        replay = 0 if arm in ('online', 'reset', 'rolled') else ep['delay']
        writes = len(ep['keys'])+replay
        assert int(row['actual_writes']) == writes and int(row['replay_writes']) == replay
        assert int(row['query_state_write_count']) == (0 if arm == 'reset' else writes)
        assert int(row['final_fast_bytes']) == int(row['peak_fast_bytes']) == size*65536
        assert float(row['per_stream_final_bytes']) == 65536
        assert int(row['peak_buffer_pairs']) == (8 if arm.removesuffix('_bounded') in ('continuous', 'periodic') else 0)
        assert int(row['peak_snapshot_pairs']) == (4 if arm.removesuffix('_bounded') in ('continuous', 'periodic') else 0)
        assert float(row['boundary_error']) == float(row['midblock_error']) == 0
        assert row['resume_points'] == ('prefill,midblock' if ep['delay'] else 'prefill')
        assert 0 <= float(row['trace_saturation']) <= 1
    expected_resource_ids = {(cell, arm, start) for cell, episodes in datasets.items()
                             for arm in arms(cell) for start in range(0, len(episodes), batch_size)}
    assert resource_ids == expected_resource_ids and application_writes == total_writes
    diagnostic_summary = []
    for (cell, arm, category), values in sorted(diagnostics.items()):
        diagnostic_summary.append(dict(cell=cell, arm=arm, category=category,
                                       **{key: float(np.mean([v[key] for v in values])) for key in values[0]}))
    return dict(vectors=vectors, all_correct=all_correct, diagnostics=diagnostic_summary,
                audit=dict(status='passed', score_vectors=len(rows), causal_writes=total_writes,
                           extra_writes=extra_writes, stale_extra_writes=stale_extras,
                           refreshed_records=sum(p['refreshed_records'] for p in plan_summary.values()),
                           changed_values=sum(p['changed_values'] for p in plan_summary.values()),
                           fallback_writes=sum(p['fallback_writes'] for p in plan_summary.values()),
                           resource_batches=len(resources), literal_plans=len(plan_ids),
                           matched_candidate_cells=len(candidate_pairs)//4,
                           files={path.name: sha(path) for path in folder.iterdir() if path.is_file()},
                           scope='full saved-output arithmetic; no model inference'))
