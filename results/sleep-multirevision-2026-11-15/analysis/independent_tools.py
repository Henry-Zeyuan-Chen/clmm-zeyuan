"""Saved-artifact arithmetic only; deliberately imports no model/evaluator code."""
from collections import OrderedDict, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PARENTS = (90511, 90515)
ROUTES = ('content', *(f'random-{s}' for s in range(90701, 90706)))
ARMS = ('online', 'repeat', 'continuous', 'periodic')
PRESSURE_CELLS = tuple(f'pressure_k16_p{p}_d{d}' for p in (0, 2) for d in (0, 64, 128))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def signature(keys, values):
    return hashlib.sha256(bytes(keys) + b'|' + bytes(values)).hexdigest()


def nested_episodes(payload):
    """Find complete literal histories in saved nested dataset JSON."""
    if isinstance(payload, dict):
        keys, values = payload.get('keys'), payload.get('values')
        if (isinstance(keys, list) and isinstance(values, list) and keys and len(keys) == len(values)
                and all(type(k) is int and 0 <= k < 64 for k in keys)
                and all(type(v) is int and 0 <= v < 16 for v in values)):
            yield payload
        else:
            for item in payload.values():
                yield from nested_episodes(item)
    elif isinstance(payload, list):
        for item in payload:
            yield from nested_episodes(item)


def saved_signatures(path):
    path = Path(path)
    if path.suffix == '.json':
        return {signature(ep['keys'], ep['values'])
                for ep in nested_episodes(json.loads(path.read_text(encoding='utf-8')))}
    if path.suffix == '.npz':
        result = set()
        with np.load(path, allow_pickle=False) as arrays:
            keys, values, lengths = arrays['keys'], arrays['values'], arrays['lengths']
            assert keys.shape == values.shape and keys.ndim == 3
            for step, length in enumerate(lengths):
                for history in range(keys.shape[1]):
                    result.add(signature(keys[step, history, :int(length)].tolist(),
                                         values[step, history, :int(length)].tolist()))
        return result
    raise ValueError(f'Unsupported saved history container {path}')


def audit_dataset(datasets, n=128, excluded=()):
    """Validate the frozen six-target geometry from literal arrivals first."""
    expected_cells = ('acquire2', 'acquire4', 'revise2', 'gate_k16_r3', *PRESSURE_CELLS)
    assert set(datasets) == set(expected_cells)
    assert all(len(items) == n for items in datasets.values())
    signatures, revision_count, refresh_count = set(), 0, 0
    for name, episodes in datasets.items():
        for ep in episodes:
            keys, values = ep['keys'], ep['values']
            assert len(keys) == len(values)
            assert ep['signature'] == signature(keys, values)
            assert ep['signature'] not in signatures
            signatures.add(ep['signature'])
            assert all(type(k) is int and 0 <= k < 64 for k in keys)
            assert all(type(v) is int and 0 <= v < 16 for v in values)
            latest = {}
            for key, value in zip(keys, values):
                latest[key] = value
            for q in ep['queries']:
                assert q['target'] == latest[q['key']]
            if name in ('acquire2', 'acquire4', 'revise2'):
                assert ep['boundary'] == len(keys) and ep['delay'] == 0
                assert len({q['target'] for q in ep['queries']}) == len(ep['queries'])
                continue
            neutral = name == 'gate_k16_r3'
            tail = 0 if neutral else int(name.rsplit('_d', 1)[1])
            phase = None if neutral else int(name.split('_p', 1)[1].split('_')[0])
            assert ep['facts'] == 16 and ep['rounds'] == 3 and ep['revisions'] == 6
            assert ep['phase'] == phase and ep['retention_tail'] == tail
            assert len(keys) == (22 if neutral else 72+tail)
            assert ep['boundary'] == (22 if neutral else 16)
            assert ep['delay'] == (0 if neutral else 56+tail)
            assert len(set(keys[:16])) == 16
            original, revised = set(keys[:16]), keys[14:16]
            q0 = ep['queries'][:6]
            qkeys = [keys[p] for p in (0, 1, 12, 13, 14, 15)]
            assert [q['key'] for q in q0] == qkeys
            assert [q['group'] for q in q0] == ['early', 'early', 'eligible', 'eligible', 'revised', 'revised']
            truth, histories, kinds, counters, arrival_versions = {}, {}, [], {}, []
            for source, (key, value) in enumerate(zip(keys, values)):
                if source < 16:
                    kind = 'initial'
                    histories[str(key)] = [value]
                    counters[key] = 0
                elif key in original:
                    assert key in revised
                    if value == truth[key]:
                        kind = 'refresh'
                        refresh_count += 1
                    else:
                        kind = 'revision'
                        assert value not in histories[str(key)]
                        histories[str(key)].append(value)
                        counters[key] += 1
                        revision_count += 1
                else:
                    kind = 'distractor'
                    if key in truth:
                        assert truth[key] == value
                truth[key] = value
                kinds.append(kind)
                arrival_versions.append(counters.get(key, 0))
                if source >= 15:
                    assert len({truth[key] for key in qkeys}) == 6
            assert ep['event_kinds'] == kinds and ep['arrival_versions'] == arrival_versions
            assert ep['version_history'] == histories
            for rank, q in enumerate(q0):
                history = histories[str(q['key'])]
                is_revised = rank >= 4
                assert len(history) == (4 if is_revised else 1)
                assert q['versions'] == history and q['initial_value'] == history[0]
                assert q['old'] == (history[-2] if is_revised else -1)
                assert q['category'] == ('revised' if is_revised else 'stable')
            if neutral:
                assert kinds == ['initial']*16 + ['revision']*6
                assert len(ep['queries']) == 6
                continue
            assert len(ep['queries']) == (7 if tail else 6)
            if tail:
                q = ep['queries'][-1]
                assert q['category'] == q['group'] == 'recent'
                assert q['key'] == keys[-1] and q['target'] == values[-1]
                assert q['old'] == -1 and q['versions'] == [values[-1]]
            for r in range(3):
                start = 16+24*r
                assert kinds[start:start+4] == ['distractor', 'distractor', 'refresh', 'refresh']
                assert keys[start+2:start+4] == revised
                second = ['revision', 'revision', 'distractor', 'distractor'] if phase == 0 else ['distractor', 'distractor', 'revision', 'revision']
                assert kinds[start+4:start+8] == second
                if r < 2:
                    assert kinds[start+8:start+24] == ['distractor']*16
            assert kinds[72:] == ['distractor']*tail
    assert not signatures.intersection(excluded)
    for i in range(n):
        reference = datasets['gate_k16_r3'][i]
        neutral_revision = [(k, v) for k, v, kind in zip(reference['keys'], reference['values'], reference['event_kinds']) if kind == 'revision']
        for name in PRESSURE_CELLS:
            ep = datasets[name][i]
            assert ep['seed'] == reference['seed']
            assert ep['keys'][:16] == reference['keys'][:16]
            assert ep['values'][:16] == reference['values'][:16]
            assert ep['version_history'] == reference['version_history']
            assert ep['queries'][:6] == reference['queries']
            actual_revision = [(k, v) for k, v, kind in zip(ep['keys'], ep['values'], ep['event_kinds']) if kind == 'revision']
            assert actual_revision == neutral_revision
        for p in (0, 2):
            longest = datasets[f'pressure_k16_p{p}_d128'][i]
            for d in (0, 64):
                shorter = datasets[f'pressure_k16_p{p}_d{d}'][i]
                for field in ('keys', 'values', 'event_kinds', 'arrival_versions'):
                    assert shorter[field] == longest[field][:72+d]
        def distractors(ep):
            return [(k, v) for k, v, kind in zip(ep['keys'], ep['values'], ep['event_kinds']) if kind == 'distractor']
        assert distractors(datasets['pressure_k16_p0_d128'][i]) == distractors(datasets['pressure_k16_p2_d128'][i])
    return dict(cells=len(datasets), histories_per_cell=n, literal_histories=len(signatures),
                genuine_revision_arrivals=revision_count, refresh_arrivals=refresh_count,
                excluded_signature_count=len(set(excluded)), overlaps=0,
                scope='geometry reconstructed from literal arrivals; no model execution')


def independent_plan(ep, arm, capacity=8, block_size=4):
    """Reconstruct causal literal events from arrivals without any CLMM imports.

    The returned diagnostics evaluate staleness when an extra write is APPLIED,
    after the actual arrival at the same clock tick has changed the world.
    """
    assert arm in ARMS
    keys, values, boundary = ep['keys'], ep['values'], int(ep['boundary'])
    assert len(keys) == len(values) and 0 < boundary <= len(keys)
    assert capacity == 8 and block_size == 4
    assert all(type(k) is int and 0 <= k < 64 for k in keys)
    assert all(type(v) is int and 0 <= v < 16 for v in values)
    assert ep.get('signature', signature(keys, values)) == signature(keys, values)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(ep['seed']), 77])))
    fifo, latest, events, selected, blocks = OrderedDict(), {}, [], [], []

    def emit(source, kind, clock):
        source = int(source)
        event = dict(source=source, key=keys[source], value=values[source],
                     kind=kind, after_online=clock)
        events.append(event)

    def arrive(source):
        key = keys[source]
        latest[key] = values[source]
        if key in fifo:
            del fifo[key]
        fifo[key] = source
        if len(fifo) > capacity:
            fifo.popitem(last=False)
        emit(source, 'online', source)

    for source in range(boundary):
        arrive(source)
    if arm not in ('continuous', 'periodic'):
        for source in range(boundary, len(keys)):
            arrive(source)
            if arm == 'repeat':
                emit(source, 'extra', source)
    for start in range(boundary, len(keys), block_size) if arm in ('continuous', 'periodic') else ():
        stop = min(start + block_size, len(keys))
        available = tuple(fifo.values())
        assert available and max(available) < start
        # Production scheduler reserves four draws even for a partial block.
        sample = [int(rng.choice(available)) for _ in range(block_size)]
        used = sample[:stop-start]
        selected.extend(used)
        blocks.append(dict(start=start, stop=stop, available=list(available),
                           sample=sample, selected=used))
        for offset, source in enumerate(range(start, stop)):
            arrive(source)
            if arm == 'continuous':
                emit(sample[offset], 'extra', source)
        if arm == 'periodic':
            for source in used:
                emit(source, 'extra', stop - 1)

    # Evaluate latest actual world values at the exact applied event order.
    world, stale, current = {}, 0, 0
    for event in events:
        if event['kind'] == 'online':
            world[event['key']] = event['value']
        else:
            is_stale = event['value'] != world[event['key']]
            stale += is_stale
            current += not is_stale
    extras = [event for event in events if event['kind'] == 'extra']
    assert len(events) == len(keys) + (0 if arm == 'online' else len(keys)-boundary)
    return dict(events=events, selected=selected, blocks=blocks,
                boundary=boundary, writes=len(events), real_observations=len(keys),
                extra_writes=len(extras), stale_extra_writes=stale,
                current_extra_writes=current,
                final_buffer_sources=list(fifo.values()) if arm in ('continuous', 'periodic') else [])


def query_diagnostics(ep, plan, query, route_table=None):
    """Independent applied-write exposure and key-specific obsolete labels."""
    key, target = int(query['key']), int(query['target'])
    keys, values = ep['keys'], ep['values']
    versions = list(dict.fromkeys(v for k, v in zip(keys, values) if k == key))
    assert versions and versions[-1] == target
    old_values = versions[:-1]
    events = plan['events']
    real_target = max(i for i, event in enumerate(events)
                      if event['kind'] == 'online' and event['key'] == key)
    target_write = max(i for i, event in enumerate(events)
                       if event['key'] == key and event['value'] == target)
    afterward = events[target_write+1:]
    own_extra = [event for event in events if event['kind'] == 'extra' and event['key'] == key]
    direct = [event for event in events if event['key'] == key]
    latest, stale_own = {}, 0
    for event in events:
        if event['kind'] == 'online':
            latest[event['key']] = event['value']
        elif event['key'] == key:
            stale_own += event['value'] != latest[key]
    key_residence, final_residence, expected_draws, final_expected_draws = 0, 0, 0., 0.
    final_any_probability = []
    for block in plan['blocks']:
        available = block['available']
        same_key = [i for i in available if keys[i] == key]
        final = [i for i in same_key if values[i] == target]
        n = block['stop']-block['start']
        key_residence += bool(same_key)
        final_residence += bool(final)
        expected_draws += n * len(same_key) / len(available)
        final_expected_draws += n * len(final) / len(available)
        final_any_probability.append(1-(1-len(final)/len(available))**n)
    result = dict(old_values=old_values, penultimate=old_values[-1] if old_values else -1,
                  after_last_real_target_observation=len(events)-real_target-1,
                  after_final_target_valued_write=len(afterward),
                  subsequent_obsolete_same_key_writes=sum(e['key'] == key and e['value'] != target for e in afterward),
                  final_applied_value_for_key=next(e['value'] for e in reversed(events) if e['key'] == key),
                  last_target_valued_event=target_write,
                  last_direct_write_kind=direct[-1]['kind'], last_direct_write_source=direct[-1]['source'],
                  target_extra_writes=len(own_extra), stale_target_extra_writes=stale_own,
                  query_final_obsolete_extra_writes=sum(e['value'] in old_values for e in own_extra),
                  final_target_extra_writes=sum(e['value'] == target for e in own_extra),
                  final_fifo_has_key=int(any(keys[s] == key for s in plan['final_buffer_sources'])),
                  key_available_blocks=key_residence, final_target_available_blocks=final_residence,
                  expected_key_draws=expected_draws, expected_final_target_draws=final_expected_draws,
                  final_target_any_draw_probability_by_block=final_any_probability)
    if route_table is not None:
        module = int(route_table[key])
        same = sum(int(route_table[event['key']]) == module for event in afterward)
        real_following = events[real_target+1:]
        real_same = sum(int(route_table[event['key']]) == module for event in real_following)
        result.update(query_module=module, following_same_module_writes=same,
                      following_other_module_writes=len(afterward)-same,
                      after_real_same_module_writes=real_same,
                      after_real_other_module_writes=len(real_following)-real_same)
    return result


def bootstrap_ids(n=128, draws=5000, seed=91104):
    return np.random.Generator(np.random.PCG64(seed)).integers(0, n, size=(draws, n))


def interval(vector, ids, level=.95):
    vector = np.asarray(vector, dtype=np.float64)
    assert vector.shape == (ids.shape[1],) and np.isfinite(vector).all()
    tail = (1-level)/2
    lo, hi = np.quantile(vector[ids].mean(axis=1), [tail, 1-tail])
    return dict(estimate=float(vector.mean()), lower=float(lo), upper=float(hi), interval_level=level)


def aggregate_histories(vectors, cells, arm, category, family='balanced', parents=PARENTS, routes=ROUTES):
    """vectors[(parent,route,cell,arm,category)] => one value per paired history."""
    def family_vector(wanted_routes):
        return np.mean([vectors[(parent, route, cell, arm, category)]
                        for parent in parents for route in wanted_routes for cell in cells], axis=0)
    if family == 'balanced':
        return .5*family_vector(('content',)) + .5*family_vector(routes[1:])
    if family == 'content':
        return family_vector(('content',))
    if family == 'mean5random':
        return family_vector(routes[1:])
    raise ValueError(f'Unknown route family {family}')
