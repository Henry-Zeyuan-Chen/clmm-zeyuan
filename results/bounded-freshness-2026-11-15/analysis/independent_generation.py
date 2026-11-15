"""Independent pure-NumPy regeneration of the declared random literal streams."""
import hashlib
import json

import numpy as np

from independent_tools import PRESSURE_CELLS, signature


def _rng(*seed_words):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_words)))


def _short(seed, facts, revisions):
    rng = _rng(seed, facts, revisions)
    keys = rng.permutation(64).tolist()[:facts]
    values = rng.choice(16, facts, replace=False).tolist()
    latest = dict(zip(keys, values))
    for position in range(revisions):
        key = keys[position]
        allowed = [value for value in range(16) if value not in set(latest.values())]
        value = int(rng.choice(allowed))
        keys.append(key)
        values.append(value)
        latest[key] = value
    return keys, values


def _world(seed):
    rng = _rng(seed, 16, 3, 811)
    order = rng.permutation(64).tolist()
    initial = order[:16]
    values = rng.integers(16, size=16).tolist()
    targets = (0, 1, 12, 13, 14, 15)
    for position, value in zip(targets, rng.choice(16, 6, replace=False).tolist()):
        values[position] = value
    history = {key: [value] for key, value in zip(initial, values)}
    truth = dict(zip(initial, values))
    for _ in range(3):
        for key in initial[14:16]:
            forbidden = {truth[initial[position]] for position in targets} | set(history[key])
            value = int(rng.choice([candidate for candidate in range(16) if candidate not in forbidden]))
            history[key].append(value)
            truth[key] = value
    distractor_values = rng.integers(16, size=64).tolist()
    distractor_keys = rng.choice(order[16:], size=172, replace=True).tolist()
    return initial, values, history, distractor_keys, distractor_values


def _stream(world, phase=None, tail=0):
    initial, initial_values, histories, distractors, table = world
    keys, values = list(initial), list(initial_values)
    position = 0

    def background(n):
        nonlocal position
        selected = distractors[position:position+n]
        keys.extend(selected)
        values.extend(table[key] for key in selected)
        position += n

    def pair(version):
        keys.extend(initial[14:16])
        values.extend(histories[key][version] for key in initial[14:16])

    for version in (1, 2, 3):
        if phase is None:
            pair(version)
            continue
        background(2)
        pair(version-1)
        if phase == 0:
            pair(version)
            background(2)
        else:
            background(2)
            pair(version)
        if version < 3:
            background(16)
    if phase is not None:
        background(tail)
    return keys, values


def audit_generation(datasets, development_seed, n, excluded):
    excluded = set(excluded)
    short_signatures, comparisons = set(), 0
    for index, (name, facts, revisions) in enumerate((('acquire2', 2, 0), ('acquire4', 4, 0), ('revise2', 2, 1))):
        rng = _rng(development_seed, facts, revisions, index)
        accepted = 0
        while accepted < n:
            candidate = int(rng.integers(2_000_000_000, 3_000_000_000))
            keys, values = _short(candidate, facts, revisions)
            digest = signature(keys, values)
            if digest in excluded:
                continue
            actual = datasets[name][accepted]
            assert actual['seed'] == candidate
            assert (actual['keys'], actual['values']) == (keys, values)
            short_signatures.add(digest)
            accepted += 1
            comparisons += 1
    excluded.update(short_signatures)
    rng = _rng(development_seed, 16, 3, 829)
    accepted, attempts, base_seeds = 0, 0, []
    while accepted < n:
        attempts += 1
        assert attempts <= max(10000, n*100)
        candidate = int(rng.integers(2_000_000_000, 3_000_000_000))
        world = _world(candidate)
        streams = {'gate_k16_r3': _stream(world)}
        streams.update((f'pressure_k16_p{phase}_d{tail}', _stream(world, phase, tail))
                       for phase in (0, 2) for tail in (0, 64, 128))
        signatures = {signature(keys, values) for keys, values in streams.values()}
        if signatures.intersection(excluded):
            continue
        for name, (keys, values) in streams.items():
            actual = datasets[name][accepted]
            assert actual['seed'] == candidate
            assert (actual['keys'], actual['values']) == (keys, values), (name, accepted)
            comparisons += 1
        excluded.update(signatures)
        base_seeds.append(candidate)
        accepted += 1
    assert set(streams) == {'gate_k16_r3', *PRESSURE_CELLS}
    return dict(status='passed', literal_streams_exact=comparisons, development_seed=development_seed,
                accepted_base_histories=accepted, candidate_attempts=attempts,
                base_seed_sequence_sha256=hashlib.sha256(json.dumps(base_seeds).encode()).hexdigest(),
                scope='independent private PCG64 draws, rejection logic and full keys/values equality; no CLMM imports')
