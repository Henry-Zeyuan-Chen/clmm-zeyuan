"""Paired, observer-labelled streams with three genuine changes per query key.

Only ``keys`` and ``values`` are observations. Version histories, query labels,
and the event audit below belong to the evaluator and must not reach a replay
controller or a model. All random draws use private PCG64 generators.
"""
from __future__ import annotations

import numpy as np

from clmm.pretrain.kv_streams import NKEYS, NVALUES, evaluation_data, signature


PHASES = (0, 2)
RETENTION_TAILS = (0, 64, 128)
PRESSURE_CELLS = tuple(f"pressure_k16_p{phase}_d{tail}"
                       for phase in PHASES for tail in RETENTION_TAILS)
_QUERY_POSITIONS = (0, 1, 12, 13, 14, 15)
_ROUNDS = 3
_FACTS = 16


def _world(seed):
    """Draw a complete source world once, without a phase or tail argument."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 16, 3, 811])))
    permutation = rng.permutation(NKEYS).tolist()
    initial = permutation[:_FACTS]
    values = rng.integers(NVALUES, size=_FACTS).tolist()
    for position, value in zip(_QUERY_POSITIONS, rng.choice(NVALUES, 6, replace=False).tolist()):
        values[position] = value
    histories = {str(key): [value] for key, value in zip(initial, values)}
    latest = dict(zip(initial, values))
    protected = [initial[position] for position in _QUERY_POSITIONS]
    revised = initial[14:16]
    for _ in range(_ROUNDS):
        for key in revised:
            # Distinct current targets after EACH write, and no return to any
            # of this key's earlier values (not merely its immediate old one).
            banned = {latest[item] for item in protected} | set(histories[str(key)])
            value = int(rng.choice([item for item in range(NVALUES) if item not in banned]))
            histories[str(key)].append(value)
            latest[key] = value
    table = rng.integers(NVALUES, size=NKEYS).tolist()
    distractors = rng.choice(permutation[_FACTS:], size=172, replace=True).tolist()
    return dict(seed=seed, initial=initial, initial_values=values, histories=histories,
                revised=revised, distractors=distractors, table=table)


def _queries(world):
    queries = []
    for position in _QUERY_POSITIONS:
        key = world["initial"][position]
        versions = list(world["histories"][str(key)])
        revised = len(versions) > 1
        queries.append(dict(key=key, target=versions[-1], old=versions[-2] if revised else -1,
                            category="revised" if revised else "stable",
                            group="early" if position < 2 else "eligible" if position < 14 else "revised",
                            versions=versions, initial_value=versions[0]))
    return queries


def _episode(world, phase, tail, *, neutral=False):
    keys, values = list(world["initial"]), list(world["initial_values"])
    kinds, versions = ["initial"] * _FACTS, [0] * _FACTS
    cursor = 0

    def write(key, value, kind, version=0):
        keys.append(key)
        values.append(value)
        kinds.append(kind)
        versions.append(version)

    def distract(count):
        nonlocal cursor
        for key in world["distractors"][cursor:cursor + count]:
            write(key, world["table"][key], "distractor")
        cursor += count

    def revise(round_index):
        for key in world["revised"]:
            write(key, world["histories"][str(key)][round_index + 1], "revision", round_index + 1)

    for round_index in range(_ROUNDS):
        if neutral:
            revise(round_index)
            continue
        distract(2)
        for key in world["revised"]:
            write(key, world["histories"][str(key)][round_index], "refresh", round_index)
        if phase == 0:
            revise(round_index)
            distract(2)
        else:
            distract(2)
            revise(round_index)
        if round_index < _ROUNDS - 1:
            distract(16)
    if not neutral:
        distract(tail)
    boundary = len(keys) if neutral else _FACTS
    queries = _queries(world)
    if tail and not neutral:
        queries.append(dict(key=keys[-1], target=values[-1], old=-1, category="recent", group="recent",
                            versions=[values[-1]], initial_value=values[-1]))
    return dict(seed=world["seed"], facts=_FACTS, delay=len(keys) - boundary,
                revisions=2 * _ROUNDS, pressure=not neutral, keys=keys, values=values,
                boundary=boundary, queries=queries, signature=signature(keys, values),
                retention_tail=0 if neutral else tail, rounds=_ROUNDS,
                phase=None if neutral else phase, event_kinds=kinds, arrival_versions=versions,
                version_history={key: list(history) for key, history in world["histories"].items()})


def make_datasets(seed, episodes=128, seen=()):
    """Return three existing short gates, one neutral gate, and six pressure cells.

    The seven new cells share accepted base seeds by episode index. A base is
    rejected jointly if ANY of its seven whole-history signatures is excluded,
    a short-gate history, or an already accepted history. ``delay`` deliberately
    counts *all* post-prefill arrivals (56 + retention_tail), so the unchanged
    ``kv_streams.replay_plan`` budget and ``SleepSession`` interface apply.
    ``revisions`` counts six genuine writes; ``rounds`` counts three per key.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    excluded = set(seen)
    datasets = evaluation_data(seed, episodes=episodes, seen=excluded, suite="gate")
    excluded.update(ep["signature"] for items in datasets.values() for ep in items)
    datasets["gate_k16_r3"] = []
    datasets.update((name, []) for name in PRESSURE_CELLS)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 16, 3, 829])))
    attempts = 0
    while len(datasets["gate_k16_r3"]) < episodes:
        attempts += 1
        if attempts > max(10000, episodes * 100):
            raise RuntimeError("Multi-revision evaluation rejection cap exceeded")
        world = _world(int(rng.integers(2_000_000_000, 3_000_000_000)))
        candidates = {"gate_k16_r3": _episode(world, 0, 0, neutral=True)}
        candidates.update((f"pressure_k16_p{phase}_d{tail}", _episode(world, phase, tail))
                          for phase in PHASES for tail in RETENTION_TAILS)
        signatures = {ep["signature"] for ep in candidates.values()}
        if signatures.intersection(excluded):
            continue
        for name, ep in candidates.items():
            datasets[name].append(ep)
        excluded.update(signatures)
    return datasets


def classify_events(ep, events):
    """Copy chronological literal events and add observer-only truth labels.

    ``latest_online`` is reconstructed solely from real episode arrivals up to
    ``after_online``. An extra is ``stale`` iff it differs from that truth.
    ``obsolete_anyquery`` instead compares a queried key with its FINAL target:
    it recognizes every older value in that key's complete version history.
    It can therefore be true before a replay is stale at its emission time.
    Neither the supplied events nor the episode is modified.
    """
    queries = {query["key"]: query for query in ep["queries"]}
    keys, values = ep["keys"], ep["values"]
    latest, cursor, previous_time, classified = {}, 0, -1, []
    for event in events:
        source, at = event["source"], event["after_online"]
        if (type(source) is not int or type(at) is not int
                or not 0 <= source <= at < len(keys) or at < previous_time
                or event["kind"] not in ("online", "extra")
                or (event["key"], event["value"]) != (keys[source], values[source])
                or event["kind"] == "online" and source != at):
            raise ValueError("Events must be chronological, causal literal observation records")
        while cursor <= at:
            latest[keys[cursor]] = values[cursor]
            cursor += 1
        previous_time = at
        query = queries.get(event["key"])
        history = query.get("versions", [query["target"]]) if query else []
        extra = event["kind"] == "extra"
        classified.append(dict(event, latest_online=latest[event["key"]],
                               stale=extra and event["value"] != latest[event["key"]],
                               query_group=query.get("group", query["category"]) if query else None,
                               obsolete_anyquery=bool(extra and query and event["value"] != query["target"]
                                                      and event["value"] in history[:-1])))
    return classified
