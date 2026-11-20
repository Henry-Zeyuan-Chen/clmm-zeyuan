"""CPU parity, causality, bounded-storage and continuation checks."""
from collections import OrderedDict
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clmm.pretrain.kv_scheduler import StreamingReplay


@pytest.fixture(scope="module")
def streams():
    checkout = Path(__file__).resolve().parents[1]
    repo = Path(os.environ.get("CLMM_FROZEN_REPO", checkout)).resolve()
    sys.path.insert(0, str(repo))
    module = importlib.import_module("clmm.pretrain.kv_streams")
    assert Path(module.__file__).resolve() == repo / "clmm/pretrain/kv_streams.py"
    return module


def run_live(ep, arm, resume_after=None):
    policy = StreamingReplay(arm, ep["seed"])
    events = [policy.prime(key, value) for key, value in zip(ep["keys"][:ep["boundary"]],
                                                          ep["values"][:ep["boundary"]])]
    for index in range(ep["boundary"], len(ep["keys"])):
        events.extend(policy.step(ep["keys"][index], ep["values"][index]))
        if resume_after is not None and index - ep["boundary"] + 1 == resume_after:
            policy = StreamingReplay.from_state(json.loads(json.dumps(policy.export_state())))
    events.extend(policy.finish())
    return events, policy


@pytest.mark.parametrize("arm", StreamingReplay._arms)
@pytest.mark.parametrize("facts", [8, 16, 32])
@pytest.mark.parametrize("delay", [0, 3, 16, 64])
@pytest.mark.parametrize("revisions", [0, 2])
@pytest.mark.parametrize("seed", [90401, 90403])
def test_live_events_exactly_match_frozen_offline_plan(streams, arm, facts, delay, revisions, seed):
    ep = streams.episode(seed, facts, delay, revisions, pressure=True)
    actual, policy = run_live(ep, arm)
    assert actual == streams.replay_plan(ep, arm)["events"]
    assert policy.export_state()["closed"]
    assert policy.logical_counts()["snapshot_pairs"] == 0
    if arm in ("online", "repeat"):
        assert policy.logical_counts() == {"buffer_pairs": 0, "snapshot_pairs": 0}


@pytest.mark.parametrize("arm", StreamingReplay._arms)
@pytest.mark.parametrize("delay", [3, 16, 64])
@pytest.mark.parametrize("resume_after", [1, 2, 3, 4])
def test_json_midblock_resume_preserves_complete_event_stream(streams, arm, delay, resume_after):
    ep = streams.episode(90411, 16, delay, 2, pressure=True)
    resumed, policy = run_live(ep, arm, resume_after)
    full, reference = run_live(ep, arm)
    assert resumed == full == streams.replay_plan(ep, arm)["events"]
    assert policy.export_state() == reference.export_state()


def test_snapshots_use_only_causal_latest_fifo_literals_and_replay_does_not_refresh_age(streams):
    ep = streams.episode(90413, 16, 16, 2, pressure=True)
    for arm in ("continuous", "periodic"):
        policy, expected = StreamingReplay(arm, ep["seed"]), OrderedDict()
        for source, (key, value) in enumerate(zip(ep["keys"], ep["values"])):
            before = dict((record[0], record) for record in expected.values())
            if source < ep["boundary"]:
                events = [policy.prime(key, value)]
            else:
                events = policy.step(key, value)
                if (source - ep["boundary"]) % 4 == 0:
                    assert all(tuple(record) == before[record[0]] for record in policy.export_state()["snapshot"])
            expected.pop(key, None)
            expected[key] = (source, key, value)
            while len(expected) > 8:
                expected.popitem(last=False)
            assert policy.export_state()["buffer"] == [list(record) for record in expected.values()]
            for event in events:
                assert event["source"] <= event["after_online"] == source
                assert (event["key"], event["value"]) == (ep["keys"][event["source"]], ep["values"][event["source"]])


def test_partial_periodic_finish_flushes_only_arrived_count_then_closes():
    policy = StreamingReplay("periodic", 14)
    policy.prime(1, 2)
    assert len(policy.step(3, 4)) == 1
    assert len(policy.step(5, 6)) == 1
    snapshot = policy.export_state()["snapshot"]
    emitted = policy.finish()
    assert [event["source"] for event in emitted] == [record[0] for record in snapshot[:2]]
    assert all(event["after_online"] == 2 for event in emitted)
    assert policy.finish() == []
    assert policy.logical_counts() == {"buffer_pairs": 3, "snapshot_pairs": 0}
    for restored in (policy, StreamingReplay.from_state(json.loads(json.dumps(policy.export_state())))):
        with pytest.raises(RuntimeError):
            restored.step(0, 0)
        with pytest.raises(RuntimeError):
            restored.prime(0, 0)


@pytest.mark.parametrize("arm", StreamingReplay._arms)
def test_live_memory_and_json_payload_remain_bounded_after_ten_thousand_arrivals(arm):
    policy = StreamingReplay(arm, 90401)
    for key in range(16):
        policy.prime(key, key % 16)
    sizes = []
    for index in range(10_000):
        policy.step(index % 64, index % 16)
        counts = policy.logical_counts()
        assert counts["buffer_pairs"] <= 8 and counts["snapshot_pairs"] <= 4
        assert len(policy._buffer) <= 8 and len(policy._snapshot) <= 4
        if index in (98, 998, 9998):
            payload = policy.export_state()
            assert not ({"history", "events", "episode", "queries"} & set(payload))
            assert len(payload["buffer"]) <= 8 and len(payload["snapshot"]) <= 4
            sizes.append(len(json.dumps(payload)))
    assert max(sizes) < 3000 and max(sizes) - min(sizes) < 250
    if arm in ("online", "repeat"):
        assert policy.logical_counts() == {"buffer_pairs": 0, "snapshot_pairs": 0}


@pytest.mark.parametrize("corruption", ["schema", "history", "config", "counter", "counter_gap", "key", "value",
    "source", "duplicate", "order", "snapshot_future", "offset", "rng", "seed", "started", "closed"])
def test_import_rejects_invalid_or_mismatched_live_state(corruption):
    policy = StreamingReplay("continuous", 90401)
    for key in range(8):
        policy.prime(key, key)
    policy.step(8, 8)
    payload = deepcopy(policy.export_state())
    if corruption == "schema": payload["schema_version"] = True
    elif corruption == "history": payload["events"] = []
    elif corruption == "config": payload["config"]["arm"] = "periodic"
    elif corruption == "counter": payload["next_source"] = -1
    elif corruption == "counter_gap": payload["next_source"] += 1
    elif corruption == "key": payload["buffer"][0][1] = 64
    elif corruption == "value": payload["buffer"][0][2] = 16
    elif corruption == "source": payload["buffer"][0][0] = payload["next_source"]
    elif corruption == "duplicate": payload["buffer"][1][1] = payload["buffer"][0][1]
    elif corruption == "order": payload["buffer"].reverse()
    elif corruption == "snapshot_future": payload["snapshot"][0][0] = payload["next_source"] - 1
    elif corruption == "offset": payload["in_block"] = 4
    elif corruption == "rng": payload["rng"]["has_uint32"] = 2
    elif corruption == "seed": payload["rng"]["state"]["inc"] ^= 2
    elif corruption == "started": payload["started"] = False
    elif corruption == "closed": payload["closed"] = True
    with pytest.raises(ValueError):
        StreamingReplay.from_state(payload)


def test_invalid_arrivals_are_atomic_and_priming_cannot_resume_after_first_step():
    policy = StreamingReplay("continuous", 90401)
    policy.prime(0, 0)
    snapshot = policy.export_state()
    for key, value in ((-1, 0), (64, 0), (0, 16), (True, 0), (0, 1.0)):
        with pytest.raises(ValueError): policy.step(key, value)
        assert policy.export_state() == snapshot
    policy.step(1, 1)
    with pytest.raises(RuntimeError): policy.prime(2, 2)
    with pytest.raises(ValueError): StreamingReplay("continuous", 1).step(0, 0)


@pytest.mark.parametrize("changes", [{"arm": "other"}, {"seed": -1}, {"seed": True},
    {"capacity": 0}, {"capacity": 9}, {"block_size": 0}, {"block_size": 5},
    {"nkeys": 0}, {"nvalues": 0}])
def test_configuration_is_validated(changes):
    config = dict(arm="continuous", seed=90401)
    config.update(changes)
    with pytest.raises(ValueError):
        StreamingReplay(**config)

