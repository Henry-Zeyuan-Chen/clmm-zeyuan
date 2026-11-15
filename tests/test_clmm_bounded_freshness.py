"""Bounded replay freshness, literal compatibility, and atomic continuation."""
from copy import deepcopy
import io
import json

import pytest
import torch

from clmm.pretrain.kv_control import DeltaAdapter
from clmm.pretrain.kv_scheduler import StreamingReplay
from clmm.sleep import SleepSession


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def transport(session):
    payload = session.export_state()
    payload["policies"] = json.loads(json.dumps(payload["policies"]))
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    return SleepSession.from_state(session.model, torch.load(stream, weights_only=True))


def pair(event):
    return event["source"], event["key"], event["value"]


@pytest.mark.parametrize("arm", ["continuous", "periodic"])
def test_same_arrival_revision_is_resolved_after_fifo_update(arm):
    policy = StreamingReplay(arm, 14, freshness="bounded")
    assert policy.prime(0, 1) == dict(source=0, key=0, value=1, kind="online", after_online=0)
    emitted = policy.step(0, 2)
    frozen = [[0, 0, 1]] * 4
    assert policy.export_state()["snapshot"] == frozen
    extras = emitted[1:] if arm == "continuous" else policy.finish()
    assert extras == [dict(source=1, key=0, value=2, kind="extra", after_online=1,
                           sampled_source=0, sampled_value=1, refreshed=True, fallback=False)]
    assert policy.export_state()["buffer"] == [[1, 0, 2]]


def test_later_continuous_replays_resolve_again_without_mutating_frozen_samples():
    policy = StreamingReplay("continuous", 14, freshness="bounded")
    policy.prime(0, 1)
    first = policy.step(1, 2)[1]
    assert pair(first) == (0, 0, 1) and not first["refreshed"] and not first["fallback"]
    assert pair(policy.step(0, 3)[1]) == (2, 0, 3)
    # A real refresh with the same value still identifies the newer literal source.
    third = policy.step(0, 3)[1]
    assert pair(third) == (3, 0, 3) and third["refreshed"]
    assert policy.export_state()["snapshot"] == [[0, 0, 1]] * 4
    fourth = policy.step(2, 4)[1]
    assert pair(fourth) == (3, 0, 3)
    assert fourth["sampled_source"] == 0 and fourth["sampled_value"] == 1


@pytest.mark.parametrize("arm", ["continuous", "periodic"])
def test_eviction_falls_back_to_original_sample_not_a_previous_resolution(arm):
    policy = StreamingReplay(arm, 14, capacity=2, freshness="bounded")
    policy.prime(0, 1)
    first = policy.step(0, 2)
    second = policy.step(1, 3)
    third = policy.step(2, 4)
    assert policy.export_state()["snapshot"] == [[0, 0, 1]] * 4
    assert policy.export_state()["buffer"] == [[2, 1, 3], [3, 2, 4]]
    if arm == "continuous":
        assert pair(first[1]) == pair(second[1]) == (1, 0, 2)
        extras = [third[1]]
        assert policy.finish() == []
    else:
        extras = policy.finish()
        assert len(extras) == 3
    assert all(pair(event) == (0, 0, 1) and event["fallback"] and not event["refreshed"]
               for event in extras)
    # Replay and finish never resurrect an evicted key or refresh FIFO age.
    assert policy.export_state()["buffer"] == [[2, 1, 3], [3, 2, 4]]


@pytest.mark.parametrize("arm", ["continuous", "periodic"])
@pytest.mark.parametrize("capacity,block_size", [(8, 4), (2, 3), (1, 1)])
def test_sampling_rng_keys_budgets_and_storage_match_literal(arm, capacity, block_size):
    literal = StreamingReplay(arm, 90401, capacity=capacity, block_size=block_size)
    bounded = StreamingReplay(arm, 90401, capacity=capacity, block_size=block_size, freshness="bounded")
    observed = []
    for key in range(16):
        observed.append((key, key % 16))
        assert literal.prime(*observed[-1]) == bounded.prime(*observed[-1])
    extra_count = 0
    for index in range(1001):
        # Revisions, same-value refreshes, and new keys compete for the same FIFO.
        observed.append((index % 11, (index // 3) % 16))
        old, new = literal.step(*observed[-1]), bounded.step(*observed[-1])
        assert len(old) == len(new) and old[0] == new[0]
        old_state, new_state = literal.export_state(), bounded.export_state()
        for name in ("buffer", "snapshot", "rng", "in_block", "next_source"):
            assert old_state[name] == new_state[name]
        fifo = {record[1]: tuple(record) for record in new_state["buffer"]}
        for sampled, actual in zip(old[1:], new[1:]):
            extra_count += 1
            assert actual["key"] == sampled["key"]
            assert (actual["sampled_source"], actual["sampled_value"]) == (sampled["source"], sampled["value"])
            assert pair(actual) == fifo.get(sampled["key"], pair(sampled))
            assert actual["fallback"] == (sampled["key"] not in fifo)
            assert actual["refreshed"] == (pair(actual) != pair(sampled))
            assert observed[actual["source"]] == (actual["key"], actual["value"])
            assert actual["source"] <= actual["after_online"] == len(observed) - 1
        assert bounded.logical_counts()["buffer_pairs"] <= capacity <= 8
        assert bounded.logical_counts()["snapshot_pairs"] <= block_size <= 4
        assert set(bounded.__dict__) == set(literal.__dict__)
        assert set(new_state) == set(old_state)
    old, new = literal.finish(), bounded.finish()
    extra_count += len(new)
    assert len(old) == len(new)
    assert [e["key"] for e in old] == [e["key"] for e in new]
    assert extra_count == 1001
    assert bounded.export_state()["rng"] == literal.export_state()["rng"]
    assert bounded.export_state()["buffer"] == literal.export_state()["buffer"]


@pytest.mark.parametrize("count", [1, 2, 3])
def test_partial_periodic_finish_uses_latest_literals_and_exact_budget(count):
    policy = StreamingReplay("periodic", 7, freshness="bounded")
    policy.prime(0, 0)
    for value in range(1, count + 1):
        assert len(policy.step(0, value)) == 1
    before = policy.export_state()
    emitted = policy.finish()
    assert len(emitted) == count
    assert all(pair(event) == (count, 0, count) and event["sampled_source"] == 0
               and event["sampled_value"] == 0 and event["refreshed"] and not event["fallback"]
               and event["after_online"] == count for event in emitted)
    assert policy.export_state()["rng"] == before["rng"]
    assert policy.export_state()["buffer"] == before["buffer"]
    assert policy.finish() == []
    restored = StreamingReplay.from_state(json.loads(json.dumps(policy.export_state())))
    assert restored.freshness == "bounded" and restored.finish() == []
    with pytest.raises(RuntimeError):
        restored.step(0, 0)


@pytest.mark.parametrize("arm", ["continuous", "periodic"])
@pytest.mark.parametrize("pause_after", [1, 2, 3])
def test_batched_midblock_resume_preserves_revisions_events_state_budgets_and_reads(arm, pause_after):
    model = DeltaAdapter()
    full = SleepSession(model, 2, arm, [14, 15], freshness="bounded")
    for key, value in [(0, 0), (1, 1)]:
        full.prime(torch.tensor([key, key + 8]), torch.tensor([value, value + 1]))
    paused = transport(full)
    revisions = [(0, 2), (1, 3), (0, 4), (2, 5), (0, 6), (3, 7), (1, 8)]
    for index, (key, value) in enumerate(revisions, 1):
        keys, values = torch.tensor([key, key + 8]), torch.tensor([value, value + 1])
        assert full.observe(keys, values) == paused.observe(keys, values)
        if index == pause_after:
            paused = transport(paused)
            assert paused.freshness == "bounded"
            assert paused.export_state()["schema_version"] == 2
            assert all(p["config"]["freshness"] == "bounded" for p in paused.export_state()["policies"])
        assert equal(full.export_state(), paused.export_state())
        assert full.budget() == paused.budget()
        assert full.budget()["replay_writes"] == (index if arm == "continuous" else index // 4 * 4)
        before = paused.export_state()
        queries = torch.tensor([[0, 1, 2], [8, 9, 10]])
        assert torch.equal(full.read(queries), paused.read(queries))
        assert equal(before, paused.export_state())
    assert full.finish() == paused.finish()
    assert equal(full.export_state(), paused.export_state())
    budget = paused.budget()
    assert budget["prime_writes"] == 2 and budget["delay_observations"] == 7
    assert budget["replay_writes"] == 7 and budget["total_writes"] == 16
    assert budget["freshness"] == "bounded" and budget["snapshot_pairs"] == [0, 0]
    assert transport(paused).budget() == budget


# Captured from the unmodified v1 scheduler at 159314d, including insertion order.
LEGACY_STATE_JSON = (
    '{"schema_version":1,"config":{"arm":"periodic","seed":14,"capacity":8,"block_size":4,'
    '"nkeys":64,"nvalues":16},"config_fingerprint":"033a1677e2a794a199709310b2d8fd2270b728b4c21b45fa7ca27a47fe9c26a5",'
    '"next_source":4,"buffer":[[1,3,4],[2,1,5],[3,6,7]],"snapshot":[[1,3,4],[1,3,4],[1,3,4],[0,1,2]],'
    '"in_block":2,"rng":{"bit_generator":"PCG64","state":{"state":290355492418615979966321591611057142402,'
    '"inc":208230829656793207476646185387660152143},"has_uint32":0,"uinteger":2128070884},"started":true,"closed":false}'
)


def test_literal_v1_golden_export_events_resume_and_delta_effects_are_unchanged():
    model = DeltaAdapter()
    default = SleepSession(model, 1, "periodic", [14])
    explicit = SleepSession(model, 1, "periodic", [14], freshness="literal")
    for index, (key, value) in enumerate([(1, 2), (3, 4), (1, 5), (6, 7)]):
        emitted = []
        for session in (default, explicit):
            operation = session.prime if index < 2 else session.observe
            emitted.append(operation(torch.tensor([key]), torch.tensor([value])))
        assert emitted[0] == emitted[1] == [[dict(source=index, key=key, value=value,
                                                 kind="online", after_online=index)]]
    golden = json.loads(LEGACY_STATE_JSON)
    assert json.dumps(default.policies[0].export_state(), separators=(",", ":")) == LEGACY_STATE_JSON
    assert StreamingReplay.from_state(golden).export_state() == golden
    assert equal(default.export_state(), explicit.export_state())
    assert default.export_state()["schema_version"] == 1
    assert "freshness" not in default.export_state() and "freshness" not in default.budget()
    restored = transport(default)
    for key, value in [(8, 9), (10, 11)]:
        events = [s.observe(torch.tensor([key]), torch.tensor([value])) for s in (default, explicit, restored)]
        assert events[0] == events[1] == events[2]
    assert events[0][0][1:] == [dict(source=source, key=key, value=value, kind="extra", after_online=5)
                                for source, key, value in golden["snapshot"]]
    for session in (default, explicit, restored):
        assert session.finish() == [[]]
        assert session.budget()["total_writes"] == 10
        assert session.read(torch.tensor([[1, 3, 6, 8, 10]])).argmax(-1).tolist() == [[2, 4, 7, 9, 11]]
        assert equal(default.export_state(), session.export_state())


@pytest.mark.parametrize("arm", ["online", "repeat"])
def test_bounded_freshness_is_rejected_for_nonbuffered_arms(arm):
    with pytest.raises(ValueError, match="freshness"):
        StreamingReplay(arm, 0, freshness="bounded")
    with pytest.raises(ValueError, match="freshness"):
        SleepSession(DeltaAdapter(), 1, arm, [0], freshness="bounded")


@pytest.mark.parametrize("freshness", [None, True, 1, "latest", "Bounded"])
def test_unknown_freshness_is_rejected(freshness):
    with pytest.raises(ValueError, match="freshness"):
        StreamingReplay("continuous", 0, freshness=freshness)


@pytest.mark.parametrize("corruption", ["version", "legacy_version", "missing_freshness", "literal_v2",
    "bad_freshness", "nonbuffered", "fingerprint", "history", "future_sample", "mutated_sample", "rng"])
def test_bounded_policy_rejects_malformed_state(corruption):
    policy = StreamingReplay("continuous", 14, freshness="bounded")
    policy.prime(0, 1)
    policy.step(0, 2)
    payload = deepcopy(policy.export_state())
    if corruption == "version": payload["schema_version"] = True
    elif corruption == "legacy_version": payload["schema_version"] = 1
    elif corruption == "missing_freshness": del payload["config"]["freshness"]
    elif corruption == "literal_v2": payload["config"]["freshness"] = "literal"
    elif corruption == "bad_freshness": payload["config"]["freshness"] = True
    elif corruption == "nonbuffered": payload["config"]["arm"] = "repeat"
    elif corruption == "fingerprint": payload["config_fingerprint"] = "wrong"
    elif corruption == "history": payload["history"] = []
    elif corruption == "future_sample": payload["snapshot"][0] = [1, 0, 2]
    elif corruption == "mutated_sample": payload["snapshot"][0][2] = 3
    elif corruption == "rng": payload["rng"]["state"]["inc"] ^= 2
    if corruption != "fingerprint":
        payload["config_fingerprint"] = StreamingReplay._fingerprint(payload["config"])
    with pytest.raises(ValueError):
        StreamingReplay.from_state(payload)


@pytest.mark.parametrize("corruption", ["version", "legacy_version", "missing_freshness", "literal_v2",
    "policy_literal", "budget", "capacity", "block_size", "arm", "history"])
def test_bounded_session_rejects_mismatched_config_or_budget(corruption):
    model = DeltaAdapter()
    session = SleepSession(model, 1, "periodic", [14], freshness="bounded")
    session.prime(torch.tensor([0]), torch.tensor([1]))
    session.observe(torch.tensor([0]), torch.tensor([2]))
    before = session.export_state()
    payload = deepcopy(before)
    if corruption == "version": payload["schema_version"] = True
    elif corruption == "legacy_version": payload["schema_version"] = 1
    elif corruption == "missing_freshness": del payload["freshness"]
    elif corruption == "literal_v2": payload["freshness"] = "literal"
    elif corruption == "policy_literal":
        policy = payload["policies"][0]
        policy["schema_version"] = 1
        del policy["config"]["freshness"]
        policy["config_fingerprint"] = StreamingReplay._fingerprint(policy["config"])
    elif corruption == "budget": payload["replay_count"] = 1
    elif corruption == "capacity": payload["capacity"] = 7
    elif corruption == "block_size": payload["block_size"] = 3
    elif corruption == "arm": payload["arm"] = "continuous"
    elif corruption == "history": payload["history"] = []
    with pytest.raises(ValueError):
        SleepSession.from_state(model, payload)
    assert equal(session.export_state(), before)


@pytest.mark.parametrize("operation", ["observe", "finish"])
def test_bounded_backend_failure_rolls_back_model_policy_rng_and_revision(monkeypatch, operation):
    model = DeltaAdapter()
    session = SleepSession(model, 2, "periodic", [14, 15], freshness="bounded")
    session.prime(torch.tensor([0, 1]), torch.tensor([1, 2]))
    for value in range(3 if operation == "observe" else 2):
        session.observe(torch.tensor([0, 1]), torch.tensor([value + 3, value + 4]))
    before = session.export_state()
    original = model.write
    calls = 0

    def fail_second(keys, values, state):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("Injected bounded replay failure")
        return original(keys, values, state)

    monkeypatch.setattr(model, "write", fail_second)
    with pytest.raises(ValueError, match="Injected"):
        if operation == "observe":
            session.observe(torch.tensor([0, 1]), torch.tensor([9, 10]))
        else:
            session.finish()
    assert equal(before, session.export_state())
    monkeypatch.setattr(model, "write", original)
    reference = transport(session)
    if operation == "observe":
        assert session.observe(torch.tensor([0, 1]), torch.tensor([9, 10])) == reference.observe(
            torch.tensor([0, 1]), torch.tensor([9, 10]))
    else:
        assert session.finish() == reference.finish()
    assert equal(session.export_state(), reference.export_state())


def test_invalid_bounded_arrivals_leave_fifo_snapshot_and_rng_unchanged():
    policy = StreamingReplay("continuous", 14, freshness="bounded")
    policy.prime(0, 1)
    before = policy.export_state()
    for key, value in [(True, 1), (0, 16), (64, 0), (0, 1.0)]:
        with pytest.raises(ValueError):
            policy.step(key, value)
        assert policy.export_state() == before
