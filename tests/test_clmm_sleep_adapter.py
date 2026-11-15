"""Reference sleep orchestration: schedule equivalence and complete handoff."""
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path

import pytest
import torch

from clmm.pretrain.kv_control import DeltaAdapter, NeuralAdapter
from clmm.pretrain.kv_streams import episode, replay_plan
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
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    return SleepSession.from_state(session.model, torch.load(buffer, weights_only=True))


def arrivals(session, episodes, *, resume=False):
    events = [[] for _ in episodes]
    for index in range(len(episodes[0]["keys"])):
        keys = torch.tensor([ep["keys"][index] for ep in episodes])
        values = torch.tensor([ep["values"][index] for ep in episodes])
        emitted = (session.prime if index < episodes[0]["boundary"] else session.observe)(keys, values)
        for target, current in zip(events, emitted):
            target.extend(current)
        if resume and index == episodes[0]["boundary"]:
            session = transport(session)
    for target, current in zip(events, session.finish()):
        target.extend(current)
    return session, events


@pytest.mark.parametrize("arm", ["online", "repeat", "continuous", "periodic"])
@pytest.mark.parametrize("delay", [0, 3, 16])
def test_batched_session_matches_frozen_plans_and_portable_continuation(arm, delay):
    episodes = [episode(seed, 8, delay, 2, pressure=True) for seed in (90601, 90603)]
    model = DeltaAdapter()
    session, events = arrivals(SleepSession(model, 2, arm, [ep["seed"] for ep in episodes]), episodes, resume=True)
    assert events == [replay_plan(ep, arm)["events"] for ep in episodes]
    assert session.closed
    expected = 10 + delay * (1 if arm == "online" else 2)
    assert session.state.write_count == expected
    assert session.budget()["total_writes"] == expected
    assert session.budget()["replay_writes"] == (0 if arm == "online" else delay)
    assert all(size <= 8 for size in session.budget()["buffer_pairs"])
    assert session.budget()["snapshot_pairs"] == [0, 0]
    query = torch.tensor([[q["key"] for q in ep["queries"]] for ep in episodes])
    target = torch.tensor([[q["target"] for q in ep["queries"]] for ep in episodes])
    before = session.export_state()
    assert torch.equal(session.read(query).argmax(-1), target)
    assert equal(before, session.export_state())
    assert session.finish() == [[], []]
    with pytest.raises(RuntimeError):
        session.observe(torch.tensor([1, 2]), torch.tensor([3, 4]))


def test_periodic_session_retains_only_bounded_records_and_no_history():
    session = SleepSession(DeltaAdapter(), 1, "periodic", [3])
    session.prime(torch.tensor([0]), torch.tensor([1]))
    for index in range(100):
        session.observe(torch.tensor([index % 64]), torch.tensor([index % 16]))
    payload = session.export_state()
    assert len(payload["policies"][0]["buffer"]) <= 8
    assert len(payload["policies"][0]["snapshot"]) <= 4
    assert "history" not in payload and "events" not in session.__dict__
    assert session.budget()["online_observations"] == 101
    assert session.budget()["total_writes"] == 201


def test_invalid_arrival_and_failed_emitted_write_leave_session_unchanged(monkeypatch):
    model = DeltaAdapter()
    session = SleepSession(model, 2, "periodic", [3, 4])
    session.prime(torch.tensor([0, 1]), torch.tensor([2, 3]))
    for _ in range(3):
        session.observe(torch.tensor([2, 3]), torch.tensor([4, 5]))
    before = session.export_state()
    for keys, values in ((torch.tensor([2, 64]), torch.tensor([1, 2])),
                         (torch.tensor([1., 2.]), torch.tensor([1, 2])),
                         (torch.tensor([1]), torch.tensor([1, 2]))):
        with pytest.raises(ValueError):
            session.observe(keys, values)
        assert equal(before, session.export_state())
    original = model.write
    count = 0
    def fail_after_online(keys, values, state):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("Injected replay write failure")
        return original(keys, values, state)
    monkeypatch.setattr(model, "write", fail_after_online)
    with pytest.raises(ValueError, match="Injected"):
        session.observe(torch.tensor([4, 5]), torch.tensor([6, 7]))
    assert equal(before, session.export_state())


@pytest.mark.parametrize("operation", ["observe", "finish"])
def test_backend_invalid_return_is_rejected_before_committing_session(monkeypatch, operation):
    model = DeltaAdapter()
    session = SleepSession(model, 1, "periodic", [1])
    session.prime(torch.tensor([0]), torch.tensor([1]))
    session.observe(torch.tensor([1]), torch.tensor([2]))
    before = session.export_state()
    original = model.write
    def invalid_identity(keys, values, state):
        return replace(original(keys, values, state), method="wrong-backend")
    monkeypatch.setattr(model, "write", invalid_identity)
    with pytest.raises(ValueError, match="identity"):
        if operation == "observe":
            session.observe(torch.tensor([2]), torch.tensor([3]))
        else:
            session.finish()
    assert equal(before, session.export_state())


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("batch_size", True), ("model_method", "wrong"),
    ("model_config_fingerprint", "wrong"), ("replay_count", 1), ("prime_count", 4),
    ("closed", True), ("episode_seeds", [999]), ("arm", "online"),
    ("capacity", 7), ("block_size", 3), ("delay_count", -1)])
def test_resume_rejects_incoherent_identity_budget_or_lifecycle(field, value):
    model = DeltaAdapter()
    session = SleepSession(model, 1, "periodic", [3])
    session.prime(torch.tensor([0]), torch.tensor([1]))
    session.observe(torch.tensor([1]), torch.tensor([2]))
    payload = deepcopy(session.export_state())
    payload[field] = value
    with pytest.raises((ValueError, TypeError)):
        SleepSession.from_state(model, payload)


def test_session_preserves_outer_gradients_and_does_not_freeze_model():
    model = NeuralAdapter()
    flags = [parameter.requires_grad for parameter in model.parameters()]
    session = SleepSession(model, 1, "continuous", [4])
    session.prime(torch.tensor([0]), torch.tensor([2]))
    session.observe(torch.tensor([1]), torch.tensor([3]))
    session.read(torch.tensor([[0, 1]])).square().mean().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.parameters() if parameter.requires_grad)
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    session.detach_state()
    assert all(not weight.requires_grad for weight in session.state.weights.values())


@pytest.mark.parametrize("field,value", [("capacity", True), ("block_size", True), ("episode_seeds", [True])])
def test_resume_rejects_boolean_counter_aliases(field, value):
    model = DeltaAdapter()
    session = SleepSession(model, 1, "online", [1], capacity=1, block_size=1)
    payload = session.export_state()
    payload[field] = value
    with pytest.raises(ValueError):
        SleepSession.from_state(model, payload)


def test_actual_correlated_checkpoint_session_resume_when_artifact_is_available():
    root = Path(__file__).resolve().parents[1].parent / "2026-10-22/i/outputs/clmm-nonlinear-correlated-2026-11-15"
    checkpoint = root / "r32-90511/checkpoint.pt"
    if not checkpoint.exists():
        pytest.skip("Optional local shared checkpoint is not present")
    from clmm.pretrain.kv_correlated_portable import load_portable_checkpoint
    model, _ = load_portable_checkpoint(checkpoint, torch.device("cpu"))
    model.eval().requires_grad_(False)
    with torch.no_grad():
        session = SleepSession(model, 1, "periodic", [12])
        for key, value in [(0, 2), (1, 3), (0, 4)]:
            session.prime(torch.tensor([key]), torch.tensor([value]))
        session.observe(torch.tensor([2]), torch.tensor([5]))
        restored = transport(session)
        for key, value in [(3, 6), (4, 7), (5, 8)]:
            session.observe(torch.tensor([key]), torch.tensor([value]))
            restored.observe(torch.tensor([key]), torch.tensor([value]))
        assert torch.equal(session.read(torch.tensor([[0, 1]])), restored.read(torch.tensor([[0, 1]])))
        assert equal(session.export_state(), restored.export_state())
