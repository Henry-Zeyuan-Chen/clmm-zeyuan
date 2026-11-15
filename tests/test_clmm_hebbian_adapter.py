"""Mechanism and persistent-state checks for reference component adapters."""
from copy import deepcopy
from itertools import product
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from clmm.hebbian import AssociativeAdapter, AssociativeConfig
from clmm.pretrain.kv_control import KVState, physical_state_bytes
from clmm.pretrain.kv_correlated import CorrelatedConfig, _key_codes


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def adapter(rule="hebbian", routing="dense", **changes):
    cfg = CorrelatedConfig()
    config = AssociativeConfig(cfg.to_dict(), routing=routing, update_rule=rule, **changes)
    return AssociativeAdapter(config, _key_codes(64, 64, 32, 90501), torch.eye(64)[:16])


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    return a == b


def test_initialization_and_parameter_names_are_matched_across_all_variants():
    models, rngs = [], []
    for rule, routing in product(("gradient", "hebbian"), ("dense", "top1")):
        torch.manual_seed(90511)
        models.append(adapter(rule, routing))
        rngs.append(torch.random.get_rng_state())
    reference = models[0].state_dict()
    assert all(tree_equal(reference, model.state_dict()) for model in models)
    assert all(torch.equal(rngs[0], rng) for rng in rngs)
    assert all(model.parameter_counts() == dict(registered=81921, trainable=81921) for model in models)
    assert len({model.config.fingerprint for model in models}) == 4
    for model in models:
        assert physical_state_bytes(model.initial_state(2)) == 2 * 65536
        assert model.config.nvalues == 16 and model.config.nkeys == 64


@pytest.mark.parametrize("routing", ["dense", "top1"])
def test_hebbian_ema_formula_and_inactive_modules_are_exact(routing):
    model = adapter(routing=routing)
    keys, values = torch.tensor([0, 16]), torch.tensor([2, 5])
    state = model.initial_state(2)
    trace = torch.randn_like(state.weights["trace"]) * 0.02
    state = KVState({"trace": trace}, None, 7, model.method, model.config.fingerprint)
    raw_keys, raw_values = model.codebook.encode_keys(keys), model.codebook.encode_values(values)
    expected = trace.clone()
    active = model.routing_mask(keys)
    for b in range(2):
        for m in range(4):
            if active[b, m]:
                pre = torch.tanh(raw_keys[b] @ model.key_encoder[m])
                post = torch.tanh(raw_values[b] @ model.value_encoder[m])
                expected[b, m] = ((1 - model.eta) * trace[b, m] + model.eta * torch.outer(pre, post)).clamp(-1, 1)
    actual = model.write(keys, values, state)
    torch.testing.assert_close(actual.weights["trace"], expected, rtol=1e-6, atol=1e-8)
    assert torch.equal(actual.weights["trace"][~active], trace[~active])
    assert actual.write_count == 8
    assert torch.equal(state.weights["trace"], trace)
    if routing == "top1":
        assert active.tolist() == [[True, False, False, False], [False, True, False, False]]


@pytest.mark.parametrize("routing", ["dense", "top1"])
def test_analytic_gradient_matches_independent_current_state_autograd(routing):
    model = adapter("gradient", routing)
    keys, values = torch.tensor([0, 16]), torch.tensor([2, 5])
    start = model.initial_state(2)
    trace = (torch.randn_like(start.weights["trace"]) * 0.03).requires_grad_()
    state = KVState({"trace": trace}, None, 3, model.method, model.config.fingerprint)
    loss = torch.zeros(())
    for b in range(2):
        for m in range(4):
            pre = torch.tanh(model.codebook.keys[keys[b]] @ model.key_encoder[m])
            post = torch.tanh(model.codebook.values[values[b]] @ model.value_encoder[m])
            prediction = pre @ (model.W0[m] + model.alpha[m] * trace[b, m])
            loss = loss + 0.5 * (prediction - post).square().sum()
    gradient, = torch.autograd.grad(loss, trace)
    expected = ((1 - model.eta) * trace - model.eta * gradient).clamp(-1, 1)
    active = model.routing_mask(keys)[..., None, None]
    expected = torch.where(active, expected, trace)
    actual = model.write(keys, values, state)
    torch.testing.assert_close(actual.weights["trace"], expected, rtol=1e-6, atol=1e-8)
    assert torch.equal(actual.weights["trace"][~active.expand_as(trace)], trace[~active.expand_as(trace)])


@pytest.mark.parametrize("rule,routing", list(product(("gradient", "hebbian"), ("dense", "top1"))))
def test_online_needs_no_autograd_engine_and_offline_gradients_flow(rule, routing, monkeypatch):
    model = adapter(rule, routing)
    state = model.initial_state(2)
    with monkeypatch.context() as context:
        def forbidden(*args, **kwargs):
            raise AssertionError("Online update called autograd engine")
        context.setattr(torch.autograd, "grad", forbidden)
        context.setattr(torch.Tensor, "backward", forbidden)
        with torch.no_grad():
            written = model.write(torch.tensor([0, 16]), torch.tensor([2, 5]), state)
            assert torch.isfinite(model.read(torch.tensor([[0], [16]]), written)).all()
    state = model.initial_state(4)
    for offset in range(3):
        state = model.write(torch.tensor([0, 16, 32, 48]), torch.tensor([2, 5, 7, 9]) + offset, state)
    scores = model.read(torch.tensor([[0], [16], [32], [48]]), state)
    loss = F.mse_loss(scores, F.one_hot(torch.tensor([[4], [7], [9], [11]]), 16).float())
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("rule,routing", list(product(("gradient", "hebbian"), ("dense", "top1"))))
def test_portable_state_and_slow_weights_resume_exactly_with_pure_reads(rule, routing):
    model = adapter(rule, routing)
    state = model.write(torch.tensor([0, 16]), torch.tensor([2, 5]), model.initial_state(2))
    snapshot, slow = model.export_state(state), deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    query = torch.tensor([[0, 16], [16, 0]])
    scores = model.read(query, state)
    assert tree_equal(snapshot, model.export_state(state))
    assert tree_equal(slow, model.state_dict()) and torch.equal(rng, torch.random.get_rng_state())
    raw = io.BytesIO()
    torch.save(dict(config=model.config.to_dict(), model=slow, fast=snapshot), raw)
    raw.seek(0)
    saved = torch.load(raw, weights_only=True)
    restored = AssociativeAdapter(AssociativeConfig(**saved["config"]), saved["model"]["codebook.keys"], saved["model"]["codebook.values"])
    restored.load_state_dict(saved["model"])
    imported = restored.import_state(saved["fast"])
    assert torch.equal(scores, restored.read(query, imported))
    a = model.write(torch.tensor([16, 0]), torch.tensor([7, 8]), state)
    b = restored.write(torch.tensor([16, 0]), torch.tensor([7, 8]), imported)
    assert tree_equal(model.export_state(a), restored.export_state(b))
    assert not restored.detach_state(b).weights["trace"].requires_grad
    changed = snapshot["weights"]["trace"]
    changed.zero_()
    assert not torch.equal(changed, state.weights["trace"])


def test_fingerprints_reject_other_update_routing_codebooks_and_bad_payloads():
    source = adapter()
    payload = source.export_state(source.initial_state(2))
    for other in (adapter("gradient"), adapter(routing="top1")):
        with pytest.raises(ValueError):
            other.import_state(payload)
    for mutate in (lambda p: p.update(history=[]), lambda p: p.update(write_count=-1),
                   lambda p: p.update(batch_size=True), lambda p: p.update(aux={}),
                   lambda p: p["weights"].update(trace=torch.full((2, 4, 64, 64), 2.0)),
                   lambda p: p["weights"].update(trace=torch.full((2, 4, 64, 64), float("nan"))),
                   lambda p: p["config"].update(trace_bound=True)):
        bad = deepcopy(payload)
        mutate(bad)
        with pytest.raises(ValueError):
            source.import_state(bad)
    for name in ("alpha", "codebook.keys", "codebook.values", "router_prototypes"):
        bad = deepcopy(source.state_dict())
        bad[name][0, 0] = float("nan")
        with pytest.raises(ValueError):
            source.load_state_dict(bad)
    source.assert_codebook_valid()


def test_saturation_bounds_and_input_validation():
    model = adapter(alpha_init=100.0)
    state = model.initial_state(1)
    for _ in range(100):
        state = model.write(torch.tensor([0]), torch.tensor([3]), state)
    assert state.weights["trace"].abs().max() <= 1
    for key, value in ((64, 0), (-1, 0), (0, 16)):
        with pytest.raises(ValueError):
            model.write(torch.tensor([key]), torch.tensor([value]), state)
    with pytest.raises(ValueError):
        model.read(torch.tensor([[1.0]]), state)
    with pytest.raises(ValueError):
        model.write(torch.tensor([[0]]), torch.tensor([0]), state)


@pytest.mark.parametrize("change", [{"modules": True}, {"modules": 2}, {"dim": 32}, {"routing": "label"},
    {"update_rule": "oja"}, {"eta_init": 0}, {"eta_init": 1}, {"trace_bound": 0},
    {"alpha_init": float("nan")}, {"init_noise": -1}, {"trace_bound": True}])
def test_invalid_config(change):
    with pytest.raises(ValueError):
        AssociativeConfig(CorrelatedConfig().to_dict(), **change)
