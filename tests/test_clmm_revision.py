"""CPU mechanism and persistence checks for the protected revision factorial."""
from copy import deepcopy
from dataclasses import replace
import io

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from clmm.hebbian.adapter import AssociativeAdapter, AssociativeConfig
from clmm.hebbian.revision import REVISION_RULES, RevisionAdapter, RevisionConfig
from clmm.hebbian.routing_decay import RoutingDecayAdapter, routing_table_sha256
from clmm.pretrain.kv_control import physical_state_bytes
from clmm.pretrain.kv_correlated import CorrelatedConfig, _key_codes


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def parent(**changes):
    config = AssociativeConfig(CorrelatedConfig().to_dict(), **changes)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(90511)
        model = AssociativeAdapter(config, _key_codes(64, 64, 32, 90501), torch.eye(64)[:16])
    with torch.no_grad():
        model.W0.add_(0.013)
        model.alpha.mul_(0.9)
        model.raw_eta.add_(0.02)
    return model


def adapter(rule="local_delta", policy="content", seed=None, **parent_changes):
    return RevisionAdapter.from_parent(parent(**parent_changes), revision_rule=rule,
                                       policy=policy, random_seed=seed)


def tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(tree_equal(left[key], right[key]) for key in left))
    return left == right


def encoded(model, key, value, module):
    # Separate single-module matmuls are an oracle for the batched implementation.
    x = (model.codebook.keys[key] @ model.key_encoder[module]).tanh()
    y = (model.codebook.values[value] @ model.value_encoder[module]).tanh()
    return x, y


@pytest.mark.parametrize("policy,seed", [("content", None), ("random", 90701)])
def test_ema_is_bit_identical_to_legacy_all_keys_over_multiple_writes(policy, seed):
    dense = parent()
    legacy = RoutingDecayAdapter.from_parent(dense, policy=policy, random_seed=seed,
                                            inactive_decay="protected")
    model = RevisionAdapter.from_parent(dense, revision_rule="ema", policy=policy,
                                       random_seed=seed)
    assert torch.equal(model.route_table, legacy.route_table)
    keys = torch.arange(64)
    left, right = model.initial_state(64), legacy.initial_state(64)
    with torch.no_grad():
        for step in range(6):
            writes, values = (keys + step * 7) % 64, (keys + step * 3) % 16
            left, right = model.write(writes, values, left), legacy.write(writes, values, right)
            assert torch.equal(left.weights["trace"], right.weights["trace"])
            queries = keys[None].expand(64, -1)
            assert torch.equal(model.read(queries, left), legacy.read(queries, right))
    assert left.write_count == right.write_count == 6


@pytest.mark.parametrize("rule", REVISION_RULES)
@pytest.mark.parametrize("policy,seed", [("content", None), ("random", 90702)])
def test_full_rule_matches_independent_single_module_formula(rule, policy, seed):
    model = adapter(rule, policy, seed, trace_bound=2.0)
    keys, values = torch.tensor([0, 16, 31, 63]), torch.tensor([2, 5, 7, 9])
    trace = torch.linspace(-0.8, 0.8, 4 * 4 * 64 * 64).reshape(4, 4, 64, 64)
    snapshot = trace.clone()
    expected = trace.clone()
    with torch.no_grad():
        for batch, (key, value) in enumerate(zip(keys, values)):
            module = int(model.route_table[key])
            x, y = encoded(model, key, value, module)
            retained = trace[batch, module]
            if rule in ("erase_ema", "local_delta"):
                retained = retained - torch.outer(x / max(x @ x, model.config.projection_eps),
                                                  x @ retained)
            if rule in ("ema", "erase_ema"):
                retained = (1 - model.eta) * retained
            expected[batch, module] = (retained + model.eta * torch.outer(x, y)).clamp(-2, 2)
        actual = model.write(keys, values, model._state(trace, 7))
    torch.testing.assert_close(actual.weights["trace"], expected, rtol=2e-6, atol=3e-7)
    active = model.route(keys)
    assert torch.equal(actual.weights["trace"][~active], trace[~active])
    assert torch.equal(trace, snapshot)
    assert actual.write_count == 8
    assert torch.equal(model.decay_mask(keys), active if model.config.active_bulk_decay else torch.zeros_like(active))
    assert torch.equal(model.projection_mask(keys), active if model.config.key_local_erasure else torch.zeros_like(active))


@pytest.mark.parametrize("rule", REVISION_RULES)
def test_zero_trace_first_write_exactly_matches_eta_outer_product(rule):
    model = adapter(rule)
    keys, values = torch.arange(64), torch.arange(64) % 16
    pre = torch.tanh(torch.einsum("bd,mdh->bmh", model.codebook.encode_keys(keys), model.key_encoder))
    post = torch.tanh(torch.einsum("bd,mdh->bmh", model.codebook.encode_values(values), model.value_encoder))
    expected = model.eta * (pre[..., :, None] * post[..., None, :])
    active = model.route(keys)
    with torch.no_grad():
        result = model.write(keys, values, model.initial_state(64))
    assert torch.equal(result.weights["trace"][active], expected[active])
    assert torch.count_nonzero(result.weights["trace"][~active]) == 0


@pytest.mark.parametrize("rule", ["erase_ema", "local_delta"])
def test_erasure_overwrites_only_linear_trace_response_without_alpha_W0_correction(rule):
    model = adapter(rule, trace_bound=100)
    key, value = torch.tensor([0]), torch.tensor([5])
    module = int(model.route_table[0])
    x, y = encoded(model, 0, 5, module)
    trace = torch.linspace(-0.1, 0.1, 4 * 64 * 64).reshape(1, 4, 64, 64)
    with torch.no_grad():
        result = model.write(key, value, model._state(trace, 2))
        response = x @ result.weights["trace"][0, module]
        torch.testing.assert_close(response, model.eta * (x @ x) * y, rtol=3e-6, atol=1e-7)
        # An independent previous response cannot affect the revised x^T H.
        changed = trace.clone()
        changed[0, module] += torch.outer(x, torch.linspace(-0.2, 0.2, 64))
        second = model.write(key, value, model._state(changed, 2))
        torch.testing.assert_close(x @ second.weights["trace"][0, module], response, rtol=3e-6, atol=1e-7)
        # Arbitrary outer parameters do affect reads, but never the H update.
        model.alpha.copy_(torch.linspace(0.3, 1.7, model.alpha.numel()).reshape_as(model.alpha))
        model.W0.fill_(0.5)
        third = model.write(key, value, model._state(trace, 2))
        assert torch.equal(third.weights["trace"], result.weights["trace"])
        effective_response = x @ (model.W0[module] + model.alpha[module] * result.weights["trace"][0, module])
        assert not torch.allclose(effective_response, response)


@pytest.mark.parametrize("rule", REVISION_RULES)
def test_orthogonal_key_direction_changes_only_under_active_bulk_decay(rule):
    model = adapter(rule, trace_bound=100)
    module = int(model.route_table[0])
    x, _ = encoded(model, 0, 2, module)
    direction = torch.linspace(-0.2, 0.2, 64)
    direction -= x * (direction @ x) / (x @ x)
    assert abs(float((direction @ x).detach())) < 2e-7
    trace = torch.linspace(-0.15, 0.15, 4 * 64 * 64).reshape(1, 4, 64, 64)
    with torch.no_grad():
        updated = model.write(torch.tensor([0]), torch.tensor([2]), model._state(trace, 1))
        before = direction @ trace[0, module]
        after = direction @ updated.weights["trace"][0, module]
        expected = (1 - model.eta) * before if model.config.active_bulk_decay else before
    torch.testing.assert_close(after, expected, rtol=3e-6, atol=2e-7)


@pytest.mark.parametrize("rule", REVISION_RULES)
def test_other_module_stable_role_is_bit_identical_across_revision_writes(rule):
    model = adapter(rule)
    key = torch.tensor([0])
    active_module = int(model.route_table[0])
    other_key = int(torch.where(model.route_table != active_module)[0][0])
    trace = torch.linspace(-0.4, 0.4, 4 * 64 * 64).reshape(1, 4, 64, 64)
    initial = model._state(trace.clone(), 5)
    query = torch.tensor([[other_key]])
    with torch.no_grad():
        scores = model.read(query, initial)
        state = initial
        for value in (1, 7, 2, 15):
            state = model.write(key, torch.tensor([value]), state)
            assert torch.equal(state.weights["trace"][:, torch.arange(4) != active_module],
                               initial.weights["trace"][:, torch.arange(4) != active_module])
            assert torch.equal(model.read(query, state), scores)


@pytest.mark.parametrize("rule", ["erase_ema", "local_delta"])
@pytest.mark.parametrize("eps", [1e-8, 1e-300, 1e300])
def test_zero_norm_projection_is_finite_and_has_no_erasure(rule, eps):
    dense = parent()
    with torch.no_grad():
        dense.key_encoder.zero_()
    model = RevisionAdapter.from_parent(dense, revision_rule=rule, policy="content")
    model = RevisionAdapter(replace(model.config, projection_eps=eps), model.codebook.keys,
                            model.codebook.values, route_table=model.route_table)
    model.load_state_dict({**dense.state_dict(), "route_table": model.route_table})
    trace = torch.linspace(-0.8, 0.8, 4 * 64 * 64).reshape(1, 4, 64, 64)
    key = torch.tensor([0])
    active = model.route(key)
    with torch.no_grad():
        result = model.write(key, torch.tensor([2]), model._state(trace, 4))
        expected = (1 - model.eta) * trace if model.config.active_bulk_decay else trace
    assert torch.isfinite(result.weights["trace"]).all()
    assert torch.equal(result.weights["trace"][active], expected[active])
    assert torch.equal(result.weights["trace"][~active], trace[~active])


@pytest.mark.parametrize("rule", REVISION_RULES)
def test_active_clamp_is_applied_after_revision_and_inactive_values_are_untouched(rule):
    model = adapter(rule, trace_bound=0.001)
    keys = torch.tensor([0, 16])
    trace = torch.full((2, 4, 64, 64), 0.0008)
    with torch.no_grad():
        result = model.write(keys, torch.tensor([1, 2]), model._state(trace, 1))
    active = model.route(keys)
    assert result.weights["trace"][active].abs().max() == model.config.trace_bound
    assert torch.equal(result.weights["trace"][~active], trace[~active])
    model.export_state(result)


@pytest.mark.parametrize("rule", REVISION_RULES)
def test_online_is_label_local_rng_pure_and_has_no_autograd_engine_calls(rule, monkeypatch):
    model = adapter(rule)
    keys, values = torch.tensor([0, 16]), torch.tensor([2, 5])
    slow = deepcopy(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    initial = model.initial_state(2)
    with monkeypatch.context() as context:
        def forbidden(*args, **kwargs):
            raise AssertionError("Online update invoked the autograd engine")
        context.setattr(torch.autograd, "grad", forbidden)
        context.setattr(torch.autograd, "backward", forbidden)
        context.setattr(torch.Tensor, "backward", forbidden)
        with torch.no_grad():
            prefix_a = model.write(keys, values, initial)
            prefix_b = model.write(keys, values, initial)
            model.write(keys, torch.tensor([3, 7]), prefix_a)
            model.write(keys, torch.tensor([9, 4]), prefix_b)
            assert tree_equal(model.export_state(prefix_a), model.export_state(prefix_b))
            assert torch.isfinite(model.read(keys[:, None], prefix_a)).all()
    assert tree_equal(slow, model.state_dict())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("rule", REVISION_RULES)
def test_offline_gradients_flow_through_all_trainable_outer_parameters(rule):
    model = adapter(rule)
    keys = torch.tensor([int(torch.where(model.route_table == module)[0][0]) for module in range(4)])
    values = torch.tensor([2, 5, 7, 9])
    state = model.initial_state(4)
    for step in range(3):
        state = model.write(keys, values + step, state)
    scores = model.read(keys[:, None], state)
    F.mse_loss(scores, F.one_hot((values + 2)[:, None], 16).float()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert not model.detach_state(state).weights["trace"].requires_grad


def test_forks_preserve_parent_rng_slow_bits_tables_flags_and_storage():
    dense = parent()
    dense.eval().requires_grad_(False)
    dense.raw_eta.requires_grad_(True)
    original = deepcopy(dense.state_dict())
    rng, numpy_rng = torch.random.get_rng_state().clone(), np.random.get_state()
    fingerprints, tables = set(), {}
    for policy in ("content", "random"):
        for rule in REVISION_RULES:
            model = RevisionAdapter.from_parent(dense, revision_rule=rule, policy=policy,
                                               random_seed=90701 if policy == "random" else None)
            assert torch.equal(rng, torch.random.get_rng_state())
            assert tree_equal(original, dense.state_dict())
            assert not model.training
            assert model.parameter_counts() == dense.parameter_counts()
            for name, parameter in model.named_parameters():
                source = dict(dense.named_parameters())[name]
                assert torch.equal(parameter, source), name
                assert parameter.data_ptr() != source.data_ptr(), name
                assert parameter.requires_grad == source.requires_grad
            assert set(model.state_dict()) == set(original) | {"route_table"}
            for name, tensor in original.items():
                assert torch.equal(model.state_dict()[name], tensor), name
            state = model.initial_state(1)
            assert physical_state_bytes(state) == 65536
            assert state.write_count == 0 and state.aux is None and set(state.weights) == {"trace"}
            assert torch.count_nonzero(state.weights["trace"]) == 0
            assert model.route_table.numel() * model.route_table.element_size() == 512
            assert routing_table_sha256(model.route_table) == model.config.routing_table_sha256
            assert RevisionConfig(**model.config.to_dict()).fingerprint == model.config.fingerprint
            fingerprints.add(model.config.fingerprint)
            tables[policy, rule] = model.route_table
    assert len(fingerprints) == 8
    for policy in ("content", "random"):
        assert all(torch.equal(tables[policy, "ema"], tables[policy, rule]) for rule in REVISION_RULES)
    expected = np.random.Generator(np.random.PCG64(90701)).permutation(tables["content", "ema"].numpy())
    assert torch.equal(tables["random", "ema"], torch.from_numpy(expected))
    current = np.random.get_state()
    assert numpy_rng[0] == current[0] and np.array_equal(numpy_rng[1], current[1]) and numpy_rng[2:] == current[2:]


@pytest.mark.parametrize("rule", REVISION_RULES)
@pytest.mark.parametrize("policy,seed", [("content", None), ("random", 90701)])
def test_checkpoint_roundtrip_pure_reads_and_exact_continuation(rule, policy, seed):
    model = adapter(rule, policy, seed)
    with torch.no_grad():
        state = model.initial_state(2)
        for key, value in (([0, 16], [2, 5]), ([31, 63], [8, 9]), ([0, 16], [7, 3])):
            state = model.write(torch.tensor(key), torch.tensor(value), state)
        saved_fast, saved_slow = model.export_state(state), deepcopy(model.state_dict())
        assert saved_fast["schema_version"] == 3
        rng = torch.random.get_rng_state().clone()
        query = torch.tensor([[0, 16, 31], [16, 0, 63]])
        scores = model.read(query, state)
        for _ in range(3):
            assert torch.equal(model.read(query, state), scores)
        assert tree_equal(saved_fast, model.export_state(state))
        assert tree_equal(saved_slow, model.state_dict())
        assert torch.equal(rng, torch.random.get_rng_state())
        stream = io.BytesIO()
        torch.save(dict(config=model.config.to_dict(), model=saved_slow, fast=saved_fast), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = RevisionAdapter(RevisionConfig(**saved["config"]), saved["model"]["codebook.keys"],
                                   saved["model"]["codebook.values"], route_table=saved["model"]["route_table"])
        restored.load_state_dict(saved["model"], strict=True)
        resumed = restored.import_state(saved["fast"])
        assert torch.equal(restored.read(query, resumed), scores)
        for key, value in (([16, 0], [7, 8]), ([31, 63], [3, 2])):
            state = model.write(torch.tensor(key), torch.tensor(value), state)
            resumed = restored.write(torch.tensor(key), torch.tensor(value), resumed)
            assert tree_equal(model.export_state(state), restored.export_state(resumed))
        assert resumed.write_count == 5
        resumed.weights["trace"].zero_()
        assert not torch.equal(resumed.weights["trace"], saved_fast["weights"]["trace"])


def test_cross_rule_policy_seed_epsilon_and_legacy_fast_states_are_rejected():
    model = adapter("ema", "random", 90701)
    payload = model.export_state(model.initial_state(2))
    others = [adapter(rule, "random", 90701) for rule in REVISION_RULES if rule != "ema"]
    others += [adapter("ema"), adapter("ema", "random", 90702)]
    others += [RevisionAdapter(replace(model.config, projection_eps=1e-6), model.codebook.keys,
                              model.codebook.values, route_table=model.route_table)]
    for other in others:
        assert other.config.fingerprint != model.config.fingerprint
        with pytest.raises(ValueError):
            other.import_state(payload)
        with pytest.raises(ValueError):
            other.read(torch.tensor([[0], [1]]), model.initial_state(2))
    dense = parent()
    legacy = RoutingDecayAdapter.from_parent(dense, policy="random", random_seed=90701,
                                            inactive_decay="protected")
    for old in (legacy.export_state(legacy.initial_state(2)), dense.export_state(dense.initial_state(2))):
        with pytest.raises(ValueError):
            model.import_state(old)
    for mutate in (
        lambda p: p.update(schema_version=2), lambda p: p.update(schema_version=True),
        lambda p: p.update(history=[]), lambda p: p.update(batch_size=True),
        lambda p: p.update(write_count=-1), lambda p: p.update(aux={}),
        lambda p: p["config"].update(revision_rule="local_delta"),
        lambda p: p["config"].update(projection_eps=0),
        lambda p: p["weights"].update(trace=torch.full((2, 4, 64, 64), float("nan"))),
        lambda p: p["weights"].update(trace=torch.full((2, 4, 64, 64), 1.1)),
    ):
        invalid = deepcopy(payload)
        mutate(invalid)
        with pytest.raises(ValueError):
            model.import_state(invalid)


def test_checkpoint_table_and_slow_tensor_validation_remains_strict_and_atomic():
    model = adapter("local_delta", "random", 90701)
    original = deepcopy(model.state_dict())
    for mutate in (
        lambda p: p.pop("route_table"), lambda p: p.update(extra=torch.zeros(1)),
        lambda p: p.update(route_table=p["route_table"].int()),
        lambda p: p.update(route_table=(p["route_table"] + 1) % 4),
        lambda p: p.update(alpha=p["alpha"].double()), lambda p: p["alpha"].fill_(float("nan")),
        lambda p: p["codebook.keys"].zero_(), lambda p: p["codebook.values"].zero_(),
        lambda p: p["router_prototypes"].zero_(),
    ):
        invalid = deepcopy(original)
        mutate(invalid)
        with pytest.raises(ValueError):
            model.load_state_dict(invalid)
        assert tree_equal(original, model.state_dict())
    with pytest.raises(ValueError):
        model.load_state_dict(original, strict=False)
    with pytest.raises(ValueError):
        adapter("local_delta", "random", 90702).load_state_dict(original)
    with pytest.raises(ValueError):
        RevisionAdapter(replace(model.config, routing_table_sha256="0" * 64), model.codebook.keys,
                        model.codebook.values, route_table=model.route_table)
    model.load_state_dict(original, assign=True)
    model.assert_codebook_valid()
    with torch.no_grad():
        model.route_table[0] = (model.route_table[0] + 1) % 4
    with pytest.raises(ValueError):
        model.export_state(model.initial_state(1))


@pytest.mark.parametrize("changes", [
    {"revision_rule": "gradient"}, {"revision_rule": None}, {"update_rule": "gradient"},
    {"inactive_decay": "global"}, {"projection_eps": True}, {"projection_eps": 0},
    {"projection_eps": -1}, {"projection_eps": float("nan")}, {"projection_eps": float("inf")},
    {"projection_eps": "1e-8"}, {"policy": "label"}, {"random_seed": 90701},
    {"policy": "random"}, {"policy": "random", "random_seed": True},
    {"modules": 2}, {"dim": 32}, {"trace_bound": 0},
])
def test_invalid_revision_configs(changes):
    with pytest.raises(ValueError):
        RevisionConfig(CorrelatedConfig().to_dict(), **changes)


def test_rule_flags_and_input_and_parent_boundaries():
    flags = {"ema": (True, False), "additive": (False, False),
             "erase_ema": (True, True), "local_delta": (False, True)}
    for rule, expected in flags.items():
        config = RevisionConfig(CorrelatedConfig().to_dict(), revision_rule=rule)
        assert (config.active_bulk_decay, config.key_local_erasure) == expected
    model = adapter()
    state = model.initial_state(1)
    for key in (torch.tensor([64]), torch.tensor([-1]), torch.tensor([1.0])):
        for method in (model.routing_mask, model.decay_mask, model.projection_mask):
            with pytest.raises(ValueError):
                method(key)
    for keys, values in ((torch.tensor([[0]]), torch.tensor([1])),
                         (torch.tensor([0]), torch.tensor([16]))):
        with pytest.raises(ValueError):
            model.write(keys, values, state)
    for wrong in (object(), model, parent(routing="top1"), parent(update_rule="gradient")):
        with pytest.raises(ValueError):
            RevisionAdapter.from_parent(wrong, revision_rule="local_delta", policy="content")
