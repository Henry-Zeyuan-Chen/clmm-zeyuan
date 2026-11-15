"""CPU checks for the fixed-slow routing x inactive-decay intervention."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from clmm.hebbian.adapter import AssociativeAdapter, AssociativeConfig
from clmm.hebbian.routing_decay import (
    ROUTING_ALGORITHM, RoutingDecayAdapter, RoutingDecayConfig, routing_table_sha256,
)
from clmm.pretrain.adapter_registry import fork_associative
from clmm.pretrain.kv_control import KVState, physical_state_bytes
from clmm.pretrain.kv_correlated import CorrelatedConfig, _key_codes


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def parent(rule="hebbian", **changes):
    config = AssociativeConfig(CorrelatedConfig().to_dict(), update_rule=rule, **changes)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(90511)
        model = AssociativeAdapter(config, _key_codes(64, 64, 32, 90501), torch.eye(64)[:16])
    # Nontrivial slow values stand in for a loaded trained parent. No optimizer
    # or new training is used in these mechanism checks.
    with torch.no_grad():
        model.W0.add_(0.013)
        model.alpha.mul_(0.9)
        model.raw_eta.add_(0.02)
    return model


def adapter(rule="hebbian", policy="content", decay="protected", seed=None):
    return RoutingDecayAdapter.from_parent(parent(rule), policy=policy,
                                           inactive_decay=decay, random_seed=seed)


def tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(tree_equal(left[key], right[key]) for key in left))
    return left == right


@pytest.mark.parametrize("rule", ["hebbian", "gradient"])
def test_content_protected_is_bit_identical_to_reference_top1_all_keys(rule):
    dense = parent(rule)
    reference = fork_associative(dense, update_rule=rule, routing="top1")
    model = RoutingDecayAdapter.from_parent(dense, policy="content", inactive_decay="protected")
    keys = torch.arange(64)
    assert torch.equal(model.routing_mask(keys), reference.routing_mask(keys))
    assert torch.equal(model.route_table, reference.routing_mask(keys).long().argmax(-1))
    left, right = model.initial_state(64), reference.initial_state(64)
    with torch.no_grad():
        for offset in range(5):
            stream_keys = (keys + 7 * offset) % 64
            values = (keys + 3 * offset) % 16
            left = model.write(stream_keys, values, left)
            right = reference.write(stream_keys, values, right)
            assert torch.equal(left.weights["trace"], right.weights["trace"])
            query = keys[None].expand(64, -1)
            assert torch.equal(model.read(query, left), reference.read(query, right))
    assert left.write_count == right.write_count == 5


def test_forks_preserve_slow_bits_rng_and_all_factors_have_matched_storage():
    dense = parent()
    dense.eval().requires_grad_(False)
    original = deepcopy(dense.state_dict())
    rng = torch.random.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    tables = {}
    for policy in ("content", "random"):
        for decay in ("protected", "global"):
            model = RoutingDecayAdapter.from_parent(
                dense, policy=policy, inactive_decay=decay,
                random_seed=90701 if policy == "random" else None)
            assert tree_equal(original, dense.state_dict())
            assert torch.equal(rng, torch.random.get_rng_state())
            assert not model.training
            assert model.parameter_counts() == dense.parameter_counts()
            for name, parameter in model.named_parameters():
                source = dict(dense.named_parameters())[name]
                assert torch.equal(parameter, source), name
                assert parameter.data_ptr() != source.data_ptr(), name
                assert parameter.requires_grad == source.requires_grad
            for name, tensor in original.items():
                assert torch.equal(model.state_dict()[name], tensor), name
            state = model.initial_state(1)
            assert physical_state_bytes(state) == 65536
            assert set(state.weights) == {"trace"} and state.aux is None and state.write_count == 0
            assert torch.count_nonzero(state.weights["trace"]) == 0
            assert model.route_table.dtype == torch.int64 and model.route_table.shape == (64,)
            assert model.route_table.numel() * model.route_table.element_size() == 512
            assert set(model.state_dict()) == set(original) | {"route_table"}
            tables[policy, decay] = model.route_table
    assert torch.equal(tables["random", "protected"], tables["random", "global"])
    assert torch.equal(tables["content", "protected"], tables["content", "global"])
    current_numpy = np.random.get_state()
    assert numpy_rng[0] == current_numpy[0]
    assert np.array_equal(numpy_rng[1], current_numpy[1])
    assert numpy_rng[2:] == current_numpy[2:]


@pytest.mark.parametrize("seed", range(90701, 90706))
def test_random_routes_are_private_pcg64_permutations_with_exact_cardinalities(seed):
    dense = parent()
    content = RoutingDecayAdapter.from_parent(dense, policy="content", inactive_decay="protected")
    rng = torch.random.get_rng_state().clone()
    random = RoutingDecayAdapter.from_parent(dense, policy="random", random_seed=seed,
                                             inactive_decay="protected")
    repeat = RoutingDecayAdapter.from_parent(dense, policy="random", random_seed=seed,
                                             inactive_decay="global")
    expected = np.random.Generator(np.random.PCG64(seed)).permutation(content.route_table.numpy())
    assert torch.equal(random.route_table, torch.from_numpy(expected))
    assert torch.equal(random.route_table, repeat.route_table)
    assert torch.equal(torch.bincount(random.route_table, minlength=4),
                       torch.bincount(content.route_table, minlength=4))
    assert not torch.equal(random.route_table, content.route_table)
    repeated_keys = torch.tensor([3, 18, 3, 63, 18, 3])
    routing = random.routing_mask(repeated_keys)
    assert routing.sum(-1).tolist() == [1] * 6
    assert torch.equal(routing[0], routing[2]) and torch.equal(routing[2], routing[5])
    assert torch.equal(routing[1], routing[4])
    with torch.no_grad():
        state = random.initial_state(6)
        for offset in range(3):
            state = random.write(repeated_keys, torch.full((6,), offset), state)
            assert torch.equal(routing, random.routing_mask(repeated_keys))
    # With zero slow W0, an off-route trace has no path into a read.
    with torch.no_grad():
        random.W0.zero_()
        trace = state.weights["trace"].clone()
        trace[~routing] = 0.93
        changed = random._state(trace, state.write_count)
        assert torch.equal(random.read(repeated_keys[:, None], state),
                           random.read(repeated_keys[:, None], changed))
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("rule", ["hebbian", "gradient"])
@pytest.mark.parametrize("policy", ["content", "random"])
@pytest.mark.parametrize("decay", ["protected", "global"])
def test_exact_active_formula_and_independent_inactive_decay(rule, policy, decay):
    model = adapter(rule, policy, decay, 90701 if policy == "random" else None)
    keys, values = torch.tensor([0, 16]), torch.tensor([2, 5])
    trace = torch.linspace(-0.8, 0.8, 2 * 4 * 64 * 64).reshape(2, 4, 64, 64).requires_grad_()
    state = model._state(trace, 7)
    key, value = model.codebook.encode_keys(keys), model.codebook.encode_values(values)
    pre = torch.tanh(torch.einsum("bd,mdh->bmh", key, model.key_encoder))
    post = torch.tanh(torch.einsum("bd,mdh->bmh", value, model.value_encoder))
    if rule == "hebbian":
        change = pre[..., :, None] * post[..., None, :]
    else:
        prediction = torch.einsum("bmi,bmij->bmj", pre,
                                  model.W0[None] + model.alpha[None] * trace)
        gradient, = torch.autograd.grad(0.5 * (prediction - post).square().sum(), trace)
        change = -gradient
    active = model.routing_mask(keys)
    active_expected = ((1 - model.eta) * trace + model.eta * change).clamp(-1, 1)
    actual = model.write(keys, values, state)
    torch.testing.assert_close(actual.weights["trace"][active], active_expected[active], rtol=1e-6, atol=1e-7)
    inactive_expected = trace if decay == "protected" else (1 - model.eta) * trace
    assert torch.equal(actual.weights["trace"][~active], inactive_expected[~active])
    assert torch.equal(state.weights["trace"], trace)
    assert actual.write_count == 8
    assert active.sum(-1).tolist() == [1, 1]
    assert model.decay_mask(keys).sum(-1).tolist() == ([1, 1] if decay == "protected" else [4, 4])


@pytest.mark.parametrize("rule", ["hebbian", "gradient"])
@pytest.mark.parametrize("policy,decay", [("content", "protected"), ("random", "global")])
def test_online_has_no_autograd_engine_calls_and_offline_gradients_flow(rule, policy, decay, monkeypatch):
    model = adapter(rule, policy, decay, 90701 if policy == "random" else None)
    keys = torch.tensor([torch.where(model.route_table == module)[0][0] for module in range(4)])
    values = torch.tensor([2, 5, 7, 9])
    with monkeypatch.context() as context:
        def forbidden(*args, **kwargs):
            raise AssertionError("Online update invoked the autograd engine")
        context.setattr(torch.autograd, "grad", forbidden)
        context.setattr(torch.autograd, "backward", forbidden)
        context.setattr(torch.Tensor, "backward", forbidden)
        with torch.no_grad():
            written = model.write(keys, values, model.initial_state(4))
            assert torch.isfinite(model.read(keys[:, None], written)).all()
    state = model.initial_state(4)
    for offset in range(3):
        state = model.write(keys, values + offset, state)
    scores = model.read(keys[:, None], state)
    F.mse_loss(scores, F.one_hot((values + 2)[:, None], 16).float()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("rule", ["hebbian", "gradient"])
@pytest.mark.parametrize("policy", ["content", "random"])
@pytest.mark.parametrize("decay", ["protected", "global"])
def test_strict_saved_table_and_fast_state_resume_with_pure_reads(rule, policy, decay):
    model = adapter(rule, policy, decay, 90701 if policy == "random" else None)
    with torch.no_grad():
        state = model.write(torch.tensor([0, 16]), torch.tensor([2, 5]), model.initial_state(2))
        snapshot, slow = model.export_state(state), deepcopy(model.state_dict())
        rng = torch.random.get_rng_state().clone()
        queries = torch.tensor([[0, 16, 63], [16, 0, 31]])
        scores = model.read(queries, state)
        for _ in range(3):
            model.read(queries, state)  # paused stream/read-only intervals
        assert tree_equal(snapshot, model.export_state(state))
        assert tree_equal(slow, model.state_dict())
        assert torch.equal(rng, torch.random.get_rng_state())
        stream = io.BytesIO()
        torch.save(dict(config=model.config.to_dict(), model=slow, fast=snapshot), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = RoutingDecayAdapter(RoutingDecayConfig(**saved["config"]),
                                       saved["model"]["codebook.keys"], saved["model"]["codebook.values"],
                                       route_table=saved["model"]["route_table"])
        restored.load_state_dict(saved["model"], strict=True)
        imported = restored.import_state(saved["fast"])
        assert torch.equal(scores, restored.read(queries, imported))
        left = model.write(torch.tensor([16, 0]), torch.tensor([7, 8]), state)
        right = restored.write(torch.tensor([16, 0]), torch.tensor([7, 8]), imported)
        assert tree_equal(model.export_state(left), restored.export_state(right))
        assert not restored.detach_state(right).weights["trace"].requires_grad
        imported.weights["trace"].zero_()
        assert not torch.equal(imported.weights["trace"], snapshot["weights"]["trace"])


def test_digest_is_of_actual_int64_shape_dtype_and_little_endian_bytes():
    model = adapter(policy="random", seed=90701)
    table = model.route_table
    header = json.dumps({"shape": [64], "dtype": "torch.int64"},
                        sort_keys=True, separators=(",", ":")).encode()
    expected = hashlib.sha256(header + b"\0" + table.numpy().astype("<i8").tobytes()).hexdigest()
    assert expected == routing_table_sha256(table) == model.config.routing_table_sha256
    assert expected == model.routing_table_sha256
    assert RoutingDecayConfig(**model.config.to_dict()).fingerprint == model.config.fingerprint
    assert model.config.nkeys == 64 and model.config.nvalues == 16 and model.config.dim == 64
    assert model.config.routing_algorithm == ROUTING_ALGORITHM
    assert model.config_deviations["association_updates_per_write"] == 1


def test_fast_states_reject_cross_policy_seed_decay_rule_and_bad_payloads():
    model = adapter(policy="random", seed=90701)
    payload = model.export_state(model.initial_state(2))
    for other in (adapter(), adapter(policy="random", seed=90702),
                  adapter(policy="random", seed=90701, decay="global"),
                  adapter(rule="gradient", policy="random", seed=90701)):
        assert other.config.fingerprint != model.config.fingerprint
        with pytest.raises(ValueError):
            other.import_state(payload)
        with pytest.raises(ValueError):
            other.read(torch.tensor([[0], [1]]), model.initial_state(2))
    mutations = (
        lambda p: p.update(schema_version=1), lambda p: p.update(history=[]),
        lambda p: p.update(batch_size=True), lambda p: p.update(write_count=-1),
        lambda p: p.update(aux={}), lambda p: p["config"].update(trace_bound=True),
        lambda p: p["config"].update(routing_table_sha256="0" * 64),
        lambda p: p["weights"].update(trace=torch.full((2, 4, 64, 64), float("nan"))),
        lambda p: p["weights"].update(trace=torch.full((2, 4, 64, 64), 1.01)),
    )
    for mutate in mutations:
        invalid = deepcopy(payload)
        mutate(invalid)
        with pytest.raises(ValueError):
            model.import_state(invalid)


def test_strict_loader_validates_saved_route_codebook_and_all_slow_tensors():
    model = adapter(policy="random", seed=90701)
    original = deepcopy(model.state_dict())
    for mutate in (
        lambda p: p.pop("route_table"), lambda p: p.update(extra=torch.zeros(1)),
        lambda p: p.update(route_table=p["route_table"].to(torch.int32)),
        lambda p: p.update(route_table=(p["route_table"] + 1) % 4),
        lambda p: p.update(alpha=p["alpha"].to(torch.float64)),
        lambda p: p["alpha"].fill_(float("nan")),
        lambda p: p["codebook.keys"].zero_(),
        lambda p: p["codebook.values"].zero_(),
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
        adapter(policy="random", seed=90702).load_state_dict(original)
    for changes in ({"random_seed": 90702}, {"routing_table_sha256": "0" * 64}):
        with pytest.raises(ValueError):
            RoutingDecayAdapter(replace(model.config, **changes), model.codebook.keys,
                                model.codebook.values, route_table=model.route_table)
    model.load_state_dict(original, assign=True)
    model.assert_codebook_valid()
    with torch.no_grad():
        model.route_table[0] = (model.route_table[0] + 1) % 4
    with pytest.raises(ValueError):
        model.export_state(model.initial_state(1))


@pytest.mark.parametrize("changes", [
    {"modules": True}, {"modules": 2}, {"dim": 32}, {"update_rule": "oja"},
    {"eta_init": 0}, {"eta_init": 1}, {"eta_init": True}, {"trace_bound": 0},
    {"alpha_init": float("nan")}, {"init_noise": -1}, {"policy": "label"},
    {"inactive_decay": "sometimes"}, {"policy": "random"},
    {"policy": "random", "random_seed": True}, {"policy": "random", "random_seed": -1},
    {"policy": "random", "random_seed": 2**64}, {"random_seed": 90701},
    {"routing_algorithm": "unversioned"}, {"routing_table_sha256": "invalid"},
])
def test_invalid_configs(changes):
    with pytest.raises(ValueError):
        RoutingDecayConfig(CorrelatedConfig().to_dict(), **changes)


def test_input_validation_and_parent_boundary():
    model = adapter(decay="global")
    state = model.initial_state(1)
    for key in (torch.tensor([64]), torch.tensor([-1]), torch.tensor([1.0])):
        with pytest.raises(ValueError):
            model.routing_mask(key)
        with pytest.raises(ValueError):
            model.decay_mask(key)
    with pytest.raises(ValueError):
        model.write(torch.tensor([[0]]), torch.tensor([1]), state)
    with pytest.raises(ValueError):
        model.write(torch.tensor([0]), torch.tensor([16]), state)
    with pytest.raises(ValueError):
        model.read(torch.tensor([[1.0]]), state)
    for wrong_parent in (model, object(), parent(routing="top1")):
        with pytest.raises(ValueError):
            RoutingDecayAdapter.from_parent(wrong_parent, policy="content", inactive_decay="protected")
