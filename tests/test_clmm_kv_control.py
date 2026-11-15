"""Mechanism, purity, and portable-state tests for the KV positive control."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

from clmm.pretrain.kv_control import (
    DeltaAdapter,
    FixedCodebook,
    KVConfig,
    KVState,
    NeuralAdapter,
    half_sum_squared_error,
    physical_state_bytes,
)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return KVConfig(nkeys=8, nvalues=4, dim=8, hidden=16)


def write_prefix(adapter, keys, values, state=None):
    state = adapter.initial_state(keys.shape[0]) if state is None else state
    for index in range(keys.shape[1]):
        state = adapter.write(keys[:, index], values[:, index], state)
    return state


def tensors(tree):
    if isinstance(tree, torch.Tensor):
        yield tree
    elif type(tree) is dict:
        for value in tree.values():
            yield from tensors(value)
    elif type(tree) is tuple:
        for value in tree:
            yield from tensors(value)
    else:
        assert tree is None


def test_default_codebook_is_frozen_unit_one_hot_and_decodes_coordinates():
    config = KVConfig()
    assert config.to_dict() == {"nkeys": 64, "nvalues": 16, "dim": 64,
                                "hidden": 128, "step_size": 0.5}
    assert KVConfig(**config.to_dict()).fingerprint == config.fingerprint
    codebook = FixedCodebook(config)
    assert not list(codebook.parameters())
    assert set(dict(codebook.named_buffers())) == {"keys", "values"}
    assert torch.equal(codebook.keys @ codebook.keys.T, torch.eye(64))
    assert torch.equal(codebook.values @ codebook.values.T, torch.eye(16))
    assert torch.equal(codebook.encode_keys(torch.arange(64)), torch.eye(64))
    assert torch.equal(codebook.encode_values(torch.arange(16)), torch.eye(64)[:16])
    vectors = torch.arange(128, dtype=torch.float32).reshape(2, 64)
    assert torch.equal(codebook.decode(vectors), vectors[:, :16])


@pytest.mark.parametrize("changes", [
    {"nkeys": 65}, {"nvalues": 65}, {"dim": 0}, {"nkeys": 0},
    {"nvalues": -1}, {"hidden": 127}, {"nkeys": True}, {"dim": 64.0},
    {"step_size": 0}, {"step_size": float("nan")},
    {"step_size": float("inf")}, {"step_size": True},
])
def test_invalid_config_is_rejected(changes):
    with pytest.raises(ValueError):
        KVConfig(**changes)


def test_delta_exact_fresh_associations_and_revisions_for_all_64_keys():
    adapter = DeltaAdapter()
    state = adapter.initial_state(3)
    query = torch.arange(64).expand(3, -1)
    assert torch.count_nonzero(adapter.read(query, state)) == 0
    values = (query * 3 + torch.arange(3)[:, None]).remainder(16)
    state = write_prefix(adapter, query, values, state)
    assert torch.equal(adapter.read(query, state), F.one_hot(values, 16).float())
    revised_values = (values + 7).remainder(16)
    state = write_prefix(adapter, query.flip(1), revised_values.flip(1), state)
    assert state.write_count == 128
    assert torch.equal(adapter.read(query, state), F.one_hot(revised_values, 16).float())
    assert physical_state_bytes(state) == 3 * 64 * 64 * 4
    assert adapter.parameter_counts() == {"registered": 0, "trainable": 0}


def test_delta_write_is_exact_step_one_half_sum_sse_gradient(config):
    adapter = DeltaAdapter(config)
    base = adapter.initial_state(2)
    weight = torch.randn(2, 8, 8, generator=torch.Generator().manual_seed(41),
                         requires_grad=True)
    state = replace(base, weights={"weight": weight})
    keys, values = torch.tensor([1, 6]), torch.tensor([2, 3])
    prediction = torch.bmm(adapter.codebook.encode_keys(keys)[:, None], weight)[:, 0]
    loss = half_sum_squared_error(prediction, adapter.codebook.encode_values(values)).sum()
    gradient, = torch.autograd.grad(loss, weight)
    updated = adapter.write(keys, values, state)
    torch.testing.assert_close(updated.weights["weight"], weight - gradient, rtol=0, atol=0)
    assert torch.equal(state.weights["weight"], weight)


@pytest.mark.parametrize("adapter_class", [DeltaAdapter, NeuralAdapter])
def test_query_permutation_equivariance_and_read_purity(adapter_class, config):
    adapter = adapter_class(config).eval()
    keys = torch.tensor([[1, 4, 6], [6, 1, 4]])
    values = torch.tensor([[2, 1, 3], [3, 0, 2]])
    with torch.no_grad():
        state = write_prefix(adapter, keys, values)
    saved = [tensor.clone() for tensor in tensors((state.weights, state.aux))]
    slow = {name: tensor.clone() for name, tensor in adapter.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    query = torch.tensor([[6, 1, 4, 0], [4, 1, 6, 0]])
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        logits = adapter.read(query, state)
        swapped = adapter.read(query[:, permutation], state)
        repeated = adapter.read(query, state)
    assert logits.shape == (2, 4, 4)
    assert torch.equal(swapped, logits[:, permutation])
    assert torch.equal(repeated, logits)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert state.write_count == 3
    for before, after in zip(saved, tensors((state.weights, state.aux))):
        assert torch.equal(before, after)
    assert all(torch.equal(slow[name], tensor) for name, tensor in adapter.state_dict().items())


@pytest.mark.parametrize("adapter_class", [DeltaAdapter, NeuralAdapter])
def test_cpu_checkpoint_handoff_equals_full_prefix_and_reset_removes_memory(
        adapter_class, config, tmp_path):
    adapter = adapter_class(config)
    keys = torch.tensor([[0, 1, 0, 5, 1], [2, 4, 2, 7, 4]])
    values = torch.tensor([[1, 2, 3, 0, 1], [2, 0, 1, 3, 2]])
    prefix = write_prefix(adapter, keys[:, :3], values[:, :3])
    full = write_prefix(adapter, keys[:, 3:], values[:, 3:], prefix)
    path = tmp_path / "live-state.pt"
    torch.save({"config": config.to_dict(), "model": adapter.state_dict(),
                "fast_state": adapter.export_state(prefix)}, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restored = adapter_class(KVConfig(**checkpoint["config"]))
    restored.load_state_dict(checkpoint["model"], strict=True)
    imported = restored.import_state(checkpoint["fast_state"])
    assert imported.method == adapter.method
    assert imported.write_count == 3 and imported.batch_size == 2
    assert all(tensor.device.type == "cpu" and not tensor.requires_grad
               for tensor in tensors((imported.weights, imported.aux)))
    resumed = write_prefix(restored, keys[:, 3:], values[:, 3:], imported)
    query = torch.tensor([[0, 1, 5], [2, 4, 7]])
    assert torch.equal(restored.read(query, resumed), adapter.read(query, full))
    assert not torch.equal(restored.read(query, resumed),
                           restored.read(query, restored.initial_state(2)))
    assert torch.count_nonzero(restored.read(query, restored.initial_state(2))) == 0
    for name in full.weights:
        assert torch.equal(resumed.weights[name], full.weights[name])


def test_neural_uses_fixed_interface_gates_and_only_trainable_initial_weights(config):
    adapter = NeuralAdapter(config)
    engine = adapter.engine
    for name in ("to_queries", "to_keys", "to_values", "retrieve_norm", "store_norm",
                 "multihead_rmsnorm", "q_norm", "k_norm", "combine_heads"):
        assert isinstance(getattr(engine, name), nn.Identity)
    assert engine.retrieve_gate is None and engine.to_momentum is None
    assert engine.store_chunk_size == engine.retrieve_chunk_size == engine.heads == 1
    assert engine.qkv_receives_diff_views
    assert torch.count_nonzero(engine.to_decay_factor[0].weight) == 0
    assert torch.equal(engine.to_decay_factor[0].bias, torch.tensor([-80.0]))
    assert torch.equal(1 - engine.to_decay_factor(torch.eye(8)[None]).sigmoid(),
                       torch.ones(1, 8, 1))
    raw = torch.tensor([[-100.0, 0.0, 100.0]])
    assert torch.equal(engine.adaptive_step_transform(raw), torch.full_like(raw, 0.5))
    assert torch.count_nonzero(engine.memory_model.weights[-1]) == 0
    assert torch.count_nonzero(engine.memory_model_parameters[-1]) == 0
    assert torch.count_nonzero(engine.memory_model_parameters[0]) > 0
    assert all(parameter.requires_grad == name.startswith("engine.memory_model_parameters.")
               for name, parameter in adapter.named_parameters())
    trainable = 2 * config.dim * config.hidden
    assert adapter.parameter_counts() == {
        "registered": 2 * trainable + 2 * (config.dim + 1), "trainable": trainable}
    assert adapter.config_deviations["store_loss"] == "half_sum_squared_error"
    assert adapter.config_deviations["decay_logit"] == -80.0


def test_neural_write_matches_direct_gradient_and_preserves_outer_gradients(config):
    adapter = NeuralAdapter(config)
    state = adapter.initial_state(3)
    keys, values = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
    initial = {name: weight.detach().clone().requires_grad_()
               for name, weight in state.weights.items()}
    prediction = functional_call(adapter.engine.memory_model, initial,
                                 adapter.codebook.encode_keys(keys)[:, None])[:, 0]
    loss = half_sum_squared_error(prediction, adapter.codebook.encode_values(values)).sum()
    gradients = torch.autograd.grad(loss, tuple(initial.values()))
    state = adapter.write(keys, values, state)
    for (name, before), gradient in zip(initial.items(), gradients):
        torch.testing.assert_close(state.weights[name], before - 0.5 * gradient, rtol=0, atol=0)
    second_keys, second_values = torch.tensor([3, 4, 5]), torch.tensor([2, 3, 0])
    state = adapter.write(second_keys, second_values, state)
    query = torch.stack((keys, second_keys), dim=1)
    labels = torch.stack((values, second_values), dim=1)
    F.cross_entropy(adapter.read(query, state).flatten(0, 1), labels.flatten()).backward()
    for name, parameter in adapter.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None, name
    detached = adapter.detach_state(state)
    assert all(not tensor.requires_grad and tensor.grad_fn is None
               for tensor in tensors((detached.weights, detached.aux)))
    assert torch.equal(adapter.read(query, detached), adapter.read(query, state))


@pytest.mark.parametrize("adapter_class", [DeltaAdapter, NeuralAdapter])
def test_fast_state_is_bounded_plain_and_export_is_an_independent_snapshot(adapter_class, config):
    adapter = adapter_class(config)
    state = adapter.initial_state(2)
    with torch.no_grad():
        state = adapter.write(torch.tensor([0, 1]), torch.tensor([1, 2]), state)
        occupied = physical_state_bytes(state)
        for index in range(40):
            state = adapter.write(torch.tensor([index % 8, (index + 1) % 8]),
                                  torch.tensor([index % 4, (index + 1) % 4]), state)
        assert physical_state_bytes(state) == occupied
    payload = adapter.export_state(state)
    assert set(payload) == {"schema_version", "method", "config", "config_fingerprint",
                            "batch_size", "write_count", "weights", "aux"}
    assert type(state.weights) is dict
    assert all(not tensor.requires_grad for tensor in tensors((payload["weights"], payload["aux"])))
    if adapter_class is NeuralAdapter:
        assert type(state.aux) is tuple and type(state.aux[0]) is dict and state.aux[1] == {}
        assert all(state.weights[name] is state.aux[0][name] for name in state.weights)
        assert all(payload["weights"][name] is payload["aux"][0][name] for name in state.weights)
        assert physical_state_bytes(state) == sum(
            tensor.untyped_storage().nbytes() for tensor in state.weights.values())
    before = next(iter(state.weights.values())).clone()
    next(iter(payload["weights"].values())).add_(1)
    assert torch.equal(next(iter(state.weights.values())), before)


def test_physical_state_bytes_counts_shared_backing_storage_once():
    backing = torch.zeros(20)
    state = KVState({"weight": backing[:4]}, ({"other": backing[10:]}, {}), 1, "test", "test")
    assert physical_state_bytes(state) == 20 * 4


def test_import_validates_config_types_even_when_python_compares_them_equal():
    adapter = DeltaAdapter(KVConfig(nkeys=1, nvalues=1, dim=1, hidden=2))
    payload = adapter.export_state(adapter.initial_state(1))
    payload["config"]["nkeys"] = True
    # bool == int in ordinary dict equality, but this is not a valid config.
    with pytest.raises(ValueError, match="config"):
        adapter.import_state(payload)


@pytest.mark.parametrize("adapter_class", [DeltaAdapter, NeuralAdapter])
@pytest.mark.parametrize("corruption", [
    "method", "config", "fingerprint", "batch", "count", "bool_count", "dtype",
    "shape", "nonfinite", "history", "extra_weight", "aux", "version",
])
def test_import_strictly_rejects_invalid_or_unbounded_state(adapter_class, config, corruption):
    adapter = adapter_class(config)
    state = adapter.write(torch.tensor([0, 1]), torch.tensor([1, 2]), adapter.initial_state(2))
    payload = deepcopy(adapter.export_state(state))
    name = next(iter(payload["weights"]))
    if corruption == "method":
        payload["method"] = "other"
    elif corruption == "config":
        payload["config"]["step_size"] = 0.7
    elif corruption == "fingerprint":
        payload["config_fingerprint"] = "other"
    elif corruption == "batch":
        payload["batch_size"] = 3
    elif corruption == "count":
        payload["write_count"] = -1
    elif corruption == "bool_count":
        payload["write_count"] = True
    elif corruption == "dtype":
        payload["weights"][name] = payload["weights"][name].double()
    elif corruption == "shape":
        payload["weights"][name] = payload["weights"][name][:1]
    elif corruption == "nonfinite":
        payload["weights"][name].flatten()[0] = float("nan")
    elif corruption == "history":
        payload["events"] = [(0, 1)]
    elif corruption == "extra_weight":
        payload["weights"]["events"] = torch.zeros(2, 2)
    elif corruption == "aux":
        payload["aux"] = ({"history": torch.zeros(2, 5)}, {})
    elif corruption == "version":
        payload["schema_version"] = True
    with pytest.raises(ValueError):
        adapter.import_state(payload)


@pytest.mark.parametrize("adapter_class", [DeltaAdapter, NeuralAdapter])
def test_shapes_ids_methods_and_initial_batches_are_validated(adapter_class, config):
    adapter = adapter_class(config)
    state = adapter.initial_state(2)
    for batch in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            adapter.initial_state(batch)
    for key, value in [
        (torch.tensor([0]), torch.tensor([1])),
        (torch.tensor([[0], [1]]), torch.tensor([[1], [2]])),
        (torch.tensor([-1, 1]), torch.tensor([1, 2])),
        (torch.tensor([8, 1]), torch.tensor([1, 2])),
        (torch.tensor([0, 1]), torch.tensor([4, 2])),
        (torch.tensor([0.0, 1.0]), torch.tensor([1, 2])),
    ]:
        with pytest.raises(ValueError):
            adapter.write(key, value, state)
    for query in (torch.tensor([0, 1]), torch.empty(2, 0, dtype=torch.long),
                  torch.tensor([[0]]), torch.tensor([[0, 8], [0, 1]])):
        with pytest.raises(ValueError):
            adapter.read(query, state)
    with pytest.raises(ValueError, match="method or config"):
        adapter.read(torch.tensor([[0], [1]]), replace(state, method="other"))
    other = adapter_class(replace(config, step_size=0.3))
    with pytest.raises(ValueError, match="config"):
        other.import_state(adapter.export_state(state))
