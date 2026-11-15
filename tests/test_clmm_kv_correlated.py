"""Focused CPU mechanism and handoff checks for correlated nonlinear controls."""
from copy import deepcopy
import io

import pytest
import torch
from torch.func import functional_call
from torch.nn import functional as F

from clmm.pretrain.kv_control import NeuralAdapter, physical_state_bytes
from clmm.pretrain.kv_correlated import (
    CorrelatedCodebook, CorrelatedConfig, CorrelatedNeuralAdapter, codebook_tensor_sha256,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_nested_codes_are_deterministic_and_do_not_consume_global_rng():
    before = torch.random.get_rng_state().clone()
    raw = torch.randn(64, 64, generator=torch.Generator().manual_seed(90501))
    hashes, fingerprints = set(), set()
    for rank in (32, 16, 8):
        config = CorrelatedConfig(rank=rank)
        codebook = CorrelatedCodebook(config)
        expected = torch.zeros(64, 64)
        expected[:, :rank] = F.normalize(raw[:, :rank], dim=-1)
        assert torch.equal(codebook.keys, expected)
        assert torch.count_nonzero(codebook.keys[:, rank:]) == 0
        torch.testing.assert_close(codebook.keys.norm(dim=-1), torch.ones(64), rtol=0, atol=2e-7)
        assert torch.unique(codebook.keys, dim=0).shape[0] == 64
        assert torch.equal(codebook.values, torch.eye(64)[:16])
        assert CorrelatedConfig(**config.to_dict()) == config
        assert codebook_tensor_sha256(codebook.keys) == config.key_codebook_sha256
        hashes.add(config.key_codebook_sha256)
        fingerprints.add(config.fingerprint)
    assert len(hashes) == len(fingerprints) == 3
    assert torch.equal(before, torch.random.get_rng_state())


def test_same_engine_parameters_shapes_counts_and_initialization_as_baseline():
    torch.manual_seed(90511)
    baseline = NeuralAdapter()
    after_baseline = torch.random.get_rng_state().clone()
    torch.manual_seed(90511)
    correlated = CorrelatedNeuralAdapter()
    assert torch.equal(after_baseline, torch.random.get_rng_state())
    assert correlated.parameter_counts() == baseline.parameter_counts() == {"registered": 32898, "trainable": 16384}
    assert [(name, tuple(parameter.shape)) for name, parameter in correlated.named_parameters()] == [
        (name, tuple(parameter.shape)) for name, parameter in baseline.named_parameters()]
    assert all(torch.equal(parameter, dict(baseline.named_parameters())[name])
               for name, parameter in correlated.named_parameters())
    assert CorrelatedNeuralAdapter.write is NeuralAdapter.write
    assert CorrelatedNeuralAdapter.read is NeuralAdapter.read
    assert CorrelatedNeuralAdapter._validate_state is NeuralAdapter._validate_state
    assert correlated.method == "neural_correlated"
    assert correlated.codebook_hash == correlated.config.key_codebook_sha256
    assert correlated.config_deviations["key_rank"] == 32


def test_each_write_matches_current_state_half_sse_sgd_and_outer_gradients_remain_live():
    model = CorrelatedNeuralAdapter(CorrelatedConfig(rank=16))
    state = model.initial_state(2)
    queries, targets = [], []
    for keys, values in (([0, 7], [2, 4]), ([1, 9], [3, 5]), ([2, 11], [4, 6])):
        keys, values = torch.tensor(keys), torch.tensor(values)
        current = {name: weight.detach().clone().requires_grad_() for name, weight in state.weights.items()}
        predicted = functional_call(model.engine.memory_model, current, model.codebook.encode_keys(keys)[:, None])[:, 0]
        loss = 0.5 * (predicted - model.codebook.encode_values(values)).square().sum()
        gradients = torch.autograd.grad(loss, tuple(current.values()))
        state = model.write(keys, values, state)
        for (name, weight), gradient in zip(current.items(), gradients):
            torch.testing.assert_close(state.weights[name], weight - 0.5 * gradient, rtol=1e-5, atol=2e-7)
        queries.append(keys)
        targets.append(values)
    logits = model.read(torch.stack(queries, dim=1), state)
    F.cross_entropy(logits.flatten(0, 1), torch.stack(targets, dim=1).flatten()).backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None, name


def test_checkpoint_and_portable_state_continuation_are_exact_and_reads_are_pure():
    model = CorrelatedNeuralAdapter(CorrelatedConfig(rank=8))
    state = model.write(torch.tensor([2, 5]), torch.tensor([1, 7]), model.initial_state(2))
    state = model.write(torch.tensor([3, 6]), torch.tensor([4, 2]), state)
    portable = model.export_state(state)
    rng = torch.random.get_rng_state().clone()
    slow = {name: value.clone() for name, value in model.state_dict().items()}
    query = torch.tensor([[2, 3], [5, 6]])
    logits = model.read(query, state)
    assert torch.equal(logits, model.read(query, state))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in slow.items())
    assert all(torch.equal(value, portable["weights"][name]) for name, value in state.weights.items())
    raw = io.BytesIO()
    torch.save(dict(config=model.config.to_dict(), model=model.state_dict(), state=portable), raw)
    raw.seek(0)
    checkpoint = torch.load(raw, map_location="cpu", weights_only=True)
    restored = CorrelatedNeuralAdapter(CorrelatedConfig(**checkpoint["config"]))
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored.assert_codebook_valid()
    imported = restored.import_state(checkpoint["state"])
    assert imported.method == "neural_correlated" and imported.write_count == 2
    assert torch.equal(logits, restored.read(query, imported))
    keys, values = torch.tensor([4, 7]), torch.tensor([5, 3])
    full, resumed = model.write(keys, values, state), restored.write(keys, values, imported)
    assert all(torch.equal(full.weights[name], resumed.weights[name]) for name in full.weights)
    assert torch.equal(model.read(query, full), restored.read(query, resumed))
    assert physical_state_bytes(full) == physical_state_bytes(resumed)


@pytest.mark.parametrize("changes", [{"rank": 16}, {"codebook_seed": 90502}])
def test_wrong_rank_or_seed_rejects_state_and_checkpoint_codebooks(changes):
    source = CorrelatedNeuralAdapter()
    other = CorrelatedNeuralAdapter(CorrelatedConfig(**changes))
    with pytest.raises(ValueError, match="config|fingerprint"):
        other.import_state(source.export_state(source.initial_state(1)))
    with pytest.raises(ValueError, match="codebook"):
        other.load_state_dict(source.state_dict(), strict=True)
    other.assert_codebook_valid()


@pytest.mark.parametrize("name", ["codebook.keys", "codebook.values"])
def test_modified_checkpoint_codebooks_are_rejected(name):
    model = CorrelatedNeuralAdapter()
    checkpoint = deepcopy(model.state_dict())
    checkpoint[name][0, 0] += 0.1
    with pytest.raises(ValueError, match="codebook"):
        model.load_state_dict(checkpoint, strict=True)
    model.assert_codebook_valid()


@pytest.mark.parametrize("changes", [{"rank": 0}, {"rank": 64}, {"rank": True}, {"rank": 16.0},
    {"codebook_seed": -1}, {"codebook_seed": True}, {"codebook_seed": 2**64},
    {"dim": 32}, {"hidden": 64}, {"nkeys": 63}, {"nvalues": 15},
    {"step_size": float("nan")}, {"step_size": 0}, {"key_codebook_sha256": "0" * 64}])
def test_invalid_config_is_rejected(changes):
    with pytest.raises(ValueError):
        CorrelatedConfig(**changes)


def test_fixed_decoder_invalid_ids_and_distinct_baseline_state():
    model = CorrelatedNeuralAdapter()
    vectors = torch.arange(128, dtype=torch.float32).reshape(2, 64)
    assert torch.equal(model.codebook.decode(vectors), vectors[:, :16])
    state = model.initial_state(1)
    for key, value in ((-1, 0), (64, 0), (0, 16)):
        with pytest.raises(ValueError):
            model.write(torch.tensor([key]), torch.tensor([value]), state)
    with pytest.raises(ValueError):
        model.read(torch.tensor([[64]]), state)
    with pytest.raises(ValueError):
        model.read(torch.tensor([[1.0]]), state)
    with pytest.raises(ValueError, match="method"):
        model.import_state(NeuralAdapter().export_state(NeuralAdapter().initial_state(1)))
    payload = model.export_state(state)
    payload["config"]["key_codebook_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="config"):
        model.import_state(payload)
