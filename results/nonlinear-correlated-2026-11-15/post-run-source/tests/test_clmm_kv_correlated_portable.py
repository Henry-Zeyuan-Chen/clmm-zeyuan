"""CPU compatibility checks that preserve literal frozen checkpoint bits."""
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path

import pytest
import torch

from clmm.pretrain.kv_control import NeuralAdapter
from clmm.pretrain.kv_correlated import CorrelatedConfig, _key_codes, codebook_tensor_sha256
from clmm.pretrain.kv_correlated_portable import (
    PortableCorrelatedConfig, PortableCorrelatedNeuralAdapter, load_portable_checkpoint,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_parts():
    keys = _key_codes(64, 64, 32, 90501)
    keys[0, 0] = torch.nextafter(keys[0, 0], torch.tensor(float("inf")))
    config = PortableCorrelatedConfig(key_codebook_sha256=codebook_tensor_sha256(keys))
    return config, keys, torch.eye(64)[:16]


def checkpoint_for(model):
    return dict(format="clmm.kv.correlated.v1", method="neural_correlated", config=model.config.to_dict(),
                codebook_hash=model.codebook_hash, model_state=model.state_dict())


def test_slight_ulp_difference_is_authoritative_and_preserves_original_fingerprint(tmp_path):
    config, keys, values = fixture_parts()
    assert not torch.equal(keys, _key_codes(64, 64, 32, 90501))
    with pytest.raises(ValueError):
        CorrelatedConfig(**config.to_dict())
    model = PortableCorrelatedNeuralAdapter(config, keys, values)
    path = tmp_path / "ulp.pt"
    original = checkpoint_for(model)
    torch.save(original, path)
    restored, payload = load_portable_checkpoint(path, "cpu")
    assert restored.config.to_dict() == original["config"]
    encoded = json.dumps(original["config"], sort_keys=True, separators=(",", ":")).encode()
    import hashlib
    assert restored.config.fingerprint == hashlib.sha256(encoded).hexdigest()
    assert payload["codebook_hash"] == config.key_codebook_sha256
    assert torch.equal(restored.codebook.keys, keys)
    for name, tensor in original["model_state"].items():
        assert torch.equal(restored.state_dict()[name], tensor), name


def test_engine_rng_consumption_and_methods_match_baseline():
    config, keys, values = fixture_parts()
    before = torch.random.get_rng_state().clone()
    PortableCorrelatedConfig(**config.to_dict())
    assert torch.equal(before, torch.random.get_rng_state())
    torch.manual_seed(90511)
    baseline = NeuralAdapter()
    after = torch.random.get_rng_state().clone()
    torch.manual_seed(90511)
    portable = PortableCorrelatedNeuralAdapter(config, keys, values)
    assert torch.equal(after, torch.random.get_rng_state())
    assert portable.parameter_counts() == baseline.parameter_counts()
    assert all(torch.equal(tensor, dict(baseline.named_parameters())[name])
               for name, tensor in portable.named_parameters())
    assert PortableCorrelatedNeuralAdapter.write is NeuralAdapter.write
    assert PortableCorrelatedNeuralAdapter.read is NeuralAdapter.read


def test_distinct_saved_frozen_and_trainable_initial_matrices_remain_distinct(tmp_path):
    config, keys, values = fixture_parts()
    model = PortableCorrelatedNeuralAdapter(config, keys, values)
    payload = deepcopy(checkpoint_for(model))
    payload["model_state"]["engine.memory_model_parameters.0"][0, 0, 0] += 0.125
    payload["model_state"]["engine.memory_model_parameters.1"][0, 0, 0] += 0.25
    path = tmp_path / "distinct-registered-matrices.pt"
    torch.save(payload, path)
    restored, source = load_portable_checkpoint(path, "cpu")
    for name, tensor in source["model_state"].items():
        assert torch.equal(tensor, restored.state_dict()[name]), name


def test_exact_portable_state_handoff_and_continuation():
    config, keys, values = fixture_parts()
    model = PortableCorrelatedNeuralAdapter(config, keys, values)
    state = model.write(torch.tensor([0, 7]), torch.tensor([1, 3]), model.initial_state(2))
    raw = io.BytesIO()
    torch.save(model.export_state(state), raw)
    raw.seek(0)
    payload = torch.load(raw, map_location="cpu", weights_only=True)
    imported = model.import_state(payload)
    assert imported.config_fingerprint == payload["config_fingerprint"] == config.fingerprint
    query = torch.tensor([[0, 2], [7, 9]])
    assert torch.equal(model.read(query, state), model.read(query, imported))
    full = model.write(torch.tensor([2, 9]), torch.tensor([2, 4]), state)
    resumed = model.write(torch.tensor([2, 9]), torch.tensor([2, 4]), imported)
    assert all(torch.equal(tensor, resumed.weights[name]) for name, tensor in full.weights.items())
    assert torch.equal(model.read(query, full), model.read(query, resumed))
    wrong = deepcopy(payload)
    wrong["config"]["key_codebook_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="config|fingerprint"):
        model.import_state(wrong)


@pytest.mark.parametrize("change", ["config_hash", "payload_hash", "rank", "seed", "material", "padding", "values", "nan", "dtype"])
def test_invalid_literal_checkpoint_is_rejected(tmp_path, change):
    config, keys, values = fixture_parts()
    model = PortableCorrelatedNeuralAdapter(config, keys, values)
    payload = deepcopy(checkpoint_for(model))
    if change == "config_hash": payload["config"]["key_codebook_sha256"] = "0" * 64
    elif change == "payload_hash": payload["codebook_hash"] = "0" * 64
    elif change == "rank": payload["config"]["rank"] = 16
    elif change == "seed": payload["config"]["codebook_seed"] = 90502
    else:
        if change == "material": payload["model_state"]["codebook.keys"][0, 0] += 1e-3
        elif change == "padding": payload["model_state"]["codebook.keys"][0, 63] = 1e-8
        elif change == "values": payload["model_state"]["codebook.values"][0, 0] = 0.5
        elif change == "nan": payload["model_state"]["codebook.keys"][0, 0] = float("nan")
        elif change == "dtype": payload["model_state"]["engine.memory_model_parameters.0"] = payload["model_state"]["engine.memory_model_parameters.0"].double()
        if change in ("material", "padding", "nan"):
            digest = codebook_tensor_sha256(payload["model_state"]["codebook.keys"])
            payload["config"]["key_codebook_sha256"] = payload["codebook_hash"] = digest
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_portable_checkpoint(path, "cpu")


@pytest.mark.parametrize("change", [{"key_codebook_sha256": None}, {"key_codebook_sha256": "bad"},
    {"rank": True}, {"rank": 64}, {"codebook_seed": -1}, {"dim": 32}, {"step_size": float("inf")}])
def test_portable_config_is_validated(change):
    config, _, _ = fixture_parts()
    with pytest.raises(ValueError):
        replace(config, **change)


def test_downloaded_gpu_checkpoint_and_saved_boundary_state_if_available():
    default = Path.home() / "Documents/Codex/2026-10-22/i/outputs/clmm-nonlinear-correlated-2026-11-15/r32-90511"
    run = Path(os.environ.get("CLMM_CORRELATED_TEST_RUN", default))
    if not (run / "checkpoint.pt").is_file() or not (run / "evaluation/example_boundary_state.pt").is_file():
        pytest.skip("Downloaded frozen correlated run is not available")
    model, source = load_portable_checkpoint(run / "checkpoint.pt", "cpu")
    model.assert_codebook_valid()
    assert model.codebook_hash == source["config"]["key_codebook_sha256"] == source["codebook_hash"]
    for name, tensor in source["model_state"].items():
        assert torch.equal(tensor, model.state_dict()[name]), name
    payload = torch.load(run / "evaluation/example_boundary_state.pt", map_location="cpu", weights_only=True)
    imported = model.import_state(payload)
    assert imported.config_fingerprint == payload["config_fingerprint"] == model.config.fingerprint
    for name, tensor in payload["weights"].items():
        assert torch.equal(tensor, imported.weights[name])
    keys, values = torch.arange(imported.batch_size).remainder(64), torch.arange(imported.batch_size).remainder(16)
    queries = keys[:, None]
    full = model.write(keys, values, imported)
    restored = model.import_state(model.export_state(imported))
    continued = model.write(keys, values, restored)
    assert torch.equal(model.read(queries, full), model.read(queries, continued))
