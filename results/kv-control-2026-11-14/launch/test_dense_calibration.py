"""CPU-only checks of the single fixed dense-code calibration adapter."""
from copy import deepcopy
from dataclasses import replace
import io
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_dense as dense


@pytest.fixture(scope="module", autouse=True)
def frozen_repo():
    repo = Path(os.environ.get("CLMM_FROZEN_REPO", Path.home() / "Documents/Codex/clmm-replay"))
    dense.load_frozen(repo)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_fixed_codebook_is_reproducible_normalized_distinct_and_not_orthogonal():
    before = torch.random.get_rng_state().clone()
    model, other = dense.DenseDeltaAdapter(), dense.DenseDeltaAdapter()
    assert torch.equal(before, torch.random.get_rng_state())
    assert not list(model.parameters())
    assert torch.equal(model.keys, other.keys)
    assert model.keys.shape == (64, 32) and model.values.shape == (16, 32)
    torch.testing.assert_close(model.keys.norm(dim=-1), torch.ones(64), rtol=0, atol=2e-7)
    assert torch.unique(model.keys, dim=0).shape[0] == 64
    assert (model.keys @ model.keys.T - torch.eye(64)).abs().max() > 0.1
    assert torch.equal(model.values, torch.eye(32)[:16])
    assert model.config.codebook_sha256 == dense.codebook_sha256(model.keys, model.values)
    assert model.config.adapter_source_sha256 == dense.sha256(dense.__file__)
    from clmm.pretrain.kv_control import physical_state_bytes
    assert physical_state_bytes(model.initial_state(3)) == 3 * 4096


def test_step_one_update_equals_autograd_half_sum_sse():
    model = dense.DenseDeltaAdapter()
    weight = torch.randn(2, 32, 32, generator=torch.Generator().manual_seed(4), requires_grad=True)
    state = replace(model.initial_state(2), weights={"weight": weight})
    keys, values = torch.tensor([0, 63]), torch.tensor([1, 15])
    prediction = torch.bmm(model.keys[keys, None], weight)[:, 0]
    loss = 0.5 * (prediction - model.values[values]).square().sum()
    gradient, = torch.autograd.grad(loss, weight)
    updated = model.write(keys, values, state)
    torch.testing.assert_close(updated.weights["weight"], weight - gradient, rtol=0, atol=0)


def test_orthogonal_keys_preserve_previous_writes_but_dense_keys_interfere():
    basis = torch.eye(32)
    weight = torch.zeros(1, 32, 32)
    weight = dense.delta_update(basis[0:1], basis[3:4], weight)
    weight = dense.delta_update(basis[1:2], basis[4:5], weight)
    assert torch.equal(basis[:2] @ weight[0], basis[[3, 4]])
    model = dense.DenseDeltaAdapter()
    state = model.write(torch.tensor([0]), torch.tensor([3]), model.initial_state(1))
    before = model.read(torch.tensor([[0]]), state)
    state = model.write(torch.tensor([1]), torch.tensor([4]), state)
    assert not torch.equal(before, model.read(torch.tensor([[0]]), state))


def test_read_purity_and_serialized_continuation_are_exact():
    model = dense.DenseDeltaAdapter()
    state = model.write(torch.tensor([2, 5]), torch.tensor([1, 7]), model.initial_state(2))
    snapshot = model.export_state(state)
    rng = torch.random.get_rng_state().clone()
    slow = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    query = torch.tensor([[2, 3], [5, 7]])
    model.read(query, state)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(snapshot["weights"]["weight"], state.weights["weight"])
    assert all(torch.equal(tensor, slow[name]) for name, tensor in model.state_dict().items())
    raw = io.BytesIO()
    torch.save(snapshot, raw)
    raw.seek(0)
    imported = model.import_state(torch.load(raw, weights_only=True, map_location="cpu"))
    assert imported.method == "dense_delta" and imported.aux is None
    for current in (state, imported):
        assert current.write_count == 1
    next_keys, next_values = torch.tensor([3, 7]), torch.tensor([2, 4])
    full = model.write(next_keys, next_values, state)
    resumed = model.write(next_keys, next_values, imported)
    assert torch.equal(full.weights["weight"], resumed.weights["weight"])
    assert torch.equal(model.read(query, full), model.read(query, resumed))


@pytest.mark.parametrize("change", ["source", "codebook", "fingerprint", "method", "nan", "dtype",
                                    "batch", "count", "history", "aux"])
def test_invalid_state_and_source_or_codebook_fingerprints_are_rejected(change):
    model = dense.DenseDeltaAdapter()
    payload = deepcopy(model.export_state(model.initial_state(2)))
    if change in ("source", "codebook"):
        field = "adapter_source_sha256" if change == "source" else "codebook_sha256"
        payload["config"][field] = "0" * 64
    elif change == "fingerprint":
        payload["config_fingerprint"] = "0" * 64
    elif change == "method":
        payload["method"] = "delta"
    elif change == "nan":
        payload["weights"]["weight"][0, 0, 0] = float("nan")
    elif change == "dtype":
        payload["weights"]["weight"] = payload["weights"]["weight"].double()
    elif change == "batch":
        payload["batch_size"] = 1
    elif change == "count":
        payload["write_count"] = True
    elif change == "history":
        payload["events"] = [(0, 1)]
    elif change == "aux":
        payload["aux"] = {"events": torch.zeros(2, 10)}
    with pytest.raises(ValueError):
        model.import_state(payload)


def test_modified_registered_codebook_is_rejected():
    model = dense.DenseDeltaAdapter()
    payload = model.export_state(model.initial_state(1))
    model.keys[0, 0] += 0.1
    with pytest.raises(ValueError, match="codebook fingerprint"):
        model.import_state(payload)


@pytest.mark.parametrize("keys,values", [([-1], [0]), ([64], [0]), ([0], [16])])
def test_out_of_range_ids_are_rejected(keys, values):
    model = dense.DenseDeltaAdapter()
    with pytest.raises(ValueError):
        model.write(torch.tensor(keys), torch.tensor(values), model.initial_state(1))
