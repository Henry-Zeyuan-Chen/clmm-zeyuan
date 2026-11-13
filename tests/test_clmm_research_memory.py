"""Vocabulary and checkpoint regressions for the standalone memory pilot."""

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F


# These files also serve as directly executable scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from research_memory import Reader, mean_squared_memory_loss
from train_memory_pilot import make_episodes, parse_args
from titans_pytorch import NeuralMemory


def test_legacy_checkpoint_without_vocabulary_config_loads_strictly(tmp_path):
    legacy_config = dict(dim=8, depth=2, expansion_factor=2.0,
                         init_adaptive_step_bias=1.0, init_decay_bias=-4.0, max_inner_lr=0.1)
    torch.manual_seed(41)
    legacy = nn.Module()
    legacy.key = nn.Embedding(16, 8)
    legacy.value = nn.Embedding(8, 8)
    legacy.memory = NeuralMemory(
        dim=8, heads=1, chunk_size=1, qkv_receives_diff_views=True,
        momentum=False, mem_model_norm_add_residual=False,
        store_memory_loss_fn=mean_squared_memory_loss,
        default_model_kwargs={"depth": 2, "expansion_factor": 2.0},
        init_adaptive_step_bias=1.0, init_decay_bias=-4.0,
        default_step_transform_max_lr=0.1,
    )
    legacy.head = nn.Linear(8, 8)
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": legacy.state_dict(), "config": legacy_config}, path)

    checkpoint = torch.load(path, weights_only=True)
    torch.manual_seed(41)
    restored = Reader(**checkpoint["config"])
    # Adding vocabulary arguments must preserve the original initialization too.
    assert restored.state_dict().keys() == checkpoint["state_dict"].keys()
    for name, tensor in restored.state_dict().items():
        assert torch.equal(tensor, checkpoint["state_dict"][name])
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    assert restored.config["nkeys"] == 16 and restored.config["nvals"] == 8


def test_large_vocabulary_episode_bounds_and_latest_labels():
    episodes = make_episodes(torch.Generator().manual_seed(113), 32, 64, nkeys=97, nvals=23)
    assert episodes["keys"].shape == episodes["values"].shape == (32, 64)
    assert 0 <= episodes["keys"].min() and episodes["keys"].max() < 97
    assert 0 <= episodes["values"].min() and episodes["values"].max() < 23
    assert episodes["keys"].max() > 15 and episodes["values"].max() > 7
    assert episodes["query"].max() > 15 and episodes["target"].max() > 7
    for row in range(32):
        keys, values = episodes["keys"][row].tolist(), episodes["values"][row].tolist()
        lookup = dict(zip(keys, values))
        query = int(episodes["query"][row])
        assert query in lookup and int(episodes["target"][row]) == lookup[query]


def test_repeated_key_targets_use_the_final_assignment():
    episodes = make_episodes(torch.Generator().manual_seed(19), 12, 9, nkeys=1, nvals=17)
    assert torch.equal(episodes["query"], torch.zeros(12, dtype=torch.long))
    assert torch.equal(episodes["target"], episodes["values"][:, -1])
    assert torch.equal(episodes["delay"], torch.zeros(12, dtype=torch.long))
    expected_revised = (episodes["values"][:, :-1] != episodes["values"][:, -1:]).any(dim=1)
    assert expected_revised.any()
    assert torch.equal(episodes["revised"], expected_revised)


def test_every_configured_embedding_accepts_inputs_and_receives_gradients():
    torch.manual_seed(107)
    model = Reader(dim=8, depth=2, max_inner_lr=0.1, nkeys=128, nvals=32)
    keys = torch.arange(128).reshape(128, 1)
    values = keys.remainder(32)
    fast = model.write(keys, values)
    logits = model.read(keys[:, 0], fast)
    assert logits.shape == (128, 32) and torch.isfinite(logits).all()
    F.cross_entropy(logits, values[:, 0]).backward()
    assert model.key.weight.grad.shape == (128, 8)
    assert (model.key.weight.grad.abs().sum(dim=1) > 0).all()
    assert (model.value.weight.grad.abs().sum(dim=1) > 0).all()
    restored = Reader(**model.config)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.key.num_embeddings == 128
    assert restored.value.num_embeddings == restored.head.out_features == 32


def test_default_episode_stream_matches_explicit_original_vocabulary():
    default = make_episodes(torch.Generator().manual_seed(101), 4, 7)
    explicit = make_episodes(torch.Generator().manual_seed(101), 4, 7, nkeys=16, nvals=8)
    assert all(torch.equal(default[name], explicit[name]) for name in default)


def test_vocabulary_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_memory_pilot.py", "--out", "unused",
                                     "--nkeys", "128", "--nvals", "32"])
    args = parse_args()
    assert (args.nkeys, args.nvals) == (128, 32)


@pytest.mark.parametrize("invalid", [{"nkeys": 0}, {"nvals": 0}])
def test_empty_vocabulary_is_rejected(invalid):
    with pytest.raises(ValueError, match="Vocabulary sizes"):
        Reader(dim=8, **invalid)
    with pytest.raises(ValueError, match="Vocabulary sizes"):
        make_episodes(torch.Generator(), 2, 3, **invalid)
