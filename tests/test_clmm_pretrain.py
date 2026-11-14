"""Small CPU checks for shared byte-LM data, checkpoint and forward contracts."""

from dataclasses import replace
import gzip
import hashlib

import pytest
import torch

from clmm.pretrain import (ByteLMConfig, build_model, fixed_windows, load_checkpoint,
                          load_enwik8, sample_windows, save_checkpoint, zero_memory_output,
                          reset_memory_at_segments)
from titans_pytorch.neural_memory import NeuralMemory


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return ByteLMConfig(dim=16, depth=2, dim_head=8, heads=2, segment_len=8,
                        memory_dim_head=8, memory_heads=2, memory_segment_len=4,
                        memory_batch_size=8, num_persist_mem_tokens=2)


def test_config_rejects_unsupported_head_geometry():
    assert ByteLMConfig().memory_heads == 4
    with pytest.raises(ValueError, match="single-head"):
        ByteLMConfig(memory_heads=1)
    with pytest.raises(ValueError, match="divisible"):
        ByteLMConfig(memory_batch_size=7)
    with pytest.raises(ValueError, match="within"):
        ByteLMConfig(memory_layers=(3,))


def test_multihead_parameter_storage_is_independently_writable(config):
    model = build_model(config)
    parameters = dict(model.named_parameters())
    assert model.clmm_materialized_parameters
    for name in model.clmm_materialized_parameters:
        parameter = parameters[name]
        assert parameter.shape[0] == config.memory_heads and parameter.is_contiguous()
        assert torch.equal(parameter[0], parameter[1])
        second_head = parameter[1].detach().clone()
        with torch.no_grad():
            parameter[0].add_(1)
        assert torch.equal(parameter[1], second_head)


def test_chronological_data_bounds_and_local_reproducibility(tmp_path):
    path = tmp_path / "bytes"
    path.write_bytes(bytes(range(240)))
    corpus = load_enwik8(path, split_bounds=(0, 120, 180, 240))
    assert corpus.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    global_rng = torch.random.get_rng_state().clone()
    for split, lower, upper in (("train", 0, 120), ("dev", 120, 180), ("test", 180, 240)):
        first, offsets = sample_windows(corpus, split, 11, 20, torch.Generator().manual_seed(15))
        again, repeated = sample_windows(corpus, split, 11, 20, torch.Generator().manual_seed(15))
        assert torch.equal(first, again) and torch.equal(offsets, repeated)
        assert (offsets >= lower).all() and (offsets + 12 <= upper).all()
        assert first.dtype == torch.long and first.shape == (20, 12)
        assert torch.equal(first[:, 0], offsets)
    fixed, offsets = fixed_windows(corpus, "dev", 11, 5, 18)
    assert torch.unique(offsets).numel() == 5
    assert (offsets.sort().values.diff() >= 12).all()
    assert torch.equal(fixed, fixed_windows(corpus, "dev", 11, 5, 18)[0])
    assert torch.equal(global_rng, torch.random.get_rng_state())
    with pytest.raises(ValueError, match="nonoverlapping"):
        fixed_windows(corpus, "dev", 11, 6, 18)
    with pytest.raises(ValueError, match="SHA256"):
        load_enwik8(path, split_bounds=(0, 120, 180, 240), expected_sha256="bad")
    with pytest.raises(ValueError, match="too short"):
        sample_windows(corpus, "dev", 60, 1, torch.Generator())


def test_gzip_loading_matches_raw_and_validates_decoded_size(tmp_path):
    data = bytes(range(240))
    raw, zipped = tmp_path / "raw", tmp_path / "compressed-without-suffix"
    raw.write_bytes(data)
    zipped.write_bytes(gzip.compress(data))
    plain = load_enwik8(raw, split_bounds=(0, 120, 180, 240))
    compressed = load_enwik8(zipped, split_bounds=(0, 120, 180, 240))
    assert compressed.data == plain.data and compressed.sha256 == plain.sha256
    assert torch.equal(fixed_windows(compressed, "dev", 11, 5, 1)[0],
                       fixed_windows(plain, "dev", 11, 5, 1)[0])
    zipped.write_bytes(gzip.compress(data + b"x"))
    with pytest.raises(ValueError, match="Decoded corpus size"):
        load_enwik8(zipped, split_bounds=(0, 120, 180, 240))


def test_causal_logits_and_eval_does_not_mutate_outer_weights(config):
    torch.manual_seed(17)
    model = build_model(config).eval()
    tokens = torch.randint(256, (2, 24))
    changed = tokens.clone()
    changed[:, 13:] = (changed[:, 13:] + 1) % 256
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        logits = model(tokens)
        other = model(changed)
    assert logits.shape == (2, 24, 256) and torch.isfinite(logits).all()
    torch.testing.assert_close(logits[:, :13], other[:, :13], rtol=0, atol=1e-6)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_memory_gradients_optimizer_and_checkpoint_roundtrip(config, tmp_path):
    torch.manual_seed(23)
    model = build_model(config)
    tokens = torch.randint(256, (2, 25))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(tokens, return_loss=True)
    assert torch.isfinite(loss)
    loss.backward()
    memory_gradients = [parameter.grad for name, parameter in model.named_parameters()
                        if "memory_model_parameters" in name]
    assert memory_gradients and all(gradient is not None and torch.isfinite(gradient).all()
                                    for gradient in memory_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in memory_gradients) > 0
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint(path, model, config, training_info={"steps": 1})
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    restored, recovered, payload = load_checkpoint(path, expected_config=config)
    assert recovered == config and payload["training_info"] == {"steps": 1}
    assert payload["source_info"]["adapter_materialized_parameters"] == list(model.clmm_materialized_parameters)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    model.eval(); restored.eval()
    with torch.no_grad():
        assert torch.equal(model(tokens), restored(tokens))
    with pytest.raises(ValueError, match="expected_config"):
        load_checkpoint(path, expected_config=replace(config, max_inner_lr=.02))
    payload["architecture"] = "wrong"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="format or architecture"):
        load_checkpoint(path)


def test_zero_retrieval_preserves_state_and_restores_nested_exception(config):
    torch.manual_seed(31)
    model = build_model(config).eval()
    memory = next(module for module in model.modules() if isinstance(module, NeuralMemory))
    inputs = torch.randn(3, 2, 16, config.dim)
    with torch.no_grad():
        expected_retrieved, expected_state = memory.forward(inputs)
    original_forward = memory.forward
    assert "forward" not in memory.__dict__
    with pytest.raises(RuntimeError, match="test interruption"):
        with zero_memory_output(model):
            outer_wrapper = memory.forward
            with zero_memory_output(model), torch.no_grad():
                retrieved, state = memory.forward(inputs)
                assert torch.count_nonzero(retrieved) == 0 and torch.count_nonzero(expected_retrieved) > 0
                assert state.seq_index == expected_state.seq_index
                for name in state.updates.keys():
                    assert torch.equal(state.updates[name], expected_state.updates[name])
            assert memory.forward is outer_wrapper
            raise RuntimeError("test interruption")
    assert "forward" not in memory.__dict__ and memory.forward == original_forward
    with torch.no_grad():
        tokens = torch.randint(256, (2, 24))
        baseline = model(tokens)
        with zero_memory_output(model):
            ablated = model(tokens)
        assert not torch.equal(baseline, ablated)
        assert torch.equal(baseline, model(tokens))
    model.train()
    with pytest.raises(ValueError, match="evaluation-only"):
        with zero_memory_output(model):
            pass


def test_segment_reset_removes_earlier_segment_influence_and_restores(config):
    torch.manual_seed(47)
    model = build_model(config).eval()
    tokens = torch.randint(256, (2, 24))
    perturbed = tokens.clone()
    perturbed[:, :8] = (perturbed[:, :8] + 17) % 256
    with torch.no_grad():
        full, changed_full = model(tokens), model(perturbed)
        assert (full[:, 8:] - changed_full[:, 8:]).abs().max() > 1e-6
        with reset_memory_at_segments(model):
            reset, changed_reset = model(tokens), model(perturbed)
            assert reset.shape == full.shape
            torch.testing.assert_close(full[:, :8], reset[:, :8], rtol=0, atol=1e-6)
            torch.testing.assert_close(reset[:, 8:], changed_reset[:, 8:], rtol=0, atol=1e-6)
            with pytest.raises(ValueError, match="cached"):
                model(tokens, cache=(0, None, None))
            with pytest.raises(ValueError, match="Nested"):
                with reset_memory_at_segments(model):
                    pass
        assert "forward" not in model.__dict__
        assert torch.equal(full, model(tokens))
    with pytest.raises(RuntimeError):
        with reset_memory_at_segments(model):
            raise RuntimeError("interrupted")
    assert "forward" not in model.__dict__ and not hasattr(model, "_clmm_reset_segments_active")
