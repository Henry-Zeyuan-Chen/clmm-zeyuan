"""Independent bounded CPU checks; no training or efficacy evaluation."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

started = time.perf_counter()
import torch
from torch.func import functional_call


def same_weights(a, b):
    assert a.write_count == b.write_count
    assert all(torch.equal(a.weights[name], b.weights[name]) for name in a.weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from clmm.pretrain.kv_control import NeuralAdapter, physical_state_bytes
    from clmm.pretrain.kv_correlated import CorrelatedConfig, CorrelatedNeuralAdapter
    torch.set_num_threads(1)
    raw = torch.randn(64, 64, generator=torch.Generator().manual_seed(90501))
    results = []
    max_error = 0.0
    for seed in (90511, 90515):
        torch.manual_seed(seed)
        baseline = NeuralAdapter()
        expected_parameters = {name: value.detach().clone() for name, value in baseline.named_parameters()}
        expected_rng = torch.random.get_rng_state().clone()
        for rank in (32, 16, 8):
            assert time.perf_counter() - started < 55
            torch.manual_seed(seed)
            model = CorrelatedNeuralAdapter(CorrelatedConfig(rank=rank))
            assert torch.equal(torch.random.get_rng_state(), expected_rng)
            assert all(torch.equal(value, expected_parameters[name]) for name, value in model.named_parameters())
            expected_keys = torch.zeros(64, 64)
            expected_keys[:, :rank] = torch.nn.functional.normalize(raw[:, :rank], dim=-1)
            assert torch.equal(model.codebook.keys, expected_keys)
            assert torch.equal(model.codebook.values, torch.eye(64)[:16])
            assert model.parameter_counts() == {"registered": 32898, "trainable": 16384}
            assert {name for name, value in model.named_parameters() if value.requires_grad} == {
                "engine.memory_model_parameters.0", "engine.memory_model_parameters.1"}
            model.requires_grad_(False)
            slow = {name: value.clone() for name, value in model.state_dict().items()}
            state = model.initial_state(2)
            for key_ids, value_ids in (([0, 7], [2, 4]), ([0, 9], [3, 5])):
                analytic = {name: [] for name in state.weights}
                for sample in range(2):
                    parameters = {name: value[sample].detach().clone().requires_grad_(True)
                                  for name, value in state.weights.items()}
                    prediction = functional_call(model.engine.memory_model, parameters,
                        model.codebook.keys[key_ids[sample]][None, None])
                    loss = 0.5 * (prediction - model.codebook.values[value_ids[sample]][None, None]).square().sum()
                    gradients = torch.autograd.grad(loss, tuple(parameters.values()))
                    for (name, value), gradient in zip(parameters.items(), gradients, strict=True):
                        analytic[name].append((value - 0.5 * gradient).detach())
                with torch.no_grad():
                    state = model.write(torch.tensor(key_ids), torch.tensor(value_ids), state)
                for name in state.weights:
                    expected = torch.stack(analytic[name])
                    error = float((state.weights[name] - expected).abs().max())
                    max_error = max(max_error, error)
                    assert torch.allclose(state.weights[name], expected, atol=1e-7, rtol=1e-6)
            queries = torch.tensor([[0, 1, 7], [7, 9, 0]])
            snapshot = model.export_state(state)
            scores = model.read(queries, state)
            assert torch.equal(scores, model.read(queries, state))
            assert all(torch.equal(value, snapshot["weights"][name]) for name, value in state.weights.items())
            stream = io.BytesIO()
            torch.save(snapshot, stream)
            stream.seek(0)
            restored = model.import_state(torch.load(stream, weights_only=True))
            same_weights(state, restored)
            assert torch.equal(model.read(queries, state), model.read(queries, restored))
            with torch.no_grad():
                continued = model.write(torch.tensor([3, 4]), torch.tensor([6, 7]), state)
                resumed = model.write(torch.tensor([3, 4]), torch.tensor([6, 7]), restored)
            same_weights(continued, resumed)
            assert torch.equal(model.read(queries, continued), model.read(queries, resumed))
            assert all(torch.equal(value, model.state_dict()[name]) for name, value in slow.items())
            results.append({"init_seed": seed, "rank": rank, "key_codebook_sha256": model.codebook_hash,
                            "live_state_bytes_per_episode": physical_state_bytes(state) // 2,
                            "compact_restored_bytes_per_episode": physical_state_bytes(restored) // 2})
    elapsed = time.perf_counter() - started
    assert elapsed < 60
    report = {"status": "passed", "device": "cpu", "torch": str(torch.__version__), "elapsed_seconds": elapsed,
              "scope": "API, analytic update, matched initialization and state mechanism only; no training or efficacy run",
              "same_engine_parameters_and_rng_across_ranks": True, "nested_gaussian_construction_matches": True,
              "analytic_half_sse_sgd_max_absolute_error": max_error, "state_roundtrip_and_continuation_bit_identical": True,
              "reads_pure_and_slow_state_unchanged": True,
              "adapter_sha256": hashlib.sha256((args.repo / "clmm/pretrain/kv_correlated.py").read_bytes()).hexdigest(),
              "conditions": results}
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
