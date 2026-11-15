"""Independent CPU-only analytic and serialization checks of the KV adapter."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import torch
from torch.func import functional_call


def digest_parameters(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensors(item)]
    if isinstance(value, tuple):
        return [tensor for item in value for tensor in tensors(item)]
    assert value is None
    return []


def same_state(left, right):
    assert left.write_count == right.write_count
    assert left.method == right.method
    assert left.config_fingerprint == right.config_fingerprint
    for a, b in zip(tensors((left.weights, left.aux)), tensors((right.weights, right.aux)), strict=True):
        assert torch.equal(a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from clmm.pretrain.kv_control import DeltaAdapter, NeuralAdapter, physical_state_bytes

    torch.set_num_threads(1)
    torch.manual_seed(193011)
    report = {"device": "cpu", "torch": torch.__version__, "checks": []}

    delta = DeltaAdapter()
    delta_state = delta.initial_state(2)
    mapping = [{}, {}]
    events = [([0, 1], [3, 4]), ([7, 12], [6, 1]), ([0, 1], [9, 8]), ([63, 62], [15, 14])]
    for keys, values in events:
        delta_state = delta.write(torch.tensor(keys), torch.tensor(values), delta_state)
        for sample in range(2):
            mapping[sample][keys[sample]] = values[sample]
    query = torch.arange(64)[None].repeat(2, 1)
    logits = delta.read(query, delta_state)
    expected = torch.zeros_like(logits)
    for sample in range(2):
        for key, value in mapping[sample].items():
            expected[sample, key, value] = 1
    assert torch.equal(logits, expected)
    assert physical_state_bytes(delta_state) == 2 * 64 * 64 * 4
    report["checks"].append("Delta logits exactly equal analytic slot-replacement result, including untouched keys and true revisions")
    report["delta_state_bytes_batch2"] = physical_state_bytes(delta_state)

    neural = NeuralAdapter()
    trainable = {name for name, parameter in neural.named_parameters() if parameter.requires_grad}
    assert trainable == {"engine.memory_model_parameters.0", "engine.memory_model_parameters.1"}
    report["checks"].append("Only the two initial neural-memory matrices are trainable offline")
    neural.requires_grad_(False)
    before = digest_parameters(neural)
    state = neural.initial_state(2)
    max_error = 0.0
    for keys, values in events:
        analytic = {name: [] for name in state.weights}
        for sample in range(2):
            parameters = {name: value[sample].detach().clone().requires_grad_(True)
                          for name, value in state.weights.items()}
            key = neural.codebook.keys[keys[sample]][None, None]
            value = neural.codebook.values[values[sample]][None, None]
            prediction = functional_call(neural.engine.memory_model, parameters, key)
            loss = 0.5 * (prediction - value).square().sum()
            gradient = torch.autograd.grad(loss, tuple(parameters.values()))
            for (name, parameter), grad in zip(parameters.items(), gradient, strict=True):
                analytic[name].append((parameter - neural.config.step_size * grad).detach())
        state = neural.write(torch.tensor(keys), torch.tensor(values), state)
        for name, rows in analytic.items():
            expected = torch.stack(rows)
            actual = state.weights[name]
            max_error = max(max_error, (actual - expected).abs().max().item())
            assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    report["checks"].append("Four vendored neural writes match independently differentiated half-sum-SSE SGD with current fast weights")
    report["neural_analytic_max_absolute_error"] = max_error

    snapshot = [tensor.clone() for tensor in tensors((state.weights, state.aux))]
    normal = neural.read(query, state)
    assert torch.equal(normal, neural.read(query, state))
    for previous, current in zip(snapshot, tensors((state.weights, state.aux)), strict=True):
        assert torch.equal(previous, current)
    report["checks"].append("Repeated all-key reads leave every adaptive-state tensor unchanged")

    for adapter, adaptive in ((delta, delta_state), (neural, state)):
        stream = io.BytesIO()
        torch.save(adapter.export_state(adaptive), stream)
        stream.seek(0)
        restored = adapter.import_state(torch.load(stream, weights_only=True))
        same_state(adaptive, restored)
        live_bytes = physical_state_bytes(adaptive)
        restored_bytes = physical_state_bytes(restored)
        assert restored_bytes <= live_bytes
        report[f"{adapter.method}_serialization_storage_bytes_batch2"] = {
            "live_before_export": live_bytes, "restored_compact": restored_bytes}
        assert torch.equal(adapter.read(query, adaptive), adapter.read(query, restored))
        new_keys, new_values = torch.tensor([19, 27]), torch.tensor([10, 11])
        continued = adapter.write(new_keys, new_values, adaptive)
        resumed = adapter.write(new_keys, new_values, restored)
        same_state(continued, resumed)
        assert torch.equal(adapter.read(query, continued), adapter.read(query, resumed))
        report["checks"].append(f"{adapter.method} torch save/load and continuation are bit-identical; compact clones do not increase backing storage")
    assert digest_parameters(neural) == before
    report["checks"].append("All registered slow neural parameters remain bit-identical during writes, reads, and serialization")
    report["neural_state_bytes_batch2"] = physical_state_bytes(state)
    report["source_sha256"] = hashlib.sha256((args.repo / "clmm/pretrain/kv_control.py").read_bytes()).hexdigest()
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
