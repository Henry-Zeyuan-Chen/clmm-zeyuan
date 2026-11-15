"""Static checkpoint/codebook identity audit; performs no model inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hash(tensor):
    header = json.dumps({"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + tensor.numpy().astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    key_hashes, initial_engines, tensors_by_hash, reports, evaluations = {}, {}, {}, [], []
    for path in sorted(args.root.glob("*/metadata.json")):
        if path.parent.name.startswith("smoke"):
            continue
        metadata = json.loads(path.read_text())
        if metadata["status"] != "succeeded":
            continue
        folder, rank, seed = path.parent, metadata["config"]["rank"], metadata["args"]["init_seed"]
        for filename in ("initial_checkpoint.pt", "checkpoint.pt"):
            checkpoint = folder / filename
            if not checkpoint.exists():
                continue
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert payload["format"] == "clmm.kv.correlated.v1" and payload["method"] == "neural_correlated"
            assert payload["source_commit"] == metadata["source_commit"]
            assert payload["config"] == metadata["config"]
            state = payload["model_state"]
            assert all(torch.isfinite(value).all() for value in state.values())
            keys, values = state["codebook.keys"], state["codebook.values"]
            assert keys.dtype == values.dtype == torch.float32
            assert keys.shape == (64, 64) and values.shape == (16, 64)
            assert torch.count_nonzero(keys[:, rank:]) == 0
            assert torch.allclose(keys.norm(dim=1), torch.ones(64), atol=2e-7, rtol=0)
            assert torch.unique(keys, dim=0).shape[0] == 64
            assert torch.equal(values, torch.eye(64)[:16])
            key_hash = code_hash(keys)
            assert key_hash == metadata["config"]["key_codebook_sha256"] == metadata["codebook_hash"] == payload["codebook_hash"]
            assert rank not in key_hashes or key_hashes[rank] == key_hash
            key_hashes[rank] = key_hash
            raw = torch.randn(64, 64, generator=torch.Generator().manual_seed(90501))
            expected = torch.zeros(64, 64)
            expected[:, :rank] = torch.nn.functional.normalize(raw[:, :rank], dim=-1)
            assert torch.allclose(keys, expected, atol=2e-7, rtol=1e-6)
            assert sum(value.numel() for name, value in state.items() if name.startswith("engine.")) == 32898
            digest = sha(checkpoint)
            tensors_by_hash[digest] = (state, payload, folder.name, filename)
            if filename == "initial_checkpoint.pt":
                assert payload["training"]["step"] == 0
                assert torch.count_nonzero(state["engine.memory_model_parameters.1"]) == 0
                engine = {name: value for name, value in state.items() if name.startswith("engine.")}
                if seed in initial_engines:
                    assert engine.keys() == initial_engines[seed].keys()
                    assert all(torch.equal(value, initial_engines[seed][name]) for name, value in engine.items())
                initial_engines[seed] = engine
            reports.append(dict(run=folder.name, checkpoint=filename, sha256=digest, rank=rank, init_seed=seed,
                                codebook_sha256=key_hash, nested_construction_max_error=float((keys - expected).abs().max())))
        if metadata["args"]["mode"] == "train":
            initial = torch.load(folder / "initial_checkpoint.pt", map_location="cpu", weights_only=True)["model_state"]
            final = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)["model_state"]
            changed = [name for name in initial if not torch.equal(initial[name], final[name])]
            assert set(changed) == {"engine.memory_model_parameters.0", "engine.memory_model_parameters.1"}
            assert torch.equal(initial["engine.memory_model_parameters.0"][:, rank:], final["engine.memory_model_parameters.0"][:, rank:])
        elif metadata.get("parent_sha256"):
            evaluations.append((folder.name, metadata))
    for name, metadata in evaluations:
        assert metadata["parent_sha256"] in tensors_by_hash
        _, payload, source_run, filename = tensors_by_hash[metadata["parent_sha256"]]
        assert payload["config"] == metadata["config"] and payload["training"] == metadata["parent_training"]
        if name.startswith("comparison-initial"):
            assert filename == "initial_checkpoint.pt" and payload["training"]["step"] == 0
        else:
            assert filename == "checkpoint.pt" and payload["training"]["step"] == 1200
    result = dict(status="passed", scope="Static saved tensors only; no model inference, training, or efficacy rerun",
                  checkpoints=reports, parent_checkpoint_references=len(evaluations),
                  codebooks={str(rank): digest for rank, digest in key_hashes.items()},
                  visited_ranks=sorted(key_hashes),
                  matched_same_seed_engine_initialization_across_ranks=True,
                  crossrank_scope="Saved ranks only; three-rank mechanism property independently checked before outcomes",
                  only_two_allowed_initial_memory_matrices_change_offline=True,
                  inactive_input_rows_remain_unchanged=True)
    (args.out / "checkpoint_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
