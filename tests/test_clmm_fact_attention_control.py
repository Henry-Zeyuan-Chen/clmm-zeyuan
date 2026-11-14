"""Paired-input, exact-initialization and bounded diagnostic contracts."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clmm.pretrain import ByteLMConfig, build_model, load_checkpoint, reset_memory_at_segments, save_checkpoint
from clmm.pretrain.fact_streams import KEY_POOLS, encode_example, make_episode, render_query
from scripts import fact_attention_control as control


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def parent(tmp_path):
    config = ByteLMConfig(dim=16, depth=1, dim_head=8, heads=2, ff_mult=2,
                          memory_layers=(1,), memory_dim_head=8, memory_heads=2,
                          memory_depth=1, memory_expansion_factor=1, num_persist_mem_tokens=0)
    with torch.random.fork_rng():
        torch.manual_seed(19)
        model = build_model(config)
    path = tmp_path / "parent.pt"
    save_checkpoint(path, model, config, source_info={"fixture": True}, training_info={"init_seed": 68101})
    return path, model, config


def arguments(path, out, segment_len=32, **changes):
    args = control.parser().parse_args(["--checkpoint", str(path), "--out", str(out),
                                       "--segment-len", str(segment_len)])
    for name, value in changes.items():
        setattr(args, name, value)
    return args


def test_cli_defaults_match_prospective_control(parent, tmp_path):
    args = arguments(parent[0], tmp_path / "new", 128)
    assert (args.steps, args.batch_size, args.train_seed, args.dev_seed) == (1000, 16, 78122, 78123)
    assert (args.lr, args.threads, args.max_seconds, args.gate_episodes) == (.0005, 2, 600, 64)
    assert (args.mode, args.suite, args.block_size, args.buffer_size) == ("attention_control", "gate", 4, 8)
    control.validate_args(args)


def test_control_changes_only_geometry_and_preserves_all_initial_tensors(parent):
    path, original, original_config = parent
    global_rng = torch.random.get_rng_state().clone()
    models = []
    for segment_len in (32, 128):
        model, config, metadata = control.build_control_model(path, segment_len)
        models.append(model)
        differing = [name for name, value in asdict(config).items() if value != asdict(original_config)[name]]
        assert differing == ([] if segment_len == 32 else ["segment_len"])
        assert config.segment_len == model.segment_len == segment_len
        assert config.memory_segment_len == 8 and config.memory_batch_size == 32
        assert metadata["initial_state_tensors_identical"]
        assert metadata["parameter_count"] == metadata["parent_parameter_count"]
        assert metadata["trained_source_block_size"] == segment_len
        assert metadata["record_bytes"] == 32 and metadata["training_input_length"] == 95
        assert metadata["local_attention_contains_entire_training_example"] == (segment_len == 128)
        for name, value in original.state_dict().items():
            assert torch.equal(value, model.state_dict()[name])
    for name, value in models[0].state_dict().items():
        assert torch.equal(value, models[1].state_dict()[name])
    assert torch.equal(global_rng, torch.random.get_rng_state())


@pytest.mark.parametrize("training_info", [{"init_seed": 68105}, {"task": "fact-revision"},
                                           {"init_seed": 68101, "parent_checkpoint_sha256": "adapted"}])
def test_other_seed_and_adapted_checkpoints_are_rejected(parent, tmp_path, training_info):
    _, model, config = parent
    path = tmp_path / "other.pt"
    save_checkpoint(path, model, config, source_info={"fixture": True}, training_info=training_info)
    with pytest.raises(ValueError, match="original primary shared checkpoint"):
        control.build_control_model(path, 128)


def test_training_data_are_paired_geometry_independent_and_reconstructable():
    data32 = control.make_training_inputs(3, 16, 78122)
    data128 = control.make_training_inputs(3, 16, 78122)
    assert data32["input_bytes"].shape == (3, 16, 95)
    assert data32["targets"].shape == (3, 16)
    assert data32["input_bytes"].dtype == data32["targets"].dtype == np.uint8
    assert set(data32["query_indices"].reshape(-1)) == {0, 1}
    assert control.training_data_sha256(data32) == control.training_data_sha256(data128)
    for name in data32:
        np.testing.assert_array_equal(data32[name], data128[name])
    for step in range(3):
        for index in range(16):
            episode = make_episode(int(data32["episode_seeds"][step, index]), 2, 0, "train",
                                   revision_fraction=0., include_recent=False)
            query = episode.queries[int(data32["query_indices"][step, index])]
            prefix, target, _ = encode_example(episode, query, "online")
            assert data32["input_bytes"][step, index].tobytes() == prefix
            assert int(data32["targets"][step, index]) == target
            assert prefix[-31:] == render_query(query)
            assert len(episode.events) == 2
            assert all(event.kind == "fact" and event.version == 1 and event.key in KEY_POOLS["train"]
                       for event in episode.events)
    other = control.make_training_inputs(3, 16, 78124)
    assert control.training_data_sha256(other) != control.training_data_sha256(data32)


@pytest.mark.parametrize("segment_len", (32, 128))
def test_two_step_training_saves_full_inputs_optimizer_and_exact_final_checkpoint(parent, tmp_path, segment_len):
    path, original, _ = parent
    out = tmp_path / f"arm{segment_len}"
    out.mkdir()
    args = arguments(path, out, segment_len, steps=2, batch_size=2, gate_episodes=1)
    model, config, _ = control.build_control_model(path, segment_len)
    data = control.make_training_inputs(args.steps, args.batch_size, args.train_seed)
    x = torch.tensor(data["input_bytes"][0].astype(np.int64))
    y = torch.tensor(data["targets"][0].astype(np.int64))
    with torch.no_grad():
        expected_logits = model(x)[:, -1]
        expected_full_loss = float(F.cross_entropy(expected_logits, y))
        forced_loss = float(F.cross_entropy(expected_logits[:, ord("a"):ord("p") + 1], y - ord("a")))
    assert expected_full_loss > forced_loss + 1.
    metadata = control.train_control(model, config, args, out, torch.device("cpu"), time.perf_counter() + 60, data)
    assert metadata["completed_steps"] == 2 and metadata["supervised_answers"] == 4
    assert metadata["input_bytes"] == 4 * 95
    assert metadata["output_checkpoint_sha256"] == control.common.sha256(out / "checkpoint.pt")
    with np.load(out / "training_inputs.npz") as saved:
        for name in data:
            np.testing.assert_array_equal(saved[name], data[name])
    plan = json.loads((out / "training_plan.json").read_text())
    assert len(plan) == 2 and all(step["facts"] == 2 and step["delay"] == 0 for step in plan)
    with (out / "training.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [1, 2]
    assert float(rows[0]["loss"]) == pytest.approx(expected_full_loss, abs=1e-6)
    assert all(float(row["gradient_norm"]) > 0 for row in rows)
    assert all(int(row["attention_segment_len"]) == segment_len for row in rows)
    optimizer = torch.load(out / "optimizer.pt", weights_only=True)
    assert optimizer["param_groups"][0]["lr"] == .0005
    assert optimizer["param_groups"][0]["weight_decay"] == .01
    assert all(int(state["step"]) == 2 for state in optimizer["state"].values())
    restored, restored_config, payload = load_checkpoint(out / "checkpoint.pt")
    assert restored_config == config
    assert payload["training_info"]["source_block_size"] == segment_len
    assert payload["training_info"]["parent_checkpoint_sha256"] == control.common.sha256(path)
    assert payload["training_info"]["fixed_final_checkpoint"] is True
    assert payload["training_info"]["answer_vocabulary"] == 256
    assert any(not torch.equal(value, original.state_dict()[name]) for name, value in restored.state_dict().items())
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    with torch.no_grad():
        model.eval()
        restored.eval()
        torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)


def test_expired_budget_preserves_inputs_optimizer_partial_state_and_step_count(parent, tmp_path):
    out = tmp_path / "expired"
    out.mkdir()
    args = arguments(parent[0], out, steps=2, batch_size=2)
    model, config, _ = control.build_control_model(parent[0], 32)
    data = control.make_training_inputs(2, 2, args.train_seed)
    with pytest.raises(TimeoutError):
        control.train_control(model, config, args, out, torch.device("cpu"), 0, data)
    assert not (out / "checkpoint.pt").exists()
    assert (out / "training_inputs.npz").exists() and (out / "training_plan.json").exists()
    assert (out / "optimizer.pt").exists() and (out / "partial_state.pt").exists()
    assert json.loads((out / "training_progress.json").read_text()) == {
        "completed_steps": 0, "planned_steps": 2, "fixed_final_checkpoint_saved": False}


def test_common_gate_data_and_counterfactuals_are_identical_for_both_windows(parent, tmp_path):
    datasets = []
    for segment_len in (32, 128):
        out = tmp_path / f"evaluation{segment_len}"
        out.mkdir()
        args = arguments(parent[0], out, segment_len, gate_episodes=1)
        datasets.append(control.common.build_evaluation(args, out))
        labels = control.label_reset_diagnostic(datasets[-1], segment_len)
        assert len(labels) == 8
        assert all(label["reset_expected_noop"] == (segment_len == 128) for label in labels)
    assert (tmp_path / "evaluation32/evaluation_data.jsonl.gz").read_bytes() == (
        tmp_path / "evaluation128/evaluation_data.jsonl.gz").read_bytes()
    for before, after in zip(datasets[0], datasets[1]):
        assert before["episodes"] == after["episodes"]
        assert before["samples"] == after["samples"]
    seen = {dataset["spec"]["cell"]: dataset for dataset in datasets[0] if dataset["spec"]["family"] == "seen"}
    regular, shifted = seen["acquire_k2"], seen["counterfactual_k2"]
    for before, after in zip(regular["samples"], shifted["samples"]):
        assert before["row"]["key"] == after["row"]["key"]
        assert before["row"]["target"] != after["row"]["target"]
        assert before["prefix"][-31:] == after["prefix"][-31:]


def test_reset_is_exact_noop_at_128_and_full_vocabulary_evaluator_is_reused(parent, tmp_path):
    model, _, _ = control.build_control_model(parent[0], 128)
    model.eval()
    data = control.make_training_inputs(1, 2, 78122)
    x = torch.tensor(data["input_bytes"][0].astype(np.int64))
    with torch.no_grad():
        expected = model(x)
        with reset_memory_at_segments(model):
            actual = model(x)
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    out = tmp_path / "eval"
    out.mkdir()
    args = arguments(parent[0], out, 128, gate_episodes=1, eval_batch_size=4)
    datasets = control.common.build_evaluation(args, out)
    metadata = control.common.evaluate(model, datasets, args, out, torch.device("cpu"), time.perf_counter() + 60)
    assert metadata["checkpoint_unchanged"] and metadata["evaluation_queries"] == 24
    assert np.load(out / "answer_logits.npy").shape == (24, 256)
    logits = torch.zeros(1, 256)
    logits[0, ord("a")], logits[0, ord("!")] = 2., 20.
    scored = control.common.score_logits(logits, torch.tensor([ord("a")]))
    assert scored["prediction"].item() == ord("!")
    assert scored["forced_prediction"].item() == ord("a")
    assert scored["nll"].item() > 15


@pytest.mark.parametrize("changes", [{"segment_len": 64}, {"suite": "full"}, {"lr": .001},
                                       {"steps": 0}, {"batch_size": 0}, {"block_size": 8},
                                       {"buffer_size": 4}, {"train_seed": -1}, {"dev_seed": True}])
def test_control_rejects_protocol_changes(parent, tmp_path, changes):
    args = arguments(parent[0], tmp_path / "new")
    for name, value in changes.items():
        setattr(args, name, value)
    with pytest.raises(ValueError):
        control.validate_args(args)


def test_existing_output_is_not_overwritten(parent, tmp_path):
    args = arguments(parent[0], tmp_path)
    with pytest.raises(FileExistsError, match="NEW directory"):
        control.run(args)
