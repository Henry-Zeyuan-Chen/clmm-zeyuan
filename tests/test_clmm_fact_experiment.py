"""Checks for answer-only scoring and the paired experimental manifest."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fact_runner", ROOT / "scripts/fact_revision_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_forced_choice_does_not_hide_invalid_full_vocabulary_prediction():
    logits = torch.zeros(1, 256)
    logits[0, ord("a")] = 2
    logits[0, ord("!")] = 20
    result = runner.score_logits(logits, torch.tensor([ord("a")]))
    assert result["prediction"].item() == ord("!")
    assert result["forced_prediction"].item() == ord("a")
    assert result["legal_probability_mass"].item() < 1e-6
    assert result["nll"].item() > 15


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_answer_logits_rejected(bad):
    logits = torch.zeros(1, 256)
    logits[0, 0] = bad
    with pytest.raises(RuntimeError, match="Invalid answer logits"):
        runner.score_logits(logits, torch.tensor([97]))


def test_counterfactual_keeps_query_text_but_changes_every_target():
    episode = runner.make_episode(934, 8, 4, "dev")
    swapped = runner.counterfactual_episode(episode)
    from clmm.pretrain.fact_streams import render_query
    for before, after in zip(episode.queries, swapped.queries):
        assert render_query(before) == render_query(after)
        assert before.target != after.target
        assert before.key == after.key
        if before.old_value:
            assert before.old_value != after.old_value
    assert all(a.value != b.value for a, b in zip(episode.events, swapped.events))


def test_manifest_pairs_arms_and_delays_and_excludes_answer(tmp_path):
    args = SimpleNamespace(dev_seed=78103, gate_episodes=2, grid_episodes=2,
                           primary_episodes=2, suite="full", block_size=4, buffer_size=8)
    datasets = runner.build_evaluation(args, tmp_path)
    short = next(d for d in datasets if d["spec"]["cell"] == "acquire_k2" and d["spec"]["family"] == "seen")
    assert len(short["samples"]) == 8
    for sample in short["samples"]:
        assert len(sample["prefix"]) == 95
        assert sample["prefix"].endswith((b": ", b"= "))
        assert sample["row"]["extra_records"] == 0
    pressure = {d["spec"]["delay"]: d for d in datasets if d["spec"]["facts"] == 8 and d["spec"]["cell"].startswith("pressure")}
    for index in range(2):
        zero = pressure[0]["episodes"][index]
        long = pressure[32]["episodes"][index]
        assert zero["events"] == long["events"][:12]
    for dataset in pressure.values():
        samples = dataset["samples"]
        for episode in range(2):
            for category in ("stable", "revised"):
                paired = {s["row"]["condition"]: s for s in samples
                          if s["row"]["episode"] == episode and s["row"]["category"] == category}
                c, p, r = [paired[condition] for condition in ("continuous", "periodic", "repeat_current")]
                assert c["row"]["input_length"] == p["row"]["input_length"] == r["row"]["input_length"]
                assert c["audit"]["selected_replay_event_ids"] == p["audit"]["selected_replay_event_ids"]
                assert c["audit"]["query_prefix"] == p["audit"]["query_prefix"]
    repeated = tmp_path / "repeated"
    repeated.mkdir()
    runner.build_evaluation(args, repeated)
    assert (tmp_path / "evaluation_data.jsonl.gz").read_bytes() == (repeated / "evaluation_data.jsonl.gz").read_bytes()
    assert (tmp_path / "evaluation_data_canonical.sha256").read_text() == (repeated / "evaluation_data_canonical.sha256").read_text()


def test_training_curriculum_is_fixed_by_progress_and_seed():
    a, b = np.random.default_rng(22), np.random.default_rng(22)
    assert [runner.training_spec(i, 100, a) for i in range(100)] == [runner.training_spec(i, 100, b) for i in range(100)]
    assert runner.training_spec(0, 100, a) == (1, 0, 0.)


def test_expired_budget_stops():
    with pytest.raises(TimeoutError):
        runner.before_deadline(0)
