"""Geometry and literal scheduler checks; no learned outcome is selected here."""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from clmm.pretrain.kv_control import DeltaAdapter
from clmm.pretrain.kv_scheduler import StreamingReplay
from clmm.pretrain.kv_streams import evaluation_data, replay_plan, signature
from clmm.sleep import SleepSession
from clmm.sleep.multirevision import PRESSURE_CELLS, classify_events, make_datasets


@pytest.fixture(scope="module")
def datasets():
    return make_datasets(91123, episodes=3)


def run_live(ep, arm):
    policy = StreamingReplay(arm, ep["seed"], capacity=8, block_size=4)
    events = []
    for index, (key, value) in enumerate(zip(ep["keys"], ep["values"])):
        events.extend([policy.prime(key, value)] if index < ep["boundary"] else policy.step(key, value))
        assert policy.logical_counts()["buffer_pairs"] <= 8
        assert policy.logical_counts()["snapshot_pairs"] <= 4
    assert policy.finish() == []
    return events


def test_cell_names_short_gates_and_json_are_unchanged(datasets):
    assert tuple(datasets) == ("acquire2", "acquire4", "revise2", "gate_k16_r3", *PRESSURE_CELLS)
    assert all(len(items) == 3 for items in datasets.values())
    short = evaluation_data(91123, episodes=3, suite="gate")
    assert all(datasets[name] == episodes for name, episodes in short.items())
    assert json.loads(json.dumps(datasets)) == datasets


def test_deterministic_generation_preserves_all_global_rngs(datasets):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    assert make_datasets(91123, episodes=3) == datasets
    assert random.getstate() == python_state
    actual = np.random.get_state()
    assert actual[0] == numpy_state[0] and actual[2:] == numpy_state[2:]
    np.testing.assert_array_equal(actual[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


@pytest.mark.parametrize("name", PRESSURE_CELLS)
def test_three_genuine_noncycling_updates_and_truth_at_every_arrival(datasets, name):
    for ep in datasets[name]:
        assert ep["facts"] == ep["boundary"] == 16
        assert ep["rounds"] == 3 and ep["revisions"] == 6
        assert ep["delay"] == 56 + ep["retention_tail"]
        assert len(ep["keys"]) == 72 + ep["retention_tail"]
        assert len(set(ep["keys"][:16])) == 16
        assert set(ep["version_history"]) == {str(key) for key in ep["keys"][:16]}
        assert len(ep["keys"]) == len(ep["event_kinds"]) == len(ep["arrival_versions"])
        assert ep["signature"] == signature(ep["keys"], ep["values"])
        queries = ep["queries"][:6]
        assert [q["group"] for q in queries] == ["early", "early", "eligible", "eligible", "revised", "revised"]
        assert [q["key"] for q in queries] == [ep["keys"][i] for i in (0, 1, 12, 13, 14, 15)]
        truth, observed_versions = {}, {}
        for key, value, kind, version in zip(ep["keys"], ep["values"], ep["event_kinds"], ep["arrival_versions"]):
            assert type(key) is int and 0 <= key < 64 and type(value) is int and 0 <= value < 16
            if kind == "revision":
                assert value != truth[key]
                assert version == observed_versions[key] + 1
            elif kind == "refresh":
                assert value == truth[key] and version == observed_versions[key]
            else:
                assert version == 0
            truth[key], observed_versions[key] = value, version
            if len(truth) >= 16:
                assert len({truth[q["key"]] for q in queries}) == 6
        for query in queries:
            history = ep["version_history"][str(query["key"])]
            assert query["versions"] == history and len(set(history)) == len(history)
            assert query["target"] == truth[query["key"]] == history[-1]
            assert query["initial_value"] == history[0]
            revised = query["group"] == "revised"
            assert len(history) == (4 if revised else 1)
            assert query["old"] == (history[-2] if revised else -1)
            assert query["category"] == ("revised" if revised else "stable")
        assert len(ep["queries"]) == (7 if ep["retention_tail"] else 6)
        if ep["retention_tail"]:
            recent = ep["queries"][-1]
            assert recent["group"] == recent["category"] == "recent"
            assert (recent["key"], recent["target"]) == (ep["keys"][-1], ep["values"][-1])


def test_round_blocks_gaps_phase_pairing_and_fixed_distractors(datasets):
    for index in range(3):
        paired = [datasets[name][index] for name in PRESSURE_CELLS]
        base = paired[0]
        reference_distractors = None
        for ep in paired:
            assert ep["seed"] == base["seed"]
            assert ep["keys"][:16] == base["keys"][:16]
            assert ep["values"][:16] == base["values"][:16]
            assert ep["version_history"] == base["version_history"]
            assert ep["queries"][:6] == base["queries"][:6]
            for round_index in range(3):
                start = 16 + round_index * 24
                assert ep["event_kinds"][start:start + 4] == ["distractor", "distractor", "refresh", "refresh"]
                second = ["revision", "revision", "distractor", "distractor"] if ep["phase"] == 0 else ["distractor", "distractor", "revision", "revision"]
                assert ep["event_kinds"][start + 4:start + 8] == second
                assert ep["keys"][start + 2:start + 4] == base["keys"][14:16]
                assert ep["arrival_versions"][start + 2:start + 4] == [round_index] * 2
                if round_index < 2:
                    assert ep["event_kinds"][start + 8:start + 24] == ["distractor"] * 16
            distractors = [(key, value) for key, value, kind in zip(ep["keys"], ep["values"], ep["event_kinds"]) if kind == "distractor"]
            assert len(distractors) == 44 + ep["retention_tail"]
            assert not set(key for key, _ in distractors).intersection(ep["keys"][:16])
            table = {}
            for key, value in distractors:
                assert table.setdefault(key, value) == value
            if ep["retention_tail"] == 128:
                if reference_distractors is None:
                    reference_distractors = distractors
                else:
                    assert distractors == reference_distractors
        for phase in (0, 2):
            longest = datasets[f"pressure_k16_p{phase}_d128"][index]
            for tail in (0, 64):
                shorter = datasets[f"pressure_k16_p{phase}_d{tail}"][index]
                for field in ("keys", "values", "event_kinds", "arrival_versions"):
                    assert shorter[field] == longest[field][:72 + tail]


@pytest.mark.parametrize("arm", ["online", "repeat", "continuous", "periodic"])
@pytest.mark.parametrize("name", PRESSURE_CELLS)
def test_live_matches_old_causal_plan_and_exact_budgets(datasets, name, arm):
    ep = datasets[name][0]
    events = run_live(ep, arm)
    plan = replay_plan(ep, arm, capacity=8, block=4)
    assert events == plan["events"]
    assert len(events) == (72 + ep["retention_tail"] if arm == "online" else 128 + 2 * ep["retention_tail"])
    for event in events:
        assert event["source"] <= event["after_online"]
        assert (event["key"], event["value"]) == (ep["keys"][event["source"]], ep["values"][event["source"]])
    # A live policy received only key/value literals and its seed, not labels.
    audited = classify_events(ep, events)
    assert len(audited) == len(events)
    assert not any(row["stale"] for row in audited if row["kind"] == "online")


def test_continuous_periodic_literal_extras_match_and_only_emission_times_change(datasets):
    for name in PRESSURE_CELLS:
        ep = datasets[name][0]
        c, p = (replay_plan(ep, arm) for arm in ("continuous", "periodic"))
        assert c["selected"] == p["selected"]
        ce, pe = ([event for event in plan["events"] if event["kind"] == "extra"] for plan in (c, p))
        assert [(e["source"], e["key"], e["value"]) for e in ce] == [(e["source"], e["key"], e["value"]) for e in pe]
        assert all(a["after_online"] <= b["after_online"] for a, b in zip(ce, pe))
        assert any(a["after_online"] != b["after_online"] for a, b in zip(ce, pe))


def test_fixed_seed_exposes_staleness_difference_due_only_to_timing():
    # This seed is a fixed scheduler audit, not selected for a learned effect.
    ep = make_datasets(0, episodes=1)["pressure_k16_p2_d0"][0]
    c, p = ([row for row in classify_events(ep, run_live(ep, arm)) if row["kind"] == "extra"]
            for arm in ("continuous", "periodic"))
    changed = [(a, b) for a, b in zip(c, p) if a["stale"] != b["stale"]]
    assert changed
    for a, b in changed:
        assert (a["source"], a["key"], a["value"]) == (b["source"], b["key"], b["value"])
        assert a["after_online"] < b["after_online"]
        assert not a["stale"] and b["stale"]


def test_neutral_gate_is_identical_online_only_prefill_for_all_arms(datasets):
    ep = datasets["gate_k16_r3"][0]
    pressure = datasets["pressure_k16_p0_d0"][0]
    assert ep["keys"][:16] == pressure["keys"][:16]
    assert ep["values"][:16] == pressure["values"][:16]
    assert ep["version_history"] == pressure["version_history"]
    assert ep["queries"] == pressure["queries"]
    assert ep["boundary"] == len(ep["keys"]) == 22 and ep["delay"] == 0
    assert ep["event_kinds"] == ["initial"] * 16 + ["revision"] * 6
    assert ep["arrival_versions"] == [0] * 16 + [1, 1, 2, 2, 3, 3]
    assert not ep["pressure"] and ep["phase"] is None
    states = []
    model = DeltaAdapter()
    for arm in ("online", "repeat", "continuous", "periodic"):
        session = SleepSession(model, 1, arm, [ep["seed"]])
        events = []
        for key, value in zip(ep["keys"], ep["values"]):
            events.extend(session.prime(torch.tensor([key]), torch.tensor([value]))[0])
        assert session.finish() == [[]]
        assert events == replay_plan(ep, arm)["events"]
        assert session.budget()["total_writes"] == 22
        states.append(session.state)
    for state in states[1:]:
        assert all(torch.equal(value, state.weights[key]) for key, value in states[0].weights.items())


@pytest.mark.parametrize("excluded_cell", ["gate_k16_r3", *PRESSURE_CELLS])
def test_any_excluded_whole_history_rejects_entire_paired_base(datasets, excluded_cell):
    excluded = {datasets[excluded_cell][0]["signature"]}
    fresh = make_datasets(91123, episodes=2, seen=excluded)
    new_names = ("gate_k16_r3", *PRESSURE_CELLS)
    for name in new_names:
        assert fresh[name] == datasets[name][1:3]
    signatures = [ep["signature"] for items in fresh.values() for ep in items]
    assert len(signatures) == len(set(signatures))
    assert not excluded.intersection(signatures)


def test_excluded_short_histories_are_not_reused(datasets):
    excluded = {ep["signature"] for name in ("acquire2", "acquire4", "revise2") for ep in datasets[name]}
    fresh = make_datasets(91123, episodes=3, seen=excluded)
    assert not excluded.intersection(ep["signature"] for items in fresh.values() for ep in items)


def test_observer_tracks_real_truth_and_every_old_version_without_mutating_inputs(datasets):
    ep = datasets["pressure_k16_p0_d0"][0]
    query = ep["queries"][4]
    sources = [index for index, key in enumerate(ep["keys"]) if key == query["key"]
               and ep["event_kinds"][index] in ("initial", "revision")]
    assert len(sources) == 4
    events = [dict(source=source, key=query["key"], value=ep["values"][source], kind="extra",
                   after_online=len(ep["keys"]) - 1) for source in sources]
    original_ep, original_events = deepcopy(ep), deepcopy(events)
    classified = classify_events(ep, events)
    assert [row["stale"] for row in classified] == [True, True, True, False]
    assert [row["obsolete_anyquery"] for row in classified] == [True, True, True, False]
    assert all(row["latest_online"] == query["target"] for row in classified)
    assert ep == original_ep and events == original_events
    malformed = deepcopy(events)
    malformed[0]["value"] = (malformed[0]["value"] + 1) % 16
    with pytest.raises(ValueError, match="literal"):
        classify_events(ep, malformed)


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"seed": True}, {"seed": 1.0},
                                    {"seed": 1, "episodes": 0}, {"seed": 1, "episodes": True}])
def test_invalid_dataset_configuration(kwargs):
    with pytest.raises(ValueError):
        make_datasets(**kwargs)
