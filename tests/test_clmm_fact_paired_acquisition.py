"""Data contracts for paired-key acquisition and its duplicate-query control."""

from collections import Counter
from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch

from clmm.pretrain.fact_streams import (
    Episode, KEY_POOLS, QUERY_PREFIX_BYTES, RECORD_BYTES, VALUES,
    encode_example, make_episode, render_query,
)
from scripts import fact_paired_acquisition as paired


def decoded_episodes(plan):
    """Validate the saved plan against the independent v1 episode validator."""
    assert isinstance(plan["episodes"], list)
    episodes = [Episode.from_dict(data) for data in plan["episodes"]]
    assert [episode.to_dict() for episode in episodes] == plan["episodes"]
    json.dumps(plan, allow_nan=False)
    return episodes


def batch_signature(plan):
    episodes = decoded_episodes(plan)
    signatures = {(episode.facts, episode.revision_fraction) for episode in episodes}
    assert len(signatures) == 1, "A rectangular batch must use one curriculum task"
    return signatures.pop()


@pytest.mark.parametrize("facts,fraction", [(1, 0.), (2, 0.), (1, 1.), (2, .5)])
@pytest.mark.parametrize("split,family", [("train", "seen"), ("dev", "seen"), ("dev", "heldout")])
def test_episode_generation_keeps_v1_records_revisions_and_latest_targets(facts, fraction, split, family):
    for seed in range(24):
        episode = paired.episode_for_seed(seed, facts, fraction, split=split, family=family)
        original = make_episode(episode.seed, facts, 0, split, template_family=family,
                                revision_fraction=fraction, include_recent=False)
        assert episode.events == original.events
        assert episode.facts == facts and episode.revision_fraction == fraction
        assert episode.delay == 0 and not episode.include_recent
        assert len(episode.queries) == facts
        assert len({query.template_index for query in episode.queries}) == 1
        for query, expected in zip(episode.queries, original.queries):
            assert replace(query, template_index=expected.template_index) == expected
        if facts == 2:
            assert len({event.value for event in episode.events[:facts]}) == 2
        assert Episode.from_dict(episode.to_dict()) == episode


def test_initial_value_collisions_are_rejected_without_rewriting_v1_facts():
    collision_seed = next(seed for seed in range(256)
                          if len({record.value for record in make_episode(
                              seed, 2, 0, "train", revision_fraction=0.,
                              include_recent=False).events}) == 1)
    episode = paired.episode_for_seed(collision_seed, 2, 0.)
    assert episode == paired.episode_for_seed(collision_seed, 2, 0.)
    assert episode.seed != collision_seed
    assert len({record.value for record in episode.events[:2]}) == 2
    assert episode.events == make_episode(episode.seed, 2, 0, "train", revision_fraction=0.,
                                         include_recent=False).events


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_episode_keys_stay_in_their_fixed_split_namespace(split):
    for seed in range(32):
        episode = paired.episode_for_seed(seed, 2, .5, split=split)
        assert {record.key for record in episode.events} <= set(KEY_POOLS[split])
        assert {query.key for query in episode.queries} <= set(KEY_POOLS[split])
        assert episode.split == split
    with pytest.raises(ValueError):
        paired.episode_for_seed(1, 2, 0., split="train", family="heldout")


def test_template_family_changes_only_surface_forms_and_never_split_or_values():
    for seed in range(24):
        seen = paired.episode_for_seed(seed, 2, .5, split="dev", family="seen")
        heldout = paired.episode_for_seed(seed, 2, .5, split="dev", family="heldout")
        assert seen.seed == heldout.seed
        for left, right in zip(seen.events, heldout.events):
            assert replace(left, template_family="heldout") == right
        for left, right in zip(seen.queries, heldout.queries):
            assert replace(left, template_family="heldout") == right
            assert render_query(left) != render_query(right)


@pytest.mark.parametrize("step", [0, 124, 125, 749, 750, 999])
def test_arms_share_base_episodes_context_lengths_and_answer_budgets(step):
    batch_size = 16
    px, py, pp = paired.make_batch(step, 1000, batch_size, 78122, "paired")
    dx, dy, dp = paired.make_batch(step, 1000, batch_size, 78122, "duplicate")
    assert pp["episodes"] == dp["episodes"]
    assert pp["stage"] == dp["stage"]
    assert {key: value for key, value in pp.items() if key != "query_indices"} == {
        key: value for key, value in dp.items() if key != "query_indices"}
    assert isinstance(pp["stage"], str) and pp["stage"]
    assert px.shape == dx.shape and px.ndim == 2 and px.shape[0] == batch_size
    assert py.shape == dy.shape == (batch_size,)
    assert px.dtype == dx.dtype == py.dtype == dy.dtype == np.uint8
    assert len(pp["query_indices"]) == len(dp["query_indices"]) == batch_size
    episodes = decoded_episodes(pp)
    assert len(episodes) == batch_size // 2
    assert len({episode.seed for episode in episodes}) == batch_size // 2
    for pair_index, episode in enumerate(episodes):
        assert episode.split == "train" and episode.template_family == "seen"
        assert episode.delay == 0 and not episode.include_recent
        context = b"".join(record.encoded for record in episode.events)
        assert px.shape[1] == len(context) + QUERY_PREFIX_BYTES
        assert len(context) == len(episode.events) * RECORD_BYTES
        start = 2 * pair_index
        for x, y, plan in ((px, py, pp), (dx, dy, dp)):
            for row in (start, start + 1):
                query_index = plan["query_indices"][row]
                assert type(query_index) is int and 0 <= query_index < episode.facts
                query = episode.queries[query_index]
                prefix, target, _ = encode_example(episode, query, "online")
                assert x[row].tobytes() == prefix
                assert int(y[row]) == target
                assert x[row, :-QUERY_PREFIX_BYTES].tobytes() == context
        np.testing.assert_array_equal(dx[start], dx[start + 1])
        assert dy[start] == dy[start + 1]
        assert dp["query_indices"][start] == dp["query_indices"][start + 1]
        if episode.facts == 1:
            assert pp["query_indices"][start:start + 2] == [0, 0]
            np.testing.assert_array_equal(px[start:start + 2], dx[start:start + 2])
            np.testing.assert_array_equal(py[start:start + 2], dy[start:start + 2])
        else:
            assert pp["query_indices"][start:start + 2] == [0, 1]


def test_unrevised_pairs_differ_only_by_query_key_and_supervise_both_distinct_values():
    x, y, plan = paired.make_batch(125, 1000, 64, 78122, "paired")
    for pair_index, episode in enumerate(decoded_episodes(plan)):
        row = pair_index * 2
        left, right = episode.queries
        assert left.key != right.key and left.target != right.target
        assert y[row] != y[row + 1]
        np.testing.assert_array_equal(x[row, :-QUERY_PREFIX_BYTES], x[row + 1, :-QUERY_PREFIX_BYTES])
        left_text, right_text = x[row, -QUERY_PREFIX_BYTES:].tobytes(), x[row + 1, -QUERY_PREFIX_BYTES:].tobytes()
        assert left_text != right_text
        assert left_text.replace(left.key.encode("ascii"), b"###") == right_text.replace(
            right.key.encode("ascii"), b"###")
        for query, text in ((left, left_text), (right, right_text)):
            changed_target = next(value for value in VALUES if value != query.target)
            assert render_query(replace(query, target=changed_target)) == text


def test_duplicate_control_selects_both_query_positions_without_positional_bias():
    _, _, plan = paired.make_batch(125, 1000, 512, 78122, "duplicate")
    selected = plan["query_indices"][::2]
    assert set(selected) == {0, 1}
    assert .3 < selected.count(0) / len(selected) < .7


def test_curriculum_stage_boundaries_and_final_mixture():
    plans = {step: paired.make_batch(step, 1024, 2, 78122, "paired")[2]
             for step in (0, 127, 128, 767, 768, 1023)}
    assert batch_signature(plans[0]) == batch_signature(plans[127]) == (1, 0.)
    assert batch_signature(plans[128]) == batch_signature(plans[767]) == (2, 0.)
    assert plans[0]["stage"] == plans[127]["stage"]
    assert plans[128]["stage"] == plans[767]["stage"]
    assert plans[127]["stage"] != plans[128]["stage"]
    assert plans[767]["stage"] != plans[768]["stage"]
    counts = Counter(batch_signature(paired.make_batch(step, 1024, 2, 78122, "paired")[2])
                     for step in range(768, 1024))
    assert set(counts) == {(1, 0.), (2, 0.), (1, 1.), (2, .5)}
    # A fixed deterministic sample catches missing or severely biased tasks.
    assert all(32 < count < 96 for count in counts.values())


def test_batch_and_episode_generation_are_deterministic_and_ignore_global_rngs():
    expected = paired.make_batch(750, 1000, 16, 78122, "duplicate")
    expected_episode = paired.episode_for_seed(37, 2, .5)
    python_before, numpy_before = random.getstate(), np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    try:
        random.seed(739)
        np.random.seed(419)
        torch.manual_seed(113)
        python_seeded, numpy_seeded = random.getstate(), np.random.get_state()
        torch_seeded = torch.random.get_rng_state().clone()
        actual = paired.make_batch(750, 1000, 16, 78122, "duplicate")
        assert paired.episode_for_seed(37, 2, .5) == expected_episode
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[2] == expected[2]
        assert random.getstate() == python_seeded
        numpy_after = np.random.get_state()
        assert numpy_after[0] == numpy_seeded[0] and numpy_after[2:] == numpy_seeded[2:]
        np.testing.assert_array_equal(numpy_after[1], numpy_seeded[1])
        assert torch.equal(torch.random.get_rng_state(), torch_seeded)
    finally:
        random.setstate(python_before)
        np.random.set_state(numpy_before)
        torch.random.set_rng_state(torch_before)
    other = paired.make_batch(750, 1000, 16, 78124, "duplicate")
    assert other[2]["episodes"] != expected[2]["episodes"]
    assert not np.array_equal(other[0], expected[0])


@pytest.mark.parametrize("arm", ["paired", "duplicate"])
@pytest.mark.parametrize("batch_size", [1, 3, 15])
def test_odd_batch_sizes_are_rejected_before_generating_examples(arm, batch_size):
    with pytest.raises(ValueError):
        paired.make_batch(0, 1000, batch_size, 78122, arm)
