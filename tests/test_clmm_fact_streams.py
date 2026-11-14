"""Causal replay and byte-layout contracts independent of task accuracy."""

from dataclasses import replace
import hashlib
import json
import random

import pytest

from clmm.pretrain.fact_streams import (
    CONDITIONS, KEY_POOL_HASHES, KEY_POOLS, QUERY_PREFIX_BYTES, RECORD_BYTES,
    VALUES, Episode, FactQuery, FactRecord, encode_example, make_episode,
    render_query, schedule_episode,
)


def test_key_pools_are_disjoint_complete_and_share_all_digit_patterns():
    pools = {split: set(pool) for split, pool in KEY_POOLS.items()}
    assert {split: len(pool) for split, pool in pools.items()} == {"train": 600, "dev": 200, "test": 200}
    assert not pools["train"] & pools["dev"]
    assert not pools["train"] & pools["test"]
    assert not pools["dev"] & pools["test"]
    assert set.union(*pools.values()) == {f"{number:03d}" for number in range(1000)}
    for split, pool in KEY_POOLS.items():
        for position in range(3):
            assert {key[position] for key in pool} == set("0123456789")
        assert hashlib.sha256("\n".join(pool).encode("ascii")).hexdigest() == KEY_POOL_HASHES[split]
    ordered = KEY_POOLS["train"] + KEY_POOLS["dev"] + KEY_POOLS["test"]
    ranks = [hashlib.sha256(("clmm.fact.keys.v1:" + key).encode("ascii")).digest() for key in ordered]
    assert ranks == sorted(ranks)


@pytest.mark.parametrize("split", KEY_POOLS)
def test_full_capacity_episodes_use_only_their_namespace(split):
    pool_size = len(KEY_POOLS[split])
    episode = make_episode(11, 3, pool_size - 3, split)
    distinct_keys = {record.key for record in episode.events}
    assert distinct_keys == set(KEY_POOLS[split])
    assert len(episode.events) == pool_size + 1
    with pytest.raises(ValueError, match="capacity"):
        make_episode(11, 3, pool_size - 2, split)


def test_generation_is_deterministic_without_global_rng_or_persistent_assignments():
    global_state = random.getstate()
    first = make_episode(17, 200, 0, "dev")
    assert first == make_episode(17, 200, 0, "dev")
    first_assignments = {record.key: record.value for record in first.events[:200]}
    second_assignments = {record.key: record.value for record in make_episode(18, 200, 0, "dev").events[:200]}
    assert sum(first_assignments[key] != second_assignments[key] for key in first_assignments) > 150
    assert set(first_assignments.values()) == set(VALUES)
    for condition in CONDITIONS:
        schedule_episode(first, condition)
    assert random.getstate() == global_state


@pytest.mark.parametrize("revision_fraction, expected_revisions", [(0, 0), (0.5, 3), (1, 7)])
def test_targets_use_latest_online_facts_and_distractors_have_new_keys(revision_fraction, expected_revisions):
    episode = make_episode(29, 7, 11, "dev", revision_fraction=revision_fraction)
    initial = {record.key: record for record in episode.events[:7]}
    revisions = [record for record in episode.events if record.kind == "revision"]
    distractors = [record for record in episode.events if record.kind == "distractor"]
    assert len(revisions) == expected_revisions
    assert len({record.key for record in revisions}) == expected_revisions
    assert len({record.key for record in distractors}) == 11
    assert not set(initial) & {record.key for record in distractors}
    for record in revisions:
        assert record.value != initial[record.key].value
        assert record.version == 2
    latest = {record.key: record for record in episode.events}
    assert len(episode.queries) == 8
    assert [query.key for query in episode.queries[:7]] == list(initial)
    for query in episode.queries:
        assert query.target == latest[query.key].value
        assert query.latest_event_id == latest[query.key].event_id
        if query.category == "revised":
            assert query.old_value == initial[query.key].value != query.target
        else:
            assert query.old_value is None
    assert episode.queries[-1].category == "recent"
    assert episode.queries[-1].latest_event_id == len(episode.events) - 1


def test_optional_recent_query_and_zero_delay():
    assert len(make_episode(3, 5, 9, "train", include_recent=False).queries) == 5
    episode = make_episode(3, 5, 0, "train", revision_fraction=0)
    assert len(episode.events) == len(episode.queries) == 5
    assert {query.category for query in episode.queries} == {"stable"}


@pytest.mark.parametrize("split", KEY_POOLS)
def test_paired_delay_conditions_share_fact_revision_and_distractor_prefixes(split):
    episodes = [make_episode(491, 7, delay, split) for delay in (0, 8, 32, 100)]
    for shorter, longer in zip(episodes, episodes[1:]):
        assert shorter.events == longer.events[:len(shorter.events)]
        assert shorter.queries[:7] == longer.queries[:7]
        for condition in CONDITIONS:
            before = schedule_episode(shorter, condition)
            after = schedule_episode(longer, condition)
            # Full preceding blocks are identical; a shorter last block may
            # contain fewer replay draws than its longer counterpart.
            full_blocks = len(shorter.events) // 4
            assert before.metadata["blocks"][:full_blocks] == after.metadata["blocks"][:full_blocks]


def test_template_generalization_changes_only_surface_forms():
    seen = make_episode(23, 13, 15, "dev")
    heldout = make_episode(23, 13, 15, "dev", template_family="heldout")
    for left, right in zip(seen.events, heldout.events):
        assert replace(left, template_family="heldout") == right
        assert left.text != right.text
        assert len(left.encoded) == len(right.encoded) == RECORD_BYTES
        assert left.encoded[-3:] == right.encoded[-3:] == left.value.encode("ascii") + b".\n"
        assert left.encoded[0] == right.encoded[0] == ord(" ")
    for left, right in zip(seen.queries, heldout.queries):
        assert replace(left, template_family="heldout") == right
        assert render_query(left) != render_query(right)
    with pytest.raises(ValueError, match="seen templates"):
        make_episode(23, 13, 15, "train", template_family="heldout")


@pytest.mark.parametrize("family", ("seen", "heldout"))
@pytest.mark.parametrize("index", (0, 1))
def test_query_prefix_has_fixed_width_key_and_no_interpolated_answer(family, index):
    query = make_episode(10, 3, 0, "dev", revision_fraction=0).queries[0]
    rendered = render_query(query, index, family)
    assert len(rendered) == QUERY_PREFIX_BYTES
    assert rendered.decode("ascii").startswith(" ")
    assert rendered.endswith((b": ", b"= "))
    assert query.key.encode("ascii") in rendered
    for value in VALUES:
        assert render_query(replace(query, target=value), index, family) == rendered


def test_json_roundtrip_validates_targets_pool_hash_and_record_structure():
    episode = make_episode(15, 8, 12, "dev")
    serialized = json.dumps(episode.to_dict(), allow_nan=False)
    assert Episode.from_dict(json.loads(serialized)) == episode
    bad = json.loads(serialized)
    query = bad["queries"][0]
    query["target"] = next(value for value in VALUES if value not in (query["target"], query["old_value"]))
    with pytest.raises(ValueError, match="derived only from online"):
        Episode.from_dict(bad)
    bad = json.loads(serialized)
    revision = next(record for record in bad["events"] if record["kind"] == "revision")
    revision["value"] = next(record["value"] for record in bad["events"]
                             if record["key"] == revision["key"] and record["kind"] == "fact")
    with pytest.raises(ValueError, match="must differ"):
        Episode.from_dict(bad)
    bad = json.loads(serialized)
    bad["events"][-1]["key"] = bad["events"][0]["key"]
    with pytest.raises(ValueError, match="no overlap"):
        Episode.from_dict(bad)
    bad = json.loads(serialized)
    bad["events"][0]["key"] = KEY_POOLS["train"][0]
    with pytest.raises(ValueError, match="split namespace"):
        Episode.from_dict(bad)
    bad = json.loads(serialized)
    bad["key_pool_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pool scheme or hash"):
        Episode.from_dict(bad)
    bad = json.loads(serialized)
    bad["events"][0]["event_id"] = 1
    with pytest.raises(ValueError, match="consecutive"):
        Episode.from_dict(bad)


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("block_size, buffer_size", [(1, 1), (4, 3), (4, 8), (100, 8)])
def test_all_schedules_use_online_only_fifo_buffers_and_never_future_records(condition, block_size, buffer_size):
    episode = make_episode(73, 7, 13, "dev")
    plan = schedule_episode(episode, condition, block_size, buffer_size)
    assert plan == schedule_episode(episode, condition, block_size, buffer_size)
    assert [event.record.event_id for event in plan.events if event.kind == "online"] == list(range(len(episode.events)))
    expected_buffer = []
    for block_meta in plan.metadata["blocks"]:
        buffer_state = [{"key": record.key, "version": record.version, "event_id": record.event_id}
                        for record in expected_buffer]
        assert block_meta["buffer_at_start"] == buffer_state
        ids = block_meta["online_event_ids"]
        reference = block_meta["reference_replay_event_ids"]
        assert len(reference) == (len(ids) if expected_buffer else 0)
        assert all(event_id < ids[0] for event_id in reference)
        assert set(reference) <= {record.event_id for record in expected_buffer}
        for event_id in ids:
            record = episode.events[event_id]
            expected_buffer = [old for old in expected_buffer if old.key != record.key]
            expected_buffer.append(record)
            expected_buffer = expected_buffer[-buffer_size:]
        assert block_meta["buffer_at_end"] == [
            {"key": record.key, "version": record.version, "event_id": record.event_id}
            for record in expected_buffer]
    latest = {}
    for event in plan.events:
        assert event.record == episode.events[event.record.event_id]
        if event.kind == "online":
            latest[event.record.key] = event.record
        else:
            assert event.record.event_id <= event.after_online_event_id
            assert event.record.key in latest
        assert event.stale == (event.kind == "replay" and latest[event.record.key].version > event.record.version)
    assert plan.metadata["stale_replay_count"] == sum(event.stale for event in plan.events)
    assert len(plan.metadata["final_buffer"]) <= buffer_size
    assert json.loads(json.dumps(plan.to_dict())) == plan.metadata


def test_continuous_periodic_and_repeat_controls_have_paired_budgets_and_slots():
    episode = make_episode(61, 7, 13, "dev")  # 23 online events; last block has 3.
    plans = {condition: schedule_episode(episode, condition) for condition in CONDITIONS}
    continuous, periodic, repeat = [plans[name] for name in ("continuous", "periodic", "repeat_current")]
    assert continuous.metadata["selected_replay_event_ids"] == periodic.metadata["selected_replay_event_ids"]
    assert continuous.metadata["reference_replay_event_ids"] == repeat.metadata["reference_replay_event_ids"]
    assert len(continuous.events) == len(periodic.events) == len(repeat.events) == 42
    assert len(plans["online"].events) == 23
    assert plans["online"].metadata["extra_record_count"] == 0
    for plan in (continuous, periodic, repeat):
        assert plan.metadata["extra_record_count"] == 19
        assert plan.metadata["input_token_count"] == 42 * 32 + 31
        assert plan.metadata["training_sequence_token_count"] == 43 * 32
        assert plan.metadata["online_token_count"] == 23 * 32
        assert plan.metadata["extra_token_count"] == 19 * 32
        assert plan.metadata["final_buffer"] == plans["online"].metadata["final_buffer"]
    for block in continuous.metadata["blocks"]:
        block_index, online_ids = block["block_index"], block["online_event_ids"]
        per_condition = {name: [event for event in plan.events if event.block_index == block_index]
                         for name, plan in plans.items()}
        if block_index == 0:
            assert all([event.kind for event in events] == ["online"] * len(online_ids)
                       for events in per_condition.values())
            continue
        assert [event.kind for event in per_condition["continuous"]] == ["online", "replay"] * len(online_ids)
        assert [event.kind for event in per_condition["periodic"]] == ["online"] * len(online_ids) + ["replay"] * len(online_ids)
        assert [event.kind for event in per_condition["repeat_current"]] == ["online", "repeat_current"] * len(online_ids)
        for observation, replay in zip(per_condition["repeat_current"][::2], per_condition["repeat_current"][1::2]):
            assert observation.record == replay.record
        assert all(event.after_online_event_id == online_ids[-1]
                   for event in per_condition["periodic"] if event.kind == "replay")


def test_replay_uses_replacement_and_eviction_is_not_refreshed_by_replay():
    episode = make_episode(9, 4, 9, "dev", revision_fraction=0)
    plan = schedule_episode(episode, "continuous", block_size=4, buffer_size=1)
    assert plan.metadata["blocks"][1]["selected_replay_event_ids"] == [3, 3, 3, 3]
    assert plan.metadata["blocks"][2]["selected_replay_event_ids"] == [7, 7, 7, 7]
    assert plan.metadata["blocks"][3]["selected_replay_event_ids"] == [11]
    assert plan.metadata["final_buffer"][0]["event_id"] == 12


def test_revision_recency_replaces_a_buffer_entry_and_keeps_only_latest_version():
    episode = make_episode(101, 4, 0, "dev", revision_fraction=1)
    plan = schedule_episode(episode, "continuous", block_size=2, buffer_size=4)
    for block in plan.metadata["blocks"]:
        assert len({row["key"] for row in block["buffer_at_end"]}) == len(block["buffer_at_end"])
    assert [row["event_id"] for row in plan.metadata["final_buffer"]] == [4, 5, 6, 7]
    assert {row["version"] for row in plan.metadata["final_buffer"]} == {2}
    replaced_keys = {episode.events[4].key, episode.events[5].key}
    superseded_ids = {record.event_id for record in episode.events[:4] if record.key in replaced_keys}
    assert not superseded_ids & set(plan.metadata["blocks"][-1]["reference_replay_event_ids"])


def test_same_sampled_old_record_can_be_fresh_continuously_but_stale_periodically():
    keys = KEY_POOLS["dev"][:2]
    events = (
        FactRecord(0, keys[0], "a", 1, "fact", "seen", 0),
        FactRecord(1, keys[1], "b", 1, "fact", "seen", 1),
        FactRecord(2, keys[0], "c", 2, "revision", "seen", 0),
        FactRecord(3, keys[1], "d", 2, "revision", "seen", 1),
    )
    queries = (FactQuery(keys[0], "c", "a", "revised", 0, 2, "seen", 0),
               FactQuery(keys[1], "d", "b", "revised", 1, 3, "seen", 1))
    episode = Episode(7, 2, 0, "dev", "seen", 1.0, False, events, queries)
    # Search the tiny deterministic seed set for a selection whose first replay
    # concerns the key revised later in the block. This asserts timing, not RNG internals.
    candidates = [schedule_episode(episode, "continuous", 2, 8, schedule_seed=seed) for seed in range(20)]
    continuous = next(plan for plan in candidates if plan.metadata["selected_replay_event_ids"][0] == 1)
    periodic = schedule_episode(episode, "periodic", 2, 8, schedule_seed=continuous.metadata["schedule_seed"])
    assert continuous.metadata["selected_replay_event_ids"] == periodic.metadata["selected_replay_event_ids"]
    assert continuous.metadata["stale_replay_count"] == 1
    assert periodic.metadata["stale_replay_count"] == 2
    assert continuous.metadata["final_buffer"] == periodic.metadata["final_buffer"]
    for condition in ("continuous", "periodic"):
        for query in queries:
            _, answer, metadata = encode_example(episode, query, condition, 2, 8,
                                                 schedule_seed=continuous.metadata["schedule_seed"])
            assert answer == ord(query.target) != ord(query.old_value)
            assert metadata["query"]["latest_event_id"] in (2, 3)


def test_future_changes_do_not_change_earlier_replay_choices_or_buffers():
    short = make_episode(43, 5, 5, "dev")
    data = short.to_dict()
    # Changing a later distractor must not change any earlier schedule decision.
    last = data["events"][-1]
    last["value"] = next(value for value in VALUES if value != last["value"])
    data["queries"][-1]["target"] = last["value"]
    changed = Episode.from_dict(data)
    for condition in CONDITIONS:
        before = schedule_episode(short, condition)
        after = schedule_episode(changed, condition)
        assert before.metadata["reference_replay_event_ids"] == after.metadata["reference_replay_event_ids"]
        assert before.metadata["blocks"] == after.metadata["blocks"]
        for left, right in zip(before.events, after.events):
            if left.record.event_id != len(short.events) - 1:
                assert left == right


def test_encoded_examples_isolate_final_query_and_preserve_targets_across_schedules():
    episode = make_episode(31, 6, 8, "dev")
    for query in episode.queries:
        suffixes = []
        for condition in CONDITIONS:
            prefix, answer, metadata = encode_example(episode, query, condition)
            plan = schedule_episode(episode, condition)
            assert isinstance(prefix, bytes) and type(answer) is int
            assert answer == ord(query.target)
            assert len(prefix) % RECORD_BYTES == QUERY_PREFIX_BYTES
            assert prefix[:-QUERY_PREFIX_BYTES] == b"".join(record.encoded for record in plan.records)
            assert metadata["query_start_token"] == len(prefix) - QUERY_PREFIX_BYTES
            assert metadata["query_start_token"] % RECORD_BYTES == 0
            assert metadata["query_attention_segment"] == len(plan.records)
            assert metadata["answer_byte"] == answer
            assert len(prefix + bytes([answer])) == metadata["training_sequence_token_count"]
            assert hashlib.sha256(prefix).hexdigest() == metadata["prefix_sha256"]
            assert len(json.dumps(metadata, allow_nan=False)) > 0
            suffixes.append(prefix[-QUERY_PREFIX_BYTES:])
        assert all(suffix == render_query(query) for suffix in suffixes)
    wrong = replace(episode.queries[-1], target=next(value for value in VALUES if value != episode.queries[-1].target))
    with pytest.raises(ValueError, match="validated queries"):
        encode_example(episode, wrong, "online")


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"seed": -1}, {"facts": 0}, {"facts": 2.5}, {"delay": -1},
    {"split": "validation"}, {"template_family": "unknown"}, {"include_recent": 1},
    {"revision_fraction": -0.1}, {"revision_fraction": 1.1}, {"revision_fraction": float("nan")},
    {"revision_fraction": float("inf")}, {"revision_fraction": True},
])
def test_episode_rejects_invalid_configuration(kwargs):
    arguments = {"seed": 1, "facts": 3, "delay": 2, "split": "dev"}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        make_episode(**arguments)


@pytest.mark.parametrize("kwargs", [
    {"condition": "sleep"}, {"block_size": 0}, {"block_size": True}, {"buffer_size": 0},
    {"buffer_size": 2.5}, {"schedule_seed": -1}, {"schedule_seed": False},
])
def test_scheduler_rejects_invalid_configuration(kwargs):
    episode = make_episode(1, 3, 2, "dev")
    arguments = {"condition": "online", **kwargs}
    with pytest.raises(ValueError):
        schedule_episode(episode, **arguments)
