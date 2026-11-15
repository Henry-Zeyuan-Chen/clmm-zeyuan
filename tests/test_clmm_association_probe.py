"""Association probe data obeys the fixed byte and namespace protocol."""

from collections import Counter, defaultdict
from dataclasses import replace
import json
import random

import numpy as np
import pytest

from clmm.pretrain.association_probe import (
    build_bank, fresh_heldout, fresh_same, fresh_training,
)
from clmm.pretrain.fact_streams import (
    Episode, KEY_POOLS, VALUES, encode_example, render_query,
)


def decoded(bank):
    return [Episode.from_dict(data) for data in bank["episodes"]]


def assert_canonical(episode):
    assert episode.facts == 2 and episode.delay == 0
    assert episode.revision_fraction == 0.0 and episode.include_recent is False
    assert len(episode.events) == len(episode.queries) == 2
    assert len({record.key for record in episode.events}) == 2
    assert len({record.value for record in episode.events}) == 2
    assert all(record.kind == "fact" and record.version == 1
               and record.template_index == 0 for record in episode.events)
    assert all(query.category == "stable" and query.template_index == 0
               and query.old_value is None for query in episode.queries)
    assert Episode.from_dict(episode.to_dict()) == episode


@pytest.mark.parametrize("contexts,key_count", [(32, 16), (8, 4)])
def test_bank_balances_keys_and_all_answer_classes_without_persistent_assignments(contexts, key_count):
    bank = build_bank(contexts=contexts, key_count=key_count)
    episodes = decoded(bank)
    assert len(episodes) == bank["contexts"] == contexts
    assert bank["key_count"] == key_count
    assert bank["key_pool"] == list(KEY_POOLS["train"][:key_count])
    assert 0 <= bank["attempt"] < 10_000
    assert len({episode.seed for episode in episodes}) == contexts
    pairs, key_counts, value_counts = [], Counter(), Counter()
    assignments = defaultdict(set)
    for episode in episodes:
        assert_canonical(episode)
        assert episode.split == "train" and episode.template_family == "seen"
        pair = tuple(record.key for record in episode.events)
        pairs.append(pair)
        for record in episode.events:
            key_counts[record.key] += 1
            value_counts[record.value] += 1
            assignments[record.key].add(record.value)
    assert len(set(pairs)) == contexts
    assert key_counts == Counter({key: contexts * 2 // key_count for key in bank["key_pool"]})
    assert value_counts == Counter({value: contexts * 2 // len(VALUES) for value in VALUES})
    assert all(len(values) >= 2 for values in assignments.values())
    assert json.loads(json.dumps(bank, allow_nan=False)) == bank


def test_examples_are_exactly_95_bytes_and_context_pair_changes_only_query_key():
    examples = decoded(build_bank()) + fresh_heldout(41, family="heldout")[:4]
    for episode in examples:
        first, second = episode.queries
        left, left_answer, _ = encode_example(episode, first, "online")
        right, right_answer, _ = encode_example(episode, second, "online")
        assert len(left) == len(right) == 95
        assert left[:64] == right[:64] == b"".join(record.encoded for record in episode.events)
        assert left[64:] == render_query(first) and right[64:] == render_query(second)
        assert left[64:].replace(first.key.encode("ascii"), b"KEY") == (
            right[64:].replace(second.key.encode("ascii"), b"KEY"))
        assert left_answer == ord(episode.events[0].value)
        assert right_answer == ord(episode.events[1].value)
        assert left_answer != right_answer
        for query in episode.queries:
            for value in VALUES:
                assert render_query(replace(query, target=value)) == render_query(query)


def test_fresh_same_cycles_ordered_contexts_and_excludes_both_original_values():
    bank = build_bank()
    originals = decoded(bank)
    fresh = fresh_same(bank, 105, count=67)
    assert fresh == fresh_same(bank, 105, count=67)
    assert fresh[:9] == fresh_same(bank, 105, count=9)
    assert fresh != fresh_same(bank, 106, count=67)
    assert len({episode.seed for episode in fresh}) == len(fresh)
    for index, episode in enumerate(fresh):
        assert_canonical(episode)
        original = originals[index % len(originals)]
        assert [record.key for record in episode.events] == [record.key for record in original.events]
        assert not {record.value for record in episode.events} & {
            record.value for record in original.events}
        assert episode.split == "train" and episode.template_family == "seen"
    with pytest.raises(ValueError, match="seen templates only"):
        fresh_same(bank, 105, family="heldout")


def test_heldout_keys_are_isolated_and_template_families_share_semantics():
    seen = fresh_heldout(311)
    heldout = fresh_heldout(311, family="heldout")
    assert seen == fresh_heldout(311)
    assert seen[:7] == fresh_heldout(311, count=7)
    assert seen != fresh_heldout(312)
    assert len({episode.seed for episode in seen}) == len(seen)
    used = set()
    for left, right in zip(seen, heldout):
        assert_canonical(left)
        assert_canonical(right)
        assert left.seed == right.seed and left.split == right.split == "dev"
        assert replace(left, template_family="heldout", events=right.events,
                       queries=right.queries) == right
        for seen_record, heldout_record in zip(left.events, right.events):
            assert replace(seen_record, template_family="heldout") == heldout_record
            assert seen_record.encoded != heldout_record.encoded
            used.add(seen_record.key)
        for seen_query, heldout_query in zip(left.queries, right.queries):
            assert replace(seen_query, template_family="heldout") == heldout_query
            assert render_query(seen_query) != render_query(heldout_query)
    assert used <= set(KEY_POOLS["dev"])
    assert not used & (set(KEY_POOLS["train"]) | set(KEY_POOLS["test"]))


def test_fresh_training_is_step_local_and_uses_the_bank_pool():
    bank = build_bank(contexts=8, key_count=4)
    step = fresh_training(bank, 513, 9, count=32)
    fresh_training(bank, 513, 3, count=100)
    assert step == fresh_training(bank, 513, 9, count=32)
    assert step[:4] == fresh_training(bank, 513, 9, count=4)
    assert step != fresh_training(bank, 513, 10, count=32)
    assert step != fresh_training(bank, 514, 9, count=32)
    assert len({episode.seed for episode in step}) == len(step)
    assigned = defaultdict(set)
    for episode in step:
        assert_canonical(episode)
        assert episode.split == "train" and episode.template_family == "seen"
        for record in episode.events:
            assert record.key in bank["key_pool"]
            assigned[record.key].add(record.value)
    assert all(len(values) > 1 for values in assigned.values())


def test_all_generation_is_deterministic_and_leaves_global_rngs_unchanged():
    python_before, numpy_before = random.getstate(), np.random.get_state()
    bank = build_bank()
    assert bank == build_bank()
    assert bank != build_bank(seed=90202)
    fresh_same(bank, 12)
    fresh_heldout(12)
    fresh_heldout(12, family="heldout")
    fresh_training(bank, 12, 5)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0] and numpy_after[2:] == numpy_before[2:]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])


@pytest.mark.parametrize("kwargs", [
    {"seed": -1}, {"seed": True}, {"contexts": 0}, {"contexts": 7},
    {"contexts": 8.0}, {"contexts": True}, {"contexts": 9},
    {"contexts": 8, "key_count": 6}, {"contexts": 8, "key_count": 16},
    {"contexts": 16, "key_count": 4}, {"key_count": 2}, {"key_count": 5},
    {"key_count": 602}, {"key_count": True}, {"key_count": 4.0},
])
def test_bank_rejects_invalid_or_infeasible_dimensions(kwargs):
    with pytest.raises(ValueError):
        build_bank(**kwargs)


@pytest.mark.parametrize("factory,kwargs", [
    (fresh_same, {"seed": -1}), (fresh_same, {"seed": True}),
    (fresh_same, {"seed": 1, "count": 0}),
    (fresh_same, {"seed": 1, "family": "unknown"}),
    (fresh_heldout, {"seed": -1}), (fresh_heldout, {"seed": 1, "count": True}),
    (fresh_heldout, {"seed": 1, "family": "unknown"}),
    (fresh_training, {"seed": 1, "step": -1}),
    (fresh_training, {"seed": 1, "step": True}),
    (fresh_training, {"seed": 1, "step": 0, "count": 0}),
])
def test_fresh_generation_rejects_invalid_arguments(factory, kwargs):
    with pytest.raises(ValueError):
        if factory is fresh_heldout:
            factory(**kwargs)
        else:
            factory(build_bank(contexts=8, key_count=4), **kwargs)
