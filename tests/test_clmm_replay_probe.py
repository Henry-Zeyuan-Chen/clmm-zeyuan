"""Schedule/retention invariants independent of trained checkpoint accuracy."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.replay_probe import (
    ALGORITHMS, EVAL_BLOCKS, assert_evicted, block_events, evaluate_batch,
    make_episodes, read_frozen,
)


def test_episode_targets_are_revised_once_and_never_reintroduced():
    keys, values = make_episodes(41, 51131)
    again = make_episodes(41, 51131)
    assert torch.equal(keys, again[0]) and torch.equal(values, again[1])
    assert keys.shape == values.shape == (41, 72)
    assert torch.all(keys[:, 0] == keys[:, 4])
    assert torch.all(values[:, 0] != values[:, 4])
    assert torch.all(keys[:, 0] != keys[:, 1])
    for episode in range(len(keys)):
        assert (keys[episode] == keys[episode, 0]).nonzero().flatten().tolist() == [0, 4]
        assert (keys[episode] == keys[episode, 1]).nonzero().flatten().tolist() == [1]


@pytest.mark.parametrize("block", range(1, 9))
def test_replay_arms_have_identical_past_only_examples_and_order(block):
    bank, continuous = block_events("continuous", block)
    other_bank, periodic = block_events("periodic", block)
    assert bank == other_bank == tuple(range(8 * (block - 1), 8 * block))
    assert tuple(i for kind, i in continuous if kind == "replay") == bank
    assert tuple(i for kind, i in periodic if kind == "replay") == bank
    incoming = list(range(8 * block, 8 * (block + 1)))
    for events in (continuous, periodic):
        assert [i for kind, i in events if kind == "observation"] == incoming
    assert [kind for kind, _ in continuous] == ["observation", "replay"] * 8
    assert [kind for kind, _ in periodic] == ["observation"] * 8 + ["replay"] * 8
    assert max(bank) < min(incoming)
    assert block_events("online", block)[1] == [("observation", i) for i in incoming]


def test_eviction_checks_snapshot_fifo_and_pending_queue():
    keys, _ = make_episodes(3, 19)
    with pytest.raises(AssertionError, match="accessible"):
        assert_evicted(keys, block_events("continuous", 1)[0], 1)
    for block in EVAL_BLOCKS:
        bank, _ = block_events("continuous", block)
        assert_evicted(keys, bank, block)
        with pytest.raises(AssertionError, match="queue"):
            assert_evicted(keys, bank, block, [bank[-1]])
    corrupted = keys.clone()
    corrupted[:, 23] = keys[:, 0]
    with pytest.raises(AssertionError, match="accessible"):
        assert_evicted(corrupted, block_events("continuous", 2)[0], 2)


class CountingMemory:
    """Record actual write order and state chaining; no trained model required."""
    def __init__(self):
        self.calls = []

    def init_weights(self, batch):
        return {"count": torch.zeros(batch, 1)}

    def store_memories(self, fact, *, weights, past_state, seq_index, return_surprises):
        assert seq_index == len(self.calls)
        assert return_surprises is False
        if seq_index:
            assert past_state[0] is weights
        else:
            assert past_state is None
        self.calls.append((fact[0, :, 0, 0].clone(), fact[1, :, 0, 0].clone()))
        return None, SimpleNamespace(states=({"count": weights["count"] + 1}, {}))


class CountingReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Embedding.from_pretrained(torch.arange(16.).reshape(16, 1))
        self.value = nn.Embedding.from_pretrained(torch.arange(8.).reshape(8, 1))
        self.memory = CountingMemory()
        self.queries = []

    def read(self, query, fast):
        self.queries.append((query.clone(), fast["count"].clone()))
        return torch.zeros(len(query), 8)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_evaluator_counts_actual_writes_and_observation_delays(algorithm):
    keys, values = make_episodes(3, 17)
    reader = CountingReader()
    rows = evaluate_batch(reader, keys, values, algorithm)
    expected = [("observation", i) for i in range(8)]
    for block in range(1, 9):
        expected.extend(block_events(algorithm, block)[1])
    assert len(reader.memory.calls) == (72 if algorithm == "online" else 136)
    for (_, index), (written_key, written_value) in zip(expected, reader.memory.calls):
        torch.testing.assert_close(written_key, keys[:, index].float())
        torch.testing.assert_close(written_value, values[:, index].float())
    assert len(rows) == 3 * 3 * 2
    assert len(reader.queries) == 6
    for row in rows:
        observations = 8 + 8 * row["block"]
        replays = 0 if algorithm == "online" else 8 * row["block"]
        assert row["total_observations"] == observations
        assert row["total_replay_writes"] == replays
        assert row["total_memory_writes"] == observations + replays
        assert row["future_observations"] == observations - (5 if row["condition"] == "revised" else 2)
        last_observation = 4 if row["condition"] == "revised" else 1
        last_write = max(i for i, (_, source_index) in enumerate(expected[:observations + replays])
                         if source_index in ((0, 4) if row["condition"] == "revised" else (1,)))
        assert row["last_target_observation_index"] == last_observation
        assert row["last_target_memory_write_index"] == last_write
        assert row["future_memory_writes_since_last_target"] == observations + replays - 1 - last_write
        assert row["pending_replays"] == row["local_attention_tokens"] == 0
    for i, block in enumerate(EVAL_BLOCKS):
        for target in (0, 1):
            query, count = reader.queries[2 * i + target]
            assert torch.equal(query, keys[:, target])
            expected_count = 8 + 8 * block * (1 if algorithm == "online" else 2)
            assert torch.all(count == expected_count)


def test_query_detects_mutating_reader():
    class MutatingReader:
        def read(self, query, fast):
            fast["weight"].add_(1)
            return torch.zeros(len(query), 8)

    with pytest.raises(AssertionError, match="mutated"):
        read_frozen(MutatingReader(), (torch.tensor([0]),), {"weight": torch.zeros(1)})
