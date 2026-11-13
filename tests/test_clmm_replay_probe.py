"""Schedule/retention invariants independent of trained checkpoint accuracy."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.replay_probe import (
    ALGORITHMS, EVAL_BLOCKS, assert_evicted, block_events, evaluate_batch,
    make_episodes, read_frozen, select_recent, latest_available_sources,
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


class DecayRecordingMemory(CountingMemory):
    def __init__(self, fail_at=None):
        super().__init__()
        self.to_decay_factor = nn.Linear(1, 1)
        with torch.no_grad():
            self.to_decay_factor.weight.zero_()
            self.to_decay_factor.bias.fill_(1)
        self.decays = []
        self.fail_at = fail_at

    def store_memories(self, fact, **kwargs):
        self.decays.append(self.to_decay_factor(fact[0]).sigmoid().clone())
        if kwargs["seq_index"] == self.fail_at:
            raise RuntimeError("injected write failure")
        return super().store_memories(fact, **kwargs)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("scale", (0, 1))
def test_replay_decay_control_preserves_observation_gates_and_slow_weights(algorithm, scale):
    reader = CountingReader()
    reader.memory = DecayRecordingMemory()
    gate = reader.memory.to_decay_factor
    snapshot = {name: tensor.clone() for name, tensor in gate.state_dict().items()}
    keys, values = make_episodes(2, 31)
    evaluate_batch(reader, keys, values, algorithm, replay_decay_scale=scale)
    events = [("observation", i) for i in range(8)]
    for block in range(1, 9):
        events.extend(block_events(algorithm, block)[1])
    assert len(reader.memory.decays) == len(events)
    for (kind, _), actual in zip(events, reader.memory.decays):
        expected = 0.0 if kind == "replay" and scale == 0 else torch.sigmoid(torch.tensor(1.)).item()
        torch.testing.assert_close(actual, torch.full_like(actual, expected))
    assert not gate._forward_hooks
    assert all(torch.equal(snapshot[name], tensor) for name, tensor in gate.state_dict().items())


def test_replay_decay_hook_is_removed_when_write_raises():
    reader = CountingReader()
    # First continuous replay follows the 8 prefill and 1 observation writes.
    reader.memory = DecayRecordingMemory(fail_at=9)
    keys, values = make_episodes(1, 32)
    with pytest.raises(RuntimeError, match="injected write failure"):
        evaluate_batch(reader, keys, values, "continuous", replay_decay_scale=0)
    assert torch.all(reader.memory.decays[-1] == 0)
    assert not reader.memory.to_decay_factor._forward_hooks
    torch.testing.assert_close(reader.memory.to_decay_factor(torch.zeros(1, 1)), torch.ones(1, 1))


def test_recent_selection_uses_latest_observed_labels_excludes_targets_and_future():
    keys = torch.tensor([[0, 1, 2, 3, 0, 4, 2, 5, 3, 2, 1, 0, 6, 7]])
    values = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6]])
    query, target, observed_index = select_recent(keys, values, observations=12)
    assert query.tolist() == [[2, 3, 5, 4]]
    assert target.tolist() == [[2, 1, 7, 5]]
    assert observed_index.tolist() == [[9, 8, 7, 5]]
    # Unseen future changes must have no effect on query selection or labels.
    keys[:, 12:], values[:, 12:] = 2, 0
    repeated = select_recent(keys, values, observations=12)
    for actual, expected in zip(repeated, (query, target, observed_index)):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("scale", (0, 1))
def test_recent_readout_preserves_target_rows_and_audits_latest_write(algorithm, scale):
    keys, values = make_episodes(3, 61)
    baseline_reader, reader = CountingReader(), CountingReader()
    baseline_reader.memory, reader.memory = DecayRecordingMemory(), DecayRecordingMemory()
    baseline = evaluate_batch(baseline_reader, keys, values, algorithm, replay_decay_scale=scale)
    rows = evaluate_batch(reader, keys, values, algorithm, replay_decay_scale=scale, recent_queries=4)
    assert [row for row in rows if row["condition"] != "recent"] == baseline
    recent = [row for row in rows if row["condition"] == "recent"]
    assert len(recent) == 3 * len(EVAL_BLOCKS) * 4
    assert len({(row["episode"], row["block"], row["query_rank"]) for row in recent}) == len(recent)
    for row in recent:
        episode, key, end = row["episode"], row["target_key"], row["total_observations"]
        assert key not in (int(keys[episode, 0]), int(keys[episode, 1]))
        last_observed = max(i for i in range(end) if keys[episode, i] == key)
        assert row["target_value"] == values[episode, last_observed]
        assert row["last_target_observation_index"] == last_observed
        assert row["future_observations"] == end - 1 - last_observed
        events = [("observation", i) for i in range(8)]
        for block in range(1, row["block"] + 1):
            events.extend(block_events(algorithm, block)[1])
        last_write = max(i for i, (_, source) in enumerate(events) if keys[episode, source] == key)
        source = events[last_write][1]
        assert row["last_target_memory_write_index"] == last_write
        assert row["last_memory_write_source_observation_index"] == source
        assert row["last_memory_write_value"] == values[episode, source]
        assert row["future_memory_writes_since_last_target"] == len(events) - 1 - last_write
        bank = block_events(algorithm, row["block"])[0]
        assert row["target_absent_from_bank"] == (key not in keys[episode, list(bank)].tolist())
        assert row["target_absent_from_fifo"] == (key not in keys[episode, end - 8:end].tolist())
    assert len(reader.memory.calls) == len(baseline_reader.memory.calls)


def test_replay_label_resolution_respects_causal_arrivals_and_bank_capacity():
    # The bank already contains a target revision at4. Key2 is revised at12,
    # which continuous rank2 has not observed yet but periodic rank2 has.
    keys = torch.tensor([[0, 1, 2, 3, 0, 4, 5, 6, 7, 3, 8, 9, 2, 4, 5, 6]])
    values = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 7, 6, 5, 4]])
    bank = tuple(range(8))
    assert latest_available_sources(keys, bank, observations=11, replay_index=2).tolist() == [2]
    assert latest_available_sources(keys, bank, observations=16, replay_index=2).tolist() == [12]
    source = latest_available_sources(keys, bank, observations=9, replay_index=0)
    assert source.tolist() == [4] and values.gather(1, source[:, None]).item() == 4
    changed = keys.clone()
    changed[:, 11:] = 2  # Still unseen when only 11 observations have arrived.
    assert latest_available_sources(changed, bank, observations=11, replay_index=2).tolist() == [2]
    with pytest.raises(ValueError, match="boundary"):
        latest_available_sources(keys, bank, observations=17, replay_index=2)
    with pytest.raises(ValueError, match="eight-observation"):
        latest_available_sources(keys, tuple(range(9)), observations=16, replay_index=2)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("mode", ("stored", "latest-observed"))
def test_replay_value_intervention_preserves_keys_counts_and_records_actual_sources(algorithm, mode):
    keys, values = make_episodes(3, 101)
    reader = CountingReader()
    diagnostics = []
    rows = evaluate_batch(reader, keys, values, algorithm, recent_queries=4,
                          replay_value_source=mode, replay_diagnostics=diagnostics)
    assert len(reader.memory.calls) == (72 if algorithm == "online" else 136)
    assert len(diagnostics) == (0 if algorithm == "online" else 3 * 64)
    recorded = {(row["episode"], row["memory_write_index"]): row for row in diagnostics}
    events = [("observation", i, ()) for i in range(8)]
    for block in range(1, 9):
        bank, block_schedule = block_events(algorithm, block)
        arrivals = []
        for kind, source in block_schedule:
            if kind == "observation":
                arrivals.append(source)
            events.append((kind, source, tuple(bank) + tuple(arrivals)))
    actual_sources = []
    for write_index, ((kind, source, available), (written_keys, written_values)) in enumerate(
            zip(events, reader.memory.calls)):
        assert torch.equal(written_keys, keys[:, source].float())
        chosen = []
        for episode in range(len(keys)):
            key = int(keys[episode, source])
            latest = max(i for i in available if keys[episode, i] == key) if kind == "replay" else source
            expected_source = latest if kind == "replay" and mode == "latest-observed" else source
            chosen.append(expected_source)
            assert written_values[episode] == values[episode, expected_source]
            if kind == "replay":
                row = recorded[episode, write_index]
                assert 8 <= row["available_pair_count"] == len(available) <= 16
                assert row["latest_available_observation_index"] == latest < row["observed_count"]
                assert row["used_observation_index"] == expected_source < row["observed_count"]
                assert row["used_value"] == values[episode, expected_source]
                assert row["stored_value"] == values[episode, source]
                assert row["stored_value_stale"] == int(values[episode, source] != values[episode, latest])
                assert row["value_substituted"] == int(values[episode, source] != values[episode, expected_source])
        actual_sources.append(chosen)
    # Recent-probe audit must follow the value source actually used, not the
    # original queue entry when that entry received a corrected value.
    for row in rows:
        if row["condition"] != "recent":
            continue
        episode, write_index = row["episode"], row["last_target_memory_write_index"]
        source = actual_sources[write_index][episode]
        assert row["last_memory_write_source_observation_index"] == source
        assert row["last_memory_write_value"] == values[episode, source]
    if mode == "stored":
        baseline = evaluate_batch(CountingReader(), keys, values, algorithm, recent_queries=4)
        assert rows == baseline
