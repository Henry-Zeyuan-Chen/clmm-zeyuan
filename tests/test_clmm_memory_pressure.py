"""Small CPU audits of the pressure probe's data, schedules and read isolation."""

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts.memory_pressure_probe import (
    ARMS, BATCH_ARRAYS, block_events, evaluate_batch, generate_data, query_spec, run, slice_data,
)


class RecordingMemory:
    def __init__(self, nkeys, fail_at=None):
        self.nkeys = nkeys
        self.calls = []
        self.fail_at = fail_at
        self.to_decay_factor = nn.Linear(1, 1)
        with torch.no_grad():
            self.to_decay_factor.weight.zero_()
            self.to_decay_factor.bias.fill_(1)

    def init_weights(self, batch):
        return {"last_value": torch.zeros(batch, self.nkeys)}

    def store_memories(self, fact, *, weights, past_state, seq_index, return_surprises):
        assert seq_index == len(self.calls) and not return_surprises
        assert past_state is None if seq_index == 0 else past_state[0] is weights
        key, value = fact[0, :, 0, 0].long(), fact[1, :, 0, 0]
        decay = self.to_decay_factor(fact[0]).sigmoid().clone()
        self.calls.append((key.clone(), value.clone(), decay))
        if seq_index == self.fail_at:
            raise RuntimeError("injected store failure")
        updated = weights["last_value"].clone()
        updated.scatter_(1, key[:, None], value[:, None])
        return None, SimpleNamespace(states=({"last_value": updated}, {}))


class RecordingReader(nn.Module):
    def __init__(self, nkeys=32, fail_at=None):
        super().__init__()
        self.key = nn.Embedding.from_pretrained(torch.arange(float(nkeys))[:, None])
        self.value = nn.Embedding.from_pretrained(torch.arange(8.)[:, None])
        self.memory = RecordingMemory(nkeys, fail_at)
        self.read_calls = []

    def read(self, query, fast):
        self.read_calls.append(query.clone())
        value = fast["last_value"].gather(1, query[:, None]).long()[:, 0]
        return torch.nn.functional.one_hot(value, 8).float()


def tiny_data(episodes=3, facts=16, delays=(0, 8, 16), seed=73, buffer_size=8):
    return generate_data(episodes, facts, 32, 8, delays, seed, buffer_size=buffer_size, block_size=8)


def test_fixed_labels_target_ages_and_generation_are_batch_independent():
    data = tiny_data(episodes=5)
    prefix = tiny_data(episodes=2)
    for name in BATCH_ARRAYS:
        np.testing.assert_array_equal(data[name][:2], prefix[name])
        np.testing.assert_array_equal(np.concatenate((slice_data(data, 0, 2)[name], slice_data(data, 2, 5)[name])), data[name])
    for facts in (8, 16, 32):
        current = tiny_data(facts=facts)
        prefill = facts + 2
        query, answers, indices = query_spec(current, prefill)
        np.testing.assert_array_equal(prefill - 1 - indices[:, :4], np.tile([5, 4, 1, 0], (3, 1)))
        np.testing.assert_array_equal(query[:, :4], current["active_keys"][:, -4:])
        assert np.all(current["values"][:, facts:facts + 2] != current["values"][:, facts - 2:facts])
        for episode in range(3):
            final = dict(zip(current["active_keys"][episode], current["final_values"][episode]))
            assert len(set(current["keys"][episode, :facts])) == facts
            assert all(final[key] == value for key, value in zip(
                current["keys"][episode, prefill:], current["values"][episode, prefill:]))
            assert not set(current["keys"][episode, prefill:]) & set(query[episode, :4])


def test_numeric_trace_reconstructs_algorithm_r_revisions_and_causal_samples():
    data = tiny_data(episodes=2, delays=(0, 8, 64), buffer_size=32)
    prefill, size = data["prefill"], data["buffer_size"]
    for episode, (keys, values) in enumerate(zip(data["keys"], data["values"])):
        slots = [None] * size
        for index, (key, value) in enumerate(zip(keys, values)):
            if index >= prefill and (index - prefill) % 8 == 0:
                block = (index - prefill) // 8
                choices = data["sample_slots"][episode, block]
                assert len(set(choices)) == 8
                for rank, slot in enumerate(choices):
                    source, label_source = slots[slot]
                    assert data["sample_sources"][episode, block, rank] == source < index
                    assert data["sample_value_sources"][episode, block, rank] == label_source < index
                    assert values[label_source] == data["final_values"][episode, np.flatnonzero(data["active_keys"][episode] == keys[source])[0]]
            for slot, entry in enumerate(slots):
                if entry is not None and keys[entry[0]] == key:
                    slots[slot] = (entry[0], index)
            draw = int(data["reservoir_draw"][episode, index])
            assert 0 <= draw <= index
            if index < size:
                assert draw == index
            if draw < size:
                slots[draw] = (index, index)
            delay = index + 1 - prefill
            if delay in data["delays"]:
                qi = list(data["delays"]).index(delay)
                expected_source = [-1 if item is None else item[0] for item in slots]
                expected_value_source = [-1 if item is None else item[1] for item in slots]
                np.testing.assert_array_equal(data["working_sources_at_query"][episode, qi], expected_source)
                np.testing.assert_array_equal(data["working_value_sources_at_query"][episode, qi], expected_value_source)
        # At prefill, both copies of each revised key remain resident because
        # this fixture's reservoir is larger than the prefill.
        resident = data["working_sources_at_query"][episode, 0]
        label_sources = data["working_value_sources_at_query"][episode, 0]
        for target, replacement in zip(data["active_keys"][episode, -2:], (16, 17)):
            copies = np.flatnonzero((resident >= 0) & (keys[np.maximum(resident, 0)] == target))
            assert len(copies) == 2
            assert np.all(label_sources[copies] == replacement)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("scale", (0, 1))
def test_matched_writes_replay_tuples_auxiliary_decay_and_shadow_residency(arm, scale):
    data = tiny_data()
    reader = RecordingReader()
    rows = evaluate_batch(reader, data, arm, aux_decay_scale=scale)
    prefill = data["prefill"]
    events = [("observation", index, None) for index in range(prefill)]
    for block in range(2):
        events += [(kind, index, block) for kind, index in block_events(arm, block, prefill, 8)]
    assert len(reader.memory.calls) == len(events) == prefill + (16 if arm == "online" else 32)
    for (kind, index, block), (key, value, decay) in zip(events, reader.memory.calls):
        source = data["sample_sources"][:, block, index] if kind == "replay" else np.full(3, index)
        label_source = data["sample_value_sources"][:, block, index] if kind == "replay" else source
        np.testing.assert_array_equal(key.numpy(), data["keys"][np.arange(3), source])
        np.testing.assert_array_equal(value.numpy(), data["values"][np.arange(3), label_source])
        expected = 0.0 if kind != "observation" and scale == 0 else torch.sigmoid(torch.tensor(1.)).item()
        torch.testing.assert_close(decay, torch.full_like(decay, expected))
    assert not reader.memory.to_decay_factor._forward_hooks
    assert len(rows) == 3 * 3 * 8 and len(reader.read_calls) == 3 * 8
    assert all(row["correct"] for row in rows)
    for row in rows:
        episode, key, delay = row["episode"], row["key"], row["delay"]
        qi = list(data["delays"]).index(delay)
        working = data["working_sources_at_query"][episode, qi]
        resident = data["keys"][episode, working[working >= 0]]
        assert row["working_resident"] == (key in resident)
        assert row["working_resident_copies"] == np.sum(resident == key)
        selected_sources = data["sample_sources"][episode, :delay // 8].flatten()
        selected_keys = data["keys"][episode, selected_sources]
        assert row["shadow_ever_selected_for_replay"] == (key in selected_keys)
        assert not row["snapshot_resident"] and row["snapshot_active_pairs"] == 0
        assert row["pending_auxiliary_writes"] == row["local_attention_tokens"] == 0
        write_count = prefill + delay * (1 if arm == "online" else 2)
        past_events = events[:write_count]
        key_events, aux_events, replay_events = [], [], []
        for write_index, (kind, source_index, block) in enumerate(past_events):
            source = data["sample_sources"][episode, block, source_index] if kind == "replay" else source_index
            if data["keys"][episode, source] == key:
                key_events.append(write_index)
                if kind != "observation":
                    aux_events.append(write_index)
                if kind == "replay":
                    replay_events.append(write_index)
        assert row["last_any_write_index"] == key_events[-1]
        assert row["last_aux_write_index"] == (aux_events[-1] if aux_events else -1)
        assert row["last_replay_write_index"] == (replay_events[-1] if replay_events else -1)
        assert row["auxiliary_writes_for_key"] == len(aux_events)
        assert row["replay_writes_for_key"] == len(replay_events)
        assert row["shadow_selection_count"] == np.sum(selected_keys == key)
        assert row["total_memory_writes"] == write_count
        assert row["total_aux_writes"] == (0 if arm == "online" else delay)


def test_replay_tuples_match_and_shadow_strata_are_identical_across_arms():
    data = tiny_data(delays=(0, 8, 64))
    results, replay = {}, {}
    for arm in ARMS:
        model = RecordingReader()
        results[arm] = evaluate_batch(model, data, arm)
        if arm in ("continuous", "periodic"):
            indices = []
            count = data["prefill"]
            for block in range(8):
                for kind, _ in block_events(arm, block, data["prefill"], 8):
                    if kind == "replay":
                        indices.append(count)
                    count += 1
            replay[arm] = [(model.memory.calls[i][0].tolist(), model.memory.calls[i][1].tolist()) for i in indices]
    assert replay["continuous"] == replay["periodic"]
    fields = ("episode", "delay", "condition", "query_rank", "key", "working_resident",
              "working_resident_copies", "shadow_ever_selected_for_replay", "last_shadow_selection_block",
              "shadow_selection_count")
    anchor = [tuple(row[name] for name in fields) for row in results["online"]]
    for arm in ARMS:
        assert [tuple(row[name] for name in fields) for row in results[arm]] == anchor
    assert any(not row["working_resident"] and row["delay"] == 64 and row["condition"] != "recent"
               for row in results["online"])


def test_batch_partition_does_not_change_predictions_or_audit_rows():
    data = tiny_data(episodes=5)
    whole = evaluate_batch(RecordingReader(), data, "continuous")
    partitioned = []
    for start, stop in ((0, 2), (2, 5)):
        rows = evaluate_batch(RecordingReader(), slice_data(data, start, stop), "continuous")
        for row in rows:
            row["episode"] += start
        partitioned.extend(rows)
    key = lambda row: (row["episode"], row["delay"], row["condition"], row["query_rank"])
    assert sorted(whole, key=key) == sorted(partitioned, key=key)


def test_replay_execution_rejects_future_sources_even_when_the_label_matches():
    data = tiny_data(episodes=1)
    future = data["prefill"]
    # A preloaded trace is not permission to access a future observation.
    data["sample_sources"][0, 0, 0] = future
    data["sample_value_sources"][0, 0, 0] = future
    with pytest.raises(AssertionError, match="Noncausal"):
        evaluate_batch(RecordingReader(), data, "continuous")


def test_queries_are_immutable_and_auxiliary_hook_is_removed_on_failure():
    data = tiny_data(episodes=1)
    failing = RecordingReader(fail_at=data["prefill"] + 1)
    with pytest.raises(RuntimeError, match="injected"):
        evaluate_batch(failing, data, "repeat")
    assert not failing.memory.to_decay_factor._forward_hooks

    class MutatingReader(RecordingReader):
        def read(self, query, fast):
            fast["last_value"].add_(1)
            return torch.zeros(len(query), 8)

    with pytest.raises(AssertionError, match="mutated"):
        evaluate_batch(MutatingReader(), data, "online")


def test_real_reader_small_cpu_smoke_preserves_outer_parameters():
    from scripts.research_memory import Reader

    torch.manual_seed(37)
    model = Reader(dim=4, depth=1, expansion_factor=1, nkeys=16, nvals=8, max_inner_lr=0.05).eval()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    data = generate_data(1, 8, 16, 8, [0, 8], 91, buffer_size=8)
    for arm in ARMS:
        rows = evaluate_batch(model, data, arm)
        assert len(rows) == 16
        assert all(0 <= row["prediction"] < 8 for row in rows)
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())


def test_failure_metadata_is_saved_and_existing_artifacts_are_protected(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    args = argparse.Namespace(out=tmp_path / "failed", episodes=1, batch_size=1, device="cpu", seed=1,
                              checkpoints=[tmp_path / "missing.pt"])
    with pytest.raises(FileNotFoundError):
        run(args)
    metadata_path = args.out / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "failed" and metadata["exception_type"] == "FileNotFoundError"
    before = metadata_path.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        run(args)
    assert metadata_path.read_bytes() == before
