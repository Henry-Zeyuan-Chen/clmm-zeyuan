"""Raw-byte causality, freshness, bounded storage and atomic continuation checks."""

from copy import deepcopy
import io
import json

import pytest

from clmm.pretrain.fact_streams import FactRecord, make_episode, render_query
from clmm.pretrain.kv_scheduler import StreamingReplay
from clmm.pretrain.sequence_replay import SequenceReplay, parse_record_bytes


ARMS = [("online", "literal"), ("repeat", "literal"),
        ("continuous", "literal"), ("periodic", "literal"),
        ("continuous", "bounded"), ("periodic", "bounded")]


def raw(key, value=0, family="seen", template=0):
    return FactRecord(0, f"{key:03d}", chr(ord("a") + value), 1, "fact", family, template).encoded


def without_raw(events):
    return [{key: value for key, value in event.items() if key != "record_bytes"}
            for event in events]


def assert_storage(policy):
    state = policy.export_state()
    core = state["controller"]
    sources = {row[0] for row in core["buffer"] + core["snapshot"]}
    assert {row["source"] for row in state["payloads"]} == sources
    assert set(policy._payloads) == sources
    counts = policy.logical_counts()
    assert counts["buffer_pairs"] == len(core["buffer"]) <= 8
    assert counts["snapshot_pairs"] == len(core["snapshot"]) <= 4
    assert counts["controller_records"] == len(core["buffer"]) + len(core["snapshot"]) <= 12
    assert counts["raw_payload_records"] == len(sources) <= 12
    assert counts["raw_payload_bytes"] == 32 * len(sources) <= 384
    assert policy.budget()["online_records"] == core["next_source"]
    if core["config"]["arm"] in ("online", "repeat"):
        assert not policy._payloads
        assert counts["raw_payload_bytes"] == 0


@pytest.mark.parametrize("family,template", [("seen", 0), ("seen", 1), ("heldout", 0), ("heldout", 1)])
@pytest.mark.parametrize("key,value", [(0, 0), (7, 15), (999, 8)])
def test_parse_all_padded_observed_templates_without_canonicalization(family, template, key, value):
    record = raw(key, value, family, template)
    assert len(record) == 32 and record.startswith(b" ")
    assert parse_record_bytes(record) == (key, value)
    replay = SequenceReplay("repeat", 3)
    assert [event["record_bytes"] for event in replay.prime(record)] == [record]
    assert [event["record_bytes"] for event in replay.observe(record)] == [record, record]
    assert_storage(replay)


@pytest.mark.parametrize("bad", [None, {}, (3, 4), "fact 003 has value e.\n", bytearray(raw(3, 4)),
    memoryview(raw(3, 4)), raw(3, 4)[1:], b" " + raw(3, 4),
    b"fact 003 has value q.\n".rjust(32), b"fact 003 has value A.\n".rjust(32),
    b"fact 03 has value e.\n".rjust(32), b"fact 1000 has value e.\n".rjust(32),
    b"fact +03 has value e.\n".rjust(32), b"\tfact 003 has value e.\n".rjust(32),
    b"fact 003 has value e.\r\n".rjust(32), b"fact 003 has value e.\n".ljust(32),
    "fact ٠٠٣ has value e.\n".encode("utf8").rjust(32)])
def test_parser_rejects_nonobservations_and_non_ascii_grammar(bad):
    with pytest.raises(ValueError):
        parse_record_bytes(bad)


def test_no_query_target_episode_or_record_object_interface_and_no_attribute_access():
    episode = make_episode(73, 4, 1, "dev")
    query = episode.queries[0]

    class LabelTrap:
        def __getattribute__(self, name):
            raise AssertionError(f"Replay tried to inspect metadata: {name}")

    policy = SequenceReplay("online", 3)
    before = policy.export_state()
    for bad in (episode, episode.events[0], query, LabelTrap(), render_query(query),
                render_query(query) + bytes([query.answer_byte]), query.to_dict()):
        with pytest.raises(ValueError):
            policy.observe(bad)
        assert policy.export_state() == before
    assert len(policy.observe(episode.events[0].encoded)) == 1
    for name in ("episode", "queries", "targets", "history", "versions", "latest"):
        assert name not in vars(policy)
    with pytest.raises(TypeError):
        policy.observe(episode.events[0].encoded, query)


@pytest.mark.parametrize("arm,freshness", ARMS)
def test_every_event_and_controller_export_match_the_unchanged_core(arm, freshness):
    bridge = SequenceReplay(arm, 101, freshness=freshness)
    core = StreamingReplay(arm, 101, nkeys=1000, nvalues=16, freshness=freshness)
    observations = [raw(index % 9, (index * 7) % 16,
                        "seen" if index % 3 else "heldout", index % 2)
                    for index in range(39)]
    emitted = []
    for source, record in enumerate(observations):
        key, value = parse_record_bytes(record)
        if source < 12:
            events, expected = bridge.prime(record), [core.prime(key, value)]
        else:
            events, expected = bridge.observe(record), core.step(key, value)
        assert without_raw(events) == expected
        # In particular literal v1 schema/config/RNG/event fields remain byte-compatible.
        assert json.dumps(bridge.export_state()["controller"], sort_keys=True) == json.dumps(core.export_state(), sort_keys=True)
        for event in events:
            assert event["record_bytes"] == observations[event["source"]]
            assert event["source"] <= event["after_online"] == source
        emitted.extend(events)
        assert_storage(bridge)
    events = bridge.finish()
    assert without_raw(events) == core.finish()
    emitted.extend(events)
    assert bridge.export_state()["controller"] == core.export_state()
    assert bridge.budget()["online_records"] == len(observations)
    assert bridge.budget()["extra_records"] == sum(event["kind"] == "extra" for event in emitted)
    assert bridge.budget()["emitted_records"] == len(emitted)
    assert_storage(bridge)


@pytest.mark.parametrize("arm", ["continuous", "periodic"])
def test_revision_later_in_a_block_refreshes_exact_observed_source_bytes(arm):
    literal = SequenceReplay(arm, 57)
    bounded = SequenceReplay(arm, 57, freshness="bounded")
    original = raw(7, 0, "seen", 0)
    first = raw(8, 2, "heldout", 1)
    revised = raw(7, 15, "heldout", 0)
    revised_same_value_new_template = raw(7, 15, "seen", 1)
    records = [original, first, revised, raw(9, 4), revised_same_value_new_template]
    for policy in (literal, bounded):
        policy.prime(original)
    left, right = [], []
    for source, record in enumerate(records[1:], 1):
        left.extend(literal.observe(record))
        right.extend(bounded.observe(record))
        a, b = literal.export_state()["controller"], bounded.export_state()["controller"]
        assert a["rng"] == b["rng"] and a["snapshot"] == b["snapshot"]
        assert a["buffer"] == b["buffer"]
    literal_extras = [event for event in left if event["kind"] == "extra"]
    bounded_extras = [event for event in right if event["kind"] == "extra"]
    assert len(literal_extras) == len(bounded_extras) == 4
    assert all(event["record_bytes"] == original for event in literal_extras)
    assert [event["source"] for event in literal_extras] == [event["sampled_source"] for event in bounded_extras]
    assert [event["key"] for event in literal_extras] == [event["key"] for event in bounded_extras]
    expected = [original, revised, revised, revised_same_value_new_template] if arm == "continuous" else [revised_same_value_new_template] * 4
    assert [event["record_bytes"] for event in bounded_extras] == expected
    assert all(event["sampled_value"] == 0 and not event["fallback"] for event in bounded_extras)
    assert all(event["refreshed"] for event in bounded_extras[1:])


@pytest.mark.parametrize("arm", ["continuous", "periodic"])
def test_literal_and_bounded_share_candidate_rng_draws_and_sampled_keys_across_eviction(arm):
    literal, bounded = SequenceReplay(arm, 14), SequenceReplay(arm, 14, freshness="bounded")
    observations = [raw(index, index % 16, "heldout", index % 2) for index in range(8)]
    for record in observations:
        literal.prime(record)
        bounded.prime(record)
    left, right = [], []
    for index in range(8, 52):
        record = raw(index if index % 3 else index - 1, index % 16, "seen", index % 2)
        observations.append(record)
        a, b = literal.observe(record), bounded.observe(record)
        left.extend(a)
        right.extend(b)
        assert literal.export_state()["controller"]["rng"] == bounded.export_state()["controller"]["rng"]
        assert literal.export_state()["controller"]["snapshot"] == bounded.export_state()["controller"]["snapshot"]
        assert_storage(literal)
        assert_storage(bounded)
    left.extend(literal.finish())
    right.extend(bounded.finish())
    a = [event for event in left if event["kind"] == "extra"]
    b = [event for event in right if event["kind"] == "extra"]
    assert len(a) == len(b) == 44
    assert [event["source"] for event in a] == [event["sampled_source"] for event in b]
    assert [event["key"] for event in a] == [event["key"] for event in b]
    assert any(event["fallback"] for event in b)
    for event in b:
        assert event["record_bytes"] == observations[event["source"]]
        if event["fallback"]:
            assert event["source"] == event["sampled_source"]
            assert not event["refreshed"]
    assert literal.budget()["extra_records"] == bounded.budget()["extra_records"] == 44


def test_block_end_resolves_old_samples_after_snapshot_is_cleared_then_prunes():
    policy = SequenceReplay("periodic", 14, freshness="bounded")
    observed = [raw(index, index % 16, "heldout", index % 2) for index in range(12)]
    for record in observed[:8]:
        policy.prime(record)
    for record in observed[8:11]:
        assert len(policy.observe(record)) == 1
    state = policy.export_state()
    assert state["controller"]["snapshot"]
    final = policy.observe(observed[11])
    assert len(final) == 5
    assert policy.export_state()["controller"]["snapshot"] == []
    assert any(event.get("fallback") for event in final)
    assert all(event["record_bytes"] == observed[event["source"]] for event in final)
    assert set(policy._payloads) == {row[0] for row in policy.export_state()["controller"]["buffer"]}
    assert_storage(policy)


@pytest.mark.parametrize("arm,freshness", ARMS)
def test_payloads_are_only_fifo_snapshot_union_through_long_stream(arm, freshness):
    policy = SequenceReplay(arm, 61, freshness=freshness)
    for index in range(17):
        policy.prime(raw(index, index % 16))
        assert_storage(policy)
    sizes = []
    for index in range(1500):
        policy.observe(raw(index % 1000, index % 16, "heldout", index % 2))
        assert_storage(policy)
        if index in (97, 997, 1497):
            sizes.append(len(json.dumps(policy.export_state())))
    assert max(sizes) < 4200 and max(sizes) - min(sizes) < 1000
    assert set(vars(policy)) == {"_controller", "_payloads", "_counters"}
    assert set(policy.export_state()) == {"schema_version", "payload_encoding", "controller", "payloads", "counters"}
    policy.finish()
    assert_storage(policy)


@pytest.mark.parametrize("arm,freshness", ARMS)
@pytest.mark.parametrize("pause", [1, 2, 3, 4, 5])
def test_json_resume_mid_revision_preserves_samples_raw_templates_and_partial_finish(arm, freshness, pause):
    records = [raw(7, 0), raw(8, 2), raw(7, 9, "heldout", 1), raw(9, 4),
               raw(7, 9, "seen", 1), raw(7, 14, "heldout", 0)]

    def run(resume):
        policy = SequenceReplay(arm, 91, freshness=freshness)
        events = policy.prime(records[0])
        for index, record in enumerate(records[1:], 1):
            events.extend(policy.observe(record))
            if resume and index == pause:
                state = json.loads(json.dumps(policy.export_state()))
                policy = SequenceReplay.from_state(state)
                assert policy.export_state() == state
        events.extend(policy.finish())
        assert policy.finish() == []
        assert_storage(policy)
        return events, policy.export_state()

    assert run(True) == run(False)


def test_standard_torch_save_load_is_lossless_for_midblock_bridge_state():
    import torch

    policy = SequenceReplay("periodic", 76, freshness="bounded")
    policy.prime(raw(7))
    policy.observe(raw(7, 4, "heldout", 1))
    stream = io.BytesIO()
    torch.save(policy.export_state(), stream)
    stream.seek(0)
    restored = SequenceReplay.from_state(torch.load(stream, weights_only=True))
    assert restored.export_state() == policy.export_state()
    assert restored.finish() == policy.finish()


@pytest.mark.parametrize("corruption", ["schema", "encoding", "unexpected", "missing", "payload_kind",
    "payload_overflow", "source_bool", "source_future", "duplicate", "hex", "uppercase_hex", "extra_payload",
    "missing_payload", "raw_key", "raw_value", "raw_query", "controller_config", "controller_rng",
    "counter_bool", "counter_missing", "counter_observe", "counter_extra", "counter_prime", "counter_phase"])
def test_restore_rejects_schema_policy_source_raw_and_counter_incoherence(corruption):
    policy = SequenceReplay("periodic", 68, freshness="bounded")
    for index in range(8):
        policy.prime(raw(index, index, "heldout", index % 2))
    policy.observe(raw(1, 12, "seen", 1))
    state = deepcopy(policy.export_state())
    if corruption == "schema": state["schema_version"] = True
    elif corruption == "encoding": state["payload_encoding"] = "base64"
    elif corruption == "unexpected": state["history"] = []
    elif corruption == "missing": state.pop("controller")
    elif corruption == "payload_kind": state["payloads"] = {}
    elif corruption == "payload_overflow": state["payloads"] *= 13
    elif corruption == "source_bool": state["payloads"][0]["source"] = False
    elif corruption == "source_future": state["payloads"][0]["source"] = 9999
    elif corruption == "duplicate": state["payloads"].append(deepcopy(state["payloads"][0]))
    elif corruption == "hex": state["payloads"][0]["record_hex"] = "z" * 64
    elif corruption == "uppercase_hex": state["payloads"][0]["record_hex"] = raw(9, 9).hex().upper()
    elif corruption == "extra_payload": state["payloads"].append(dict(source=9999, record_hex=raw(9).hex()))
    elif corruption == "missing_payload": state["payloads"].pop()
    elif corruption == "raw_key": state["payloads"][0]["record_hex"] = raw(999).hex()
    elif corruption == "raw_value": state["payloads"][0]["record_hex"] = raw(0, 15).hex()
    elif corruption == "raw_query": state["payloads"][0]["record_hex"] = b"value of key 000: a".rjust(32).hex()
    elif corruption == "controller_config":
        state["controller"]["config"]["nkeys"] = 1001
        state["controller"]["config_fingerprint"] = StreamingReplay._fingerprint(state["controller"]["config"])
    elif corruption == "controller_rng": state["controller"]["rng"]["has_uint32"] = 5
    elif corruption == "counter_bool": state["counters"]["prime_records"] = True
    elif corruption == "counter_missing": state["counters"].pop("extra_records")
    elif corruption == "counter_observe": state["counters"]["observe_records"] += 1
    elif corruption == "counter_extra": state["counters"]["extra_records"] += 1
    elif corruption == "counter_prime": state["counters"]["prime_records"] -= 1
    elif corruption == "counter_phase":
        state["counters"]["prime_records"] -= 1
        state["counters"]["observe_records"] += 1
    with pytest.raises(ValueError):
        SequenceReplay.from_state(state)


@pytest.mark.parametrize("arm,freshness", ARMS)
def test_bad_observations_and_closed_or_late_prime_calls_leave_all_state_unchanged(arm, freshness):
    policy = SequenceReplay(arm, 53, freshness=freshness)
    policy.prime(raw(7))
    before = policy.export_state()
    for bad in (None, b"", b" " * 32, raw(8).decode("ascii")):
        for operation in (policy.prime, policy.observe):
            with pytest.raises(ValueError):
                operation(bad)
            assert policy.export_state() == before
    policy.observe(raw(8))
    before = policy.export_state()
    with pytest.raises(RuntimeError):
        policy.prime(raw(9))
    assert policy.export_state() == before
    policy.finish()
    before = policy.export_state()
    for operation in (policy.prime, policy.observe):
        with pytest.raises(RuntimeError):
            operation(raw(9))
        assert policy.export_state() == before


@pytest.mark.parametrize("operation", ["prime", "observe", "finish"])
def test_controller_mutation_then_exception_is_atomic(monkeypatch, operation):
    policy = SequenceReplay("periodic", 18, freshness="bounded")
    policy.prime(raw(1))
    if operation == "finish":
        policy.observe(raw(1, 4))
    before = deepcopy(policy.export_state())
    method = "step" if operation == "observe" else operation
    original = getattr(StreamingReplay, method)

    def fail_after_mutation(controller, *args):
        original(controller, *args)
        raise RuntimeError("injected after candidate controller mutation")

    monkeypatch.setattr(StreamingReplay, method, fail_after_mutation)
    with pytest.raises(RuntimeError, match="injected"):
        getattr(policy, operation)(*(() if operation == "finish" else (raw(2),)))
    assert policy.export_state() == before


def test_payload_resolution_and_poststep_validation_failures_are_atomic(monkeypatch):
    policy = SequenceReplay("continuous", 81, freshness="bounded")
    policy.prime(raw(1))
    before = deepcopy(policy.export_state())
    attach = SequenceReplay._attach

    def fail_attach(events, payloads):
        attach(events, payloads)
        raise ValueError("injected raw resolution failure")

    monkeypatch.setattr(SequenceReplay, "_attach", staticmethod(fail_attach))
    with pytest.raises(ValueError, match="injected"):
        policy.observe(raw(2))
    assert policy.export_state() == before
    monkeypatch.setattr(SequenceReplay, "_attach", staticmethod(attach))
    original = StreamingReplay.step

    def corrupt_after_step(controller, *args):
        events = original(controller, *args)
        controller._buffer[2] = (controller._next_source - 1, 2, 15)
        return events

    monkeypatch.setattr(StreamingReplay, "step", corrupt_after_step)
    with pytest.raises(ValueError, match="Raw observation"):
        policy.observe(raw(2))
    assert policy.export_state() == before


def test_empty_replay_observe_failure_does_not_consume_rng_or_source_and_exports_are_detached():
    policy = SequenceReplay("continuous", 13, freshness="bounded")
    before = policy.export_state()
    with pytest.raises(ValueError, match="prefill"):
        policy.observe(raw(1))
    assert policy.export_state() == before
    policy.prime(raw(1))
    detached = policy.export_state()
    detached["controller"]["config"]["seed"] = 999
    detached["payloads"][0]["record_hex"] = raw(2).hex()
    detached["counters"]["prime_records"] = 99
    config = policy.config
    config["arm"] = "online"
    assert policy.config["seed"] == 13 and policy.config["arm"] == "continuous"
    assert policy.budget()["prime_records"] == 1
    assert policy.observe(raw(1, 4))[1]["record_bytes"] == raw(1, 4)


@pytest.mark.parametrize("arm", ["online", "repeat"])
def test_bounded_only_applies_to_replay_arms(arm):
    with pytest.raises(ValueError):
        SequenceReplay(arm, 1, freshness="bounded")
