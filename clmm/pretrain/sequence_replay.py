"""Bounded raw-record bridge from ``StreamingReplay`` to a full byte LM.

Only observed, fixed-width fact bytes enter this bridge. It retains exact source
bytes for the union of the controller's FIFO8 and original snapshot4; it neither
has an episode/query interface nor reconstructs records from key/value codes.
Replay is additional full-model input, not a memory-only update or sleep rule.
"""

import re

from .kv_scheduler import StreamingReplay


RECORD_BYTES = 32
_GRAMMARS = tuple(re.compile(pattern) for pattern in (
    rb" *fact ([0-9]{3}) has value ([a-p])\.\n",
    rb" *key ([0-9]{3}) has code ([a-p])\.\n",
    rb" *value for ([0-9]{3}) is ([a-p])\.\n",
    rb" *([0-9]{3}) maps to value ([a-p])\.\n",
))


def parse_record_bytes(raw):
    """Parse only a 32-byte, space-left-padded supported fact observation.

    The returned numeric key/value are solely controller routing codes. The
    caller's original bytes, including its template and padding, remain the
    payload. Queries, labels, record objects and non-``bytes`` inputs are rejected.
    """
    if type(raw) is not bytes or len(raw) != RECORD_BYTES:
        raise ValueError("Observation must be exactly 32 fact-record bytes")
    for grammar in _GRAMMARS:
        match = grammar.fullmatch(raw)
        if match:
            return int(match[1]), match[2][0] - ord("a")
    raise ValueError("Observation does not match a supported padded fact grammar")


class SequenceReplay:
    """Attach original raw bytes to the existing causal replay controller.

    ``prime(raw)``, ``observe(raw)`` and ``finish()`` each return a list of
    controller event dictionaries plus ``record_bytes: bytes``. Priming emits
    online events without replay; ``observe`` advances the replay block. Only
    continuous/periodic replay accept bounded freshness. All policy decisions,
    candidate RNG draws and actual source IDs come from ``StreamingReplay``.

    At rest, payload storage contains at most twelve source records (384 raw
    bytes), with no persistent payloads for online/repeat. A transition's lookup
    can temporarily contain those records plus the one new arrival (416 bytes),
    so a block-ending old sample remains resolvable after the controller clears
    its snapshot. Every transition works on a cloned controller and commits only
    after event resolution, pruning and coherence validation all succeed.
    """

    _state_keys = {"schema_version", "payload_encoding", "controller", "payloads", "counters"}
    _counter_keys = {"prime_records", "observe_records", "extra_records"}

    def __init__(self, arm, seed, *, freshness="literal"):
        self._controller = StreamingReplay(arm, seed, capacity=8, block_size=4,
                                           nkeys=1000, nvalues=16, freshness=freshness)
        self._payloads = {}
        self._counters = dict(prime_records=0, observe_records=0, extra_records=0)

    @property
    def freshness(self):
        return self._controller.freshness

    @property
    def config(self):
        """A detached controller configuration, including its literal/v2 choice."""
        return dict(self._controller.config)

    @staticmethod
    def _references(controller_state):
        by_source = {}
        for source, key, value in controller_state["buffer"] + controller_state["snapshot"]:
            if source in by_source and by_source[source] != (key, value):
                raise ValueError("Controller disagrees about a retained source")
            by_source[source] = (key, value)
        return by_source

    @classmethod
    def _validate(cls, controller_state, payloads, counters):
        config = controller_state["config"]
        if any(config[name] != value for name, value in
               (("capacity", 8), ("block_size", 4), ("nkeys", 1000), ("nvalues", 16))):
            raise ValueError("Sequence replay requires FIFO8/snapshot4 and 1000/16 codes")
        if (type(counters) is not dict or set(counters) != cls._counter_keys
                or any(type(value) is not int or value < 0 for value in counters.values())):
            raise ValueError("Invalid sequence replay counters")
        primed, observed, extras = (counters[name] for name in
                                   ("prime_records", "observe_records", "extra_records"))
        replay = config["arm"] in ("continuous", "periodic")
        offset = controller_state["in_block"]
        expected_extras = (0 if config["arm"] == "online" else observed -
                           (offset if config["arm"] == "periodic" else 0))
        expected_offset = observed % 4 if replay and not controller_state["closed"] else 0
        if (primed + observed != controller_state["next_source"]
                or bool(observed) != controller_state["started"]
                or replay and observed and not primed
                or offset != expected_offset or extras != expected_extras):
            raise ValueError("Sequence counters disagree with controller phase or write budget")
        references = cls._references(controller_state)
        if (type(payloads) is not dict or len(payloads) > 12
                or any(type(source) is not int or source < 0 for source in payloads)
                or set(payloads) != set(references)):
            raise ValueError("Raw payloads must exactly cover the bounded FIFO/snapshot union")
        for source, raw in payloads.items():
            if parse_record_bytes(raw) != references[source]:
                raise ValueError("Raw observation disagrees with its controller source/key/value")

    @staticmethod
    def _attach(events, payloads):
        result = []
        for event in events:
            source = event["source"]
            if (type(source) is not int or source < 0
                    or type(event["after_online"]) is not int
                    or source > event["after_online"] or source not in payloads):
                raise ValueError("Emitted source has no causal observed raw payload")
            raw = payloads[source]
            if parse_record_bytes(raw) != (event["key"], event["value"]):
                raise ValueError("Emitted source disagrees with its observed raw payload")
            result.append(dict(event, record_bytes=raw))
        return result

    def _transition(self, operation, raw=None):
        codes = None if operation == "finish" else parse_record_bytes(raw)
        before = self._controller.export_state()
        controller = StreamingReplay.from_state(before)
        self._validate(before, self._payloads, self._counters)
        payloads, counters = dict(self._payloads), dict(self._counters)
        if codes is not None:
            payloads[before["next_source"]] = raw
        if len(payloads) > 13:
            raise ValueError("Transition exceeded its bounded raw lookup")
        if operation == "prime":
            events = [controller.prime(*codes)]
            counters["prime_records"] += 1
        elif operation == "observe":
            events = controller.step(*codes)
            counters["observe_records"] += 1
        elif operation == "finish":
            events = controller.finish()
        else:
            raise ValueError("Unknown sequence replay operation")
        attached = self._attach(events, payloads)
        counters["extra_records"] += sum(event["kind"] == "extra" for event in events)
        after = controller.export_state()
        retained = self._references(after)
        payloads = {source: payloads[source] for source in retained}
        self._validate(after, payloads, counters)
        self._controller, self._payloads, self._counters = controller, payloads, counters
        return attached

    def prime(self, raw):
        """Accept one pre-delay real record and return its online event in a list."""
        return self._transition("prime", raw)

    def observe(self, raw):
        """Accept one real fact-record byte string and resolve causal replay events."""
        return self._transition("observe", raw)

    def finish(self):
        """Flush a partial periodic block, prune its raw snapshot and close."""
        return self._transition("finish")

    def logical_counts(self):
        """Separate controller record slots from physical retained raw payloads."""
        pairs = self._controller.logical_counts()
        return dict(pairs, controller_records=sum(pairs.values()),
                    raw_payload_records=len(self._payloads),
                    raw_payload_bytes=RECORD_BYTES * len(self._payloads))

    def budget(self):
        """Current storage and cumulative writes; online includes primed records.

        Raw byte counts exclude Python/JSON/controller metadata overhead. Controller
        record slots can duplicate sources, whereas raw payloads are deduplicated
        by source. Serialized file sizes are therefore a separate resource count.
        """
        counts = self.logical_counts()
        online = self._counters["prime_records"] + self._counters["observe_records"]
        return dict(counts, **self._counters, online_records=online,
                    emitted_records=online + self._counters["extra_records"])

    def export_state(self):
        """Detach a JSON-compatible v1 bridge state around the core v1/v2 state."""
        return dict(schema_version=1, payload_encoding="hex32",
                    controller=self._controller.export_state(),
                    payloads=[dict(source=source, record_hex=raw.hex())
                              for source, raw in sorted(self._payloads.items())],
                    counters=dict(self._counters))

    @classmethod
    def from_state(cls, state):
        """Restore only schema-, policy-, phase- and raw-source-coherent state."""
        if (type(state) is not dict or set(state) != cls._state_keys
                or type(state["schema_version"]) is not int or state["schema_version"] != 1
                or state["payload_encoding"] != "hex32"):
            raise ValueError("Invalid sequence replay schema or payload encoding")
        controller = StreamingReplay.from_state(state["controller"])
        entries = state["payloads"]
        if type(entries) is not list or len(entries) > 12:
            raise ValueError("Invalid bounded raw payload list")
        payloads = {}
        for entry in entries:
            if (type(entry) is not dict or set(entry) != {"source", "record_hex"}
                    or type(entry["source"]) is not int or entry["source"] < 0
                    or entry["source"] in payloads or type(entry["record_hex"]) is not str
                    or len(entry["record_hex"]) != RECORD_BYTES * 2):
                raise ValueError("Invalid raw payload source or hex record")
            try:
                raw = bytes.fromhex(entry["record_hex"])
            except ValueError as error:
                raise ValueError("Invalid hex-encoded raw observation") from error
            if raw.hex() != entry["record_hex"]:
                raise ValueError("Raw payload hex encoding must be canonical lowercase")
            payloads[entry["source"]] = raw
        counters = state["counters"]
        cls._validate(controller.export_state(), payloads, counters)
        result = cls.__new__(cls)
        result._controller, result._payloads, result._counters = controller, payloads, dict(counters)
        return result
