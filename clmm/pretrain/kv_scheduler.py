"""Bounded live replay policies over literal records, without episode access."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json

import numpy as np


class StreamingReplay:
    """Emit online and replay events using at most eight plus four records.

    Only real arrivals refresh the latest-per-key FIFO. Continuous and periodic
    policies draw a frozen block snapshot before its first arrival. Online and
    repeat policies retain neither a FIFO nor a snapshot.
    """
    _arms = ("online", "repeat", "continuous", "periodic")
    _config_keys = {"arm", "seed", "capacity", "block_size", "nkeys", "nvalues"}
    _state_keys = {"schema_version", "config", "config_fingerprint", "next_source",
                   "buffer", "snapshot", "in_block", "rng", "started", "closed"}

    def __init__(self, arm, seed, capacity=8, block_size=4, nkeys=64, nvalues=16):
        if type(arm) is not str or arm not in self._arms:
            raise ValueError("Unknown replay arm")
        for name, value in (("seed", seed), ("capacity", capacity), ("block_size", block_size),
                            ("nkeys", nkeys), ("nvalues", nvalues)):
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"Invalid {name}")
        if capacity > 8 or block_size > 4:
            raise ValueError("Live replay is bounded to eight FIFO and four snapshot records")
        self.config = dict(arm=arm, seed=seed, capacity=capacity, block_size=block_size,
                           nkeys=nkeys, nvalues=nvalues)
        self._rng = np.random.default_rng(np.random.SeedSequence([seed, 77]))
        self._buffer = OrderedDict()
        self._snapshot = []
        self._next_source = 0
        self._in_block = 0
        self._started = False
        self._closed = False

    @staticmethod
    def _fingerprint(config):
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _ids(self, key, value):
        if (type(key) is not int or not 0 <= key < self.config["nkeys"]
                or type(value) is not int or not 0 <= value < self.config["nvalues"]):
            raise ValueError("Key and value must be in-range integer codes")

    def _arrive(self, key, value):
        record = (self._next_source, key, value)
        self._next_source += 1
        if self.config["arm"] in ("continuous", "periodic"):
            self._buffer.pop(key, None)
            self._buffer[key] = record
            while len(self._buffer) > self.config["capacity"]:
                self._buffer.popitem(last=False)
        return record

    @staticmethod
    def _event(record, kind, after_online):
        source, key, value = record
        return dict(source=source, key=key, value=value, kind=kind, after_online=after_online)

    def prime(self, key, value):
        """Accept one pre-delay arrival. Priming closes at the first step."""
        if self._closed or self._started:
            raise RuntimeError("Cannot prime a closed or already-started replay policy")
        self._ids(key, value)
        record = self._arrive(key, value)
        return self._event(record, "online", record[0])

    def step(self, key, value):
        if self._closed:
            raise RuntimeError("Replay policy is closed")
        self._ids(key, value)
        arm = self.config["arm"]
        if arm in ("continuous", "periodic") and self._in_block == 0:
            if not self._buffer:
                raise ValueError("Replay requires a nonempty prefill FIFO")
            # This temporary lookup contains only the bounded live FIFO.
            available = {record[0]: record for record in self._buffer.values()}
            source_ids = list(available)
            self._snapshot = [available[int(self._rng.choice(source_ids))]
                              for _ in range(self.config["block_size"])]
        self._started = True
        record = self._arrive(key, value)
        events = [self._event(record, "online", record[0])]
        if arm == "repeat":
            events.append(self._event(record, "extra", record[0]))
        elif arm in ("continuous", "periodic"):
            if arm == "continuous":
                events.append(self._event(self._snapshot[self._in_block], "extra", record[0]))
            self._in_block += 1
            if self._in_block == self.config["block_size"]:
                if arm == "periodic":
                    events.extend(self._event(item, "extra", record[0]) for item in self._snapshot)
                self._snapshot = []
                self._in_block = 0
        return events

    def finish(self):
        """Flush a terminal partial periodic block and close the policy."""
        if self._closed:
            return []
        events = []
        if self.config["arm"] == "periodic" and self._in_block:
            events = [self._event(record, "extra", self._next_source - 1)
                      for record in self._snapshot[:self._in_block]]
        self._snapshot = []
        self._in_block = 0
        self._closed = True
        return events

    def logical_counts(self):
        return dict(buffer_pairs=len(self._buffer), snapshot_pairs=len(self._snapshot))

    def export_state(self):
        return dict(schema_version=1, config=dict(self.config), config_fingerprint=self._fingerprint(self.config),
                    next_source=self._next_source, buffer=[list(item) for item in self._buffer.values()],
                    snapshot=[list(item) for item in self._snapshot], in_block=self._in_block,
                    rng=deepcopy(self._rng.bit_generator.state), started=self._started, closed=self._closed)

    @classmethod
    def from_state(cls, payload):
        if (type(payload) is not dict or set(payload) != cls._state_keys
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or type(payload["config"]) is not dict or set(payload["config"]) != cls._config_keys
                or payload["config_fingerprint"] != cls._fingerprint(payload["config"])):
            raise ValueError("Invalid or mismatched replay-state schema/configuration")
        config = payload["config"]
        policy = cls(**config)
        count, offset = payload["next_source"], payload["in_block"]
        if (type(count) is not int or count < 0 or type(offset) is not int
                or not 0 <= offset < config["block_size"]
                or type(payload["started"]) is not bool or type(payload["closed"]) is not bool):
            raise ValueError("Invalid replay counters or flags")

        def records(name, limit):
            entries = payload[name]
            if type(entries) is not list or len(entries) > limit:
                raise ValueError(f"Invalid bounded {name}")
            result = []
            for item in entries:
                if (type(item) is not list or len(item) != 3 or type(item[0]) is not int
                        or not 0 <= item[0] < count):
                    raise ValueError(f"Invalid {name} source record")
                policy._ids(item[1], item[2])
                result.append(tuple(item))
            return result

        buffer = records("buffer", config["capacity"])
        snapshot = records("snapshot", config["block_size"])
        if (len({item[1] for item in buffer}) != len(buffer)
                or [item[0] for item in buffer] != sorted({item[0] for item in buffer})):
            raise ValueError("FIFO records must have unique keys and increasing source IDs")
        replay = config["arm"] in ("continuous", "periodic")
        if not replay and (buffer or snapshot or offset):
            raise ValueError("Online and repeat policies must have no replay buffers")
        if replay and bool(buffer) != bool(count):
            raise ValueError("FIFO occupancy does not match the arrival counter")
        if replay and buffer and buffer[-1][0] != count - 1:
            raise ValueError("FIFO must end with the latest real arrival")
        if ((payload["closed"] or not payload["started"]) and (offset or snapshot)
                or payload["started"] and count == 0
                or replay and len(snapshot) != (config["block_size"] if offset else 0)):
            raise ValueError("Replay block state disagrees with its phase flags")
        by_source, by_key = {}, {}
        for item in snapshot:
            if item[0] >= count - offset:
                raise ValueError("Snapshot includes a current or future block arrival")
            if ((item[0] in by_source and by_source[item[0]] != item)
                    or (item[1] in by_key and by_key[item[1]] != item)):
                raise ValueError("Snapshot records are inconsistent with latest-per-key sampling")
            by_source[item[0]], by_key[item[1]] = item, item
        for item in buffer:
            if item[0] in by_source and by_source[item[0]] != item:
                raise ValueError("FIFO and snapshot disagree about a literal source record")
            if item[0] < count - offset and item[1] in by_key and by_key[item[1]] != item:
                raise ValueError("Snapshot was not latest-per-key at block start")
        rng = payload["rng"]
        initial_rng = policy._rng.bit_generator.state
        if (type(rng) is not dict or set(rng) != {"bit_generator", "state", "has_uint32", "uinteger"}
                or rng["bit_generator"] != "PCG64" or type(rng["state"]) is not dict
                or set(rng["state"]) != {"state", "inc"}
                or any(type(rng["state"][key]) is not int or not 0 <= rng["state"][key] < 2**128
                       for key in ("state", "inc"))
                or rng["state"]["inc"] != initial_rng["state"]["inc"]
                or type(rng["has_uint32"]) is not int or rng["has_uint32"] not in (0, 1)
                or type(rng["uinteger"]) is not int or not 0 <= rng["uinteger"] < 2**32):
            raise ValueError("Invalid PCG64 state or seed stream")
        if (not replay or not payload["started"]) and rng != initial_rng:
            raise ValueError("RNG advanced before any replay block")
        policy._buffer = OrderedDict((item[1], item) for item in buffer)
        policy._snapshot = snapshot
        policy._next_source, policy._in_block = count, offset
        policy._started, policy._closed = payload["started"], payload["closed"]
        policy._rng.bit_generator.state = deepcopy(rng)
        return policy
