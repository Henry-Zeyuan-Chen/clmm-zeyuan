"""Bounded reference wake/sleep orchestration, independent of the update rule.

The supplied model owns its slow checkpoint and functional fast-state API. This
adapter changes only when already-observed pairs are written. Exporting a
session never stores a complete observation history or any query answers.
"""
from __future__ import annotations

import torch

from clmm.pretrain.kv_control import physical_state_bytes
from clmm.pretrain.kv_scheduler import StreamingReplay


class SleepSession:
    """Batched wake/replay session with atomic model-plus-policy handoffs.

    Freeze slow parameters externally for evaluation. The session deliberately
    preserves autograd so the same interface can support outer-loop training.
    Backend writes must return a new state without mutating input state or slow
    parameters; this permits failed candidates to be discarded atomically.
    Resume with the corresponding slow checkpoint: the configuration fingerprint
    identifies the architecture/codebook, not the values of every slow weight.
    """
    _fields = {"schema_version", "model_method", "model_config_fingerprint", "batch_size",
               "arm", "capacity", "block_size", "episode_seeds", "prime_count",
               "delay_count", "replay_count", "closed", "model_state", "policies"}

    def __init__(self, model, batch_size, arm, episode_seeds, capacity=8, block_size=4):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(episode_seeds, (list, tuple)) or len(episode_seeds) != batch_size:
            raise ValueError("episode_seeds must contain one integer seed per stream")
        self.model, self.batch_size = model, batch_size
        self.arm, self.capacity, self.block_size = arm, capacity, block_size
        self.episode_seeds = list(episode_seeds)
        self.policies = [StreamingReplay(arm, seed, capacity, block_size,
                                        model.config.nkeys, model.config.nvalues)
                         for seed in episode_seeds]
        self.state = model.initial_state(batch_size)
        self.prime_count = self.delay_count = self.replay_count = 0
        self.closed = False
        self._check_coherence()

    def _check_coherence(self):
        if (type(self.arm) is not str or self.arm not in StreamingReplay._arms
                or type(self.capacity) is not int or not 1 <= self.capacity <= 8
                or type(self.block_size) is not int or not 1 <= self.block_size <= 4
                or type(self.episode_seeds) is not list or len(self.episode_seeds) != self.batch_size
                or any(type(seed) is not int or seed < 0 for seed in self.episode_seeds)):
            raise ValueError("Invalid session policy configuration or episode seeds")
        if (self.state.batch_size != self.batch_size
                or self.state.method != self.model.method
                or self.state.config_fingerprint != self.model.config.fingerprint
                or self.state.write_count != self.prime_count + self.delay_count + self.replay_count):
            raise ValueError("Model state disagrees with the session identity or write counters")
        offsets = []
        for seed, policy in zip(self.episode_seeds, self.policies):
            payload = policy.export_state()
            expected = dict(arm=self.arm, seed=seed, capacity=self.capacity,
                            block_size=self.block_size, nkeys=self.model.config.nkeys,
                            nvalues=self.model.config.nvalues)
            if (payload["config"] != expected
                    or payload["next_source"] != self.prime_count + self.delay_count
                    or payload["closed"] != self.closed
                    or payload["started"] != bool(self.delay_count)):
                raise ValueError("Replay policy identity, counters or lifecycle disagree with the session")
            offsets.append(payload["in_block"])
        if len(self.policies) != self.batch_size or len(set(offsets)) != 1:
            raise ValueError("All batch policies must have the same replay phase")
        pending = offsets[0]
        expected_offset = (self.delay_count % self.block_size
                           if self.arm in ("continuous", "periodic") and not self.closed else 0)
        expected_replay = (0 if self.arm == "online" else
                           self.delay_count - pending if self.arm == "periodic" else self.delay_count)
        if pending != expected_offset or self.replay_count != expected_replay:
            raise ValueError("Replay budget does not match the completed block phase")
        if self.arm in ("continuous", "periodic") and self.delay_count and not self.prime_count:
            raise ValueError("Buffered replay requires an observed prefill")

    def _input_ids(self, keys, values):
        device = next(iter(self.state.weights.values())).device
        for name, tensor, upper in (("keys", keys, self.model.config.nkeys),
                                    ("values", values, self.model.config.nvalues)):
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.long
                    or tensor.shape != (self.batch_size,) or tensor.device != device):
                raise ValueError(f"{name} must be an int64 [batch] tensor on the fast-state device")
            if ((tensor < 0) | (tensor >= upper)).any():
                raise ValueError(f"{name} contain an out-of-range ID")
        return keys.detach().cpu().tolist(), values.detach().cpu().tolist()

    def _apply(self, events):
        if len(events) != self.batch_size or len({len(item) for item in events}) != 1:
            raise ValueError("A batched session requires rectangular emitted writes")
        state = self.state
        device = next(iter(state.weights.values())).device
        for offset in range(len(events[0])):
            keys = torch.tensor([item[offset]["key"] for item in events], device=device, dtype=torch.long)
            values = torch.tensor([item[offset]["value"] for item in events], device=device, dtype=torch.long)
            state = self.model.write(keys, values, state)
        return state

    def _candidate(self, state, policies, *, prime_count=None, delay_count=None,
                   replay_count=None, closed=None):
        candidate = type(self).__new__(type(self))
        candidate.__dict__ = dict(self.__dict__)
        candidate.state, candidate.policies = state, policies
        for name, value in (("prime_count", prime_count), ("delay_count", delay_count),
                            ("replay_count", replay_count), ("closed", closed)):
            if value is not None:
                setattr(candidate, name, value)
        candidate._check_coherence()
        return candidate

    def _arrive(self, keys, values, *, prime):
        self._check_coherence()
        if self.closed:
            raise RuntimeError("Sleep session is closed")
        key_ids, value_ids = self._input_ids(keys, values)
        # Only bounded policy records are copied. Commit neither policy nor fast
        # state until every emitted model write has completed successfully.
        policies = [StreamingReplay.from_state(policy.export_state()) for policy in self.policies]
        events = [[policy.prime(key, value)] if prime else policy.step(key, value)
                  for policy, key, value in zip(policies, key_ids, value_ids)]
        state = self._apply(events)
        extra = sum(item["kind"] == "extra" for item in events[0])
        if state.write_count != self.state.write_count + len(events[0]):
            raise ValueError("Model write counter does not match emitted events")
        candidate = self._candidate(state, policies, prime_count=self.prime_count + int(prime),
                                    delay_count=self.delay_count + int(not prime),
                                    replay_count=self.replay_count + extra)
        self.__dict__.update(candidate.__dict__)
        return events

    def prime(self, keys, values):
        """Consume a real observation in the common online-only prefill."""
        return self._arrive(keys, values, prime=True)

    def observe(self, keys, values):
        """Consume one delay arrival and its scheduled memory-only replay."""
        return self._arrive(keys, values, prime=False)

    def finish(self):
        """Flush a terminal partial periodic block; export instead to pause."""
        self._check_coherence()
        if self.closed:
            return [[] for _ in self.policies]
        policies = [StreamingReplay.from_state(policy.export_state()) for policy in self.policies]
        events = [policy.finish() for policy in policies]
        state = self._apply(events)
        if state.write_count != self.state.write_count + len(events[0]):
            raise ValueError("Model write counter does not match the terminal flush")
        candidate = self._candidate(state, policies, replay_count=self.replay_count + len(events[0]), closed=True)
        self.__dict__.update(candidate.__dict__)
        return events

    def read(self, query_keys):
        """Read keys only. The model contract requires a side-effect-free read."""
        return self.model.read(query_keys, self.state)

    def detach_state(self):
        self.state = self.model.detach_state(self.state)
        return self.state

    def budget(self):
        """Per-stream opportunities and literal records; tensor bytes separate."""
        self._check_coherence()
        return dict(batch_size=self.batch_size, arm=self.arm, prime_writes=self.prime_count,
                    delay_observations=self.delay_count, online_observations=self.prime_count + self.delay_count,
                    replay_writes=self.replay_count, total_writes=self.state.write_count,
                    buffer_pairs=[policy.logical_counts()["buffer_pairs"] for policy in self.policies],
                    snapshot_pairs=[policy.logical_counts()["snapshot_pairs"] for policy in self.policies],
                    fast_tensor_bytes=physical_state_bytes(self.state),
                    per_stream_fast_tensor_bytes=physical_state_bytes(self.state) / self.batch_size,
                    closed=self.closed)

    def export_state(self):
        self._check_coherence()
        return dict(schema_version=1, model_method=self.model.method,
                    model_config_fingerprint=self.model.config.fingerprint, batch_size=self.batch_size,
                    arm=self.arm, capacity=self.capacity, block_size=self.block_size,
                    episode_seeds=list(self.episode_seeds), prime_count=self.prime_count,
                    delay_count=self.delay_count, replay_count=self.replay_count, closed=self.closed,
                    model_state=self.model.export_state(self.state),
                    policies=[policy.export_state() for policy in self.policies])

    @classmethod
    def from_state(cls, model, payload):
        """Prepare/validate a complete paused session without changing the model."""
        if (type(payload) is not dict or set(payload) != cls._fields
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or payload["model_method"] != model.method
                or payload["model_config_fingerprint"] != model.config.fingerprint):
            raise ValueError("Sleep-session schema or model identity mismatch")
        for name in ("batch_size", "prime_count", "delay_count", "replay_count"):
            if type(payload[name]) is not int or payload[name] < (1 if name == "batch_size" else 0):
                raise ValueError(f"Invalid session {name}")
        if (type(payload["closed"]) is not bool or type(payload["policies"]) is not list
                or type(payload["episode_seeds"]) is not list
                or len(payload["policies"]) != payload["batch_size"]
                or len(payload["episode_seeds"]) != payload["batch_size"]):
            raise ValueError("Invalid batched policies, seeds or lifecycle")
        session = cls.__new__(cls)
        session.model = model
        for name in ("batch_size", "arm", "capacity", "block_size", "prime_count",
                     "delay_count", "replay_count", "closed"):
            setattr(session, name, payload[name])
        session.episode_seeds = list(payload["episode_seeds"])
        session.policies = [StreamingReplay.from_state(item) for item in payload["policies"]]
        session.state = model.import_state(payload["model_state"])
        session._check_coherence()
        return session
