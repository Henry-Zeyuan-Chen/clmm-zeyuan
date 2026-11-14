"""Bounded synthetic fact streams and causal raw-record replay for a byte LM.

Every online/replayed record occupies one 32-byte attention segment. The final
31-byte query occupies its own segment, and its answer is the next byte. Replay
is additional input to the full model; this module does not implement memory
updates or a sleep algorithm. Targets always come from the real online stream.

The primary dev comparison uses ``template_family='seen'``. ``'heldout'`` is a
separate template-generalization condition, never a training template family.
The full three-digit key, not an individual digit, defines the split namespace.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random


FORMAT_VERSION = 1
RECORD_BYTES = 32
QUERY_PREFIX_BYTES = 31
VALUES = "abcdefghijklmnop"
CONDITIONS = ("online", "continuous", "periodic", "repeat_current")
SPLIT_BOUNDS = {"train": (0, 600), "dev": (600, 800), "test": (800, 1000)}
KEY_POOL_SCHEME = "sha256(clmm.fact.keys.v1:+key), key tie-break; ranked 600/200/200"
_KEY_ORDER = sorted((f"{index:03d}" for index in range(1000)),
                    key=lambda key: (hashlib.sha256(("clmm.fact.keys.v1:" + key).encode("ascii")).digest(), key))
KEY_POOLS = {split: tuple(_KEY_ORDER[lower:upper]) for split, (lower, upper) in SPLIT_BOUNDS.items()}
KEY_POOL_HASHES = {split: hashlib.sha256("\n".join(pool).encode("ascii")).hexdigest()
                   for split, pool in KEY_POOLS.items()}
RECORD_TEMPLATES = {
    "seen": ("fact {key} has value {value}.\n", "key {key} has code {value}.\n"),
    "heldout": ("value for {key} is {value}.\n", "{key} maps to value {value}.\n"),
}
QUERY_TEMPLATES = {
    "seen": ("value of key {key}: ", "code for key {key} = "),
    "heldout": ("recall value for {key}: ", "key {key} now has value = "),
}


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _key(key):
    if not isinstance(key, str) or len(key) != 3 or any(c not in "0123456789" for c in key):
        raise ValueError("key must be exactly three ASCII digits")


def _value(value, name="value"):
    if not isinstance(value, str) or len(value) != 1 or value not in VALUES:
        raise ValueError(f"{name} must be one of {VALUES}")


def _template(family, index):
    if family not in RECORD_TEMPLATES:
        raise ValueError("template_family must be seen or heldout")
    _integer("template_index", index)
    if index >= len(RECORD_TEMPLATES[family]):
        raise ValueError("template_index must be 0 or 1")


def _fraction(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("revision_fraction must be finite and between 0 and 1")


def _rng(seed, domain):
    """Domain-separated local RNGs keep semantics fixed across template changes."""
    digest = hashlib.sha256(f"clmm-facts-v1:{domain}:{seed}".encode("ascii")).digest()
    return random.Random(int.from_bytes(digest, "big"))


@dataclass(frozen=True)
class FactRecord:
    event_id: int
    key: str
    value: str
    version: int
    kind: str
    template_family: str
    template_index: int

    def __post_init__(self):
        _integer("event_id", self.event_id)
        _key(self.key)
        _value(self.value)
        _integer("version", self.version, 1)
        if self.kind not in ("fact", "revision", "distractor"):
            raise ValueError("record kind must be fact, revision or distractor")
        if self.version != (2 if self.kind == "revision" else 1):
            raise ValueError("Only revisions have version 2; other records have version 1")
        _template(self.template_family, self.template_index)

    @property
    def text(self):
        text = RECORD_TEMPLATES[self.template_family][self.template_index].format(
            key=self.key, value=self.value)
        encoded = text.encode("ascii")
        if len(encoded) > RECORD_BYTES or not encoded.endswith(self.value.encode("ascii") + b".\n"):
            raise ValueError("Record template violates fixed-width value placement")
        return text.rjust(RECORD_BYTES)

    @property
    def encoded(self):
        return self.text.encode("ascii")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FactQuery:
    key: str
    target: str
    old_value: str | None
    category: str
    initial_event_id: int
    latest_event_id: int
    template_family: str
    template_index: int

    def __post_init__(self):
        _key(self.key)
        _value(self.target, "target")
        _integer("initial_event_id", self.initial_event_id)
        _integer("latest_event_id", self.latest_event_id)
        if self.latest_event_id < self.initial_event_id:
            raise ValueError("Query latest_event_id precedes initial_event_id")
        if self.category not in ("stable", "revised", "recent"):
            raise ValueError("Query category must be stable, revised or recent")
        if self.category == "revised":
            _value(self.old_value, "old_value")
            if self.old_value == self.target or self.latest_event_id == self.initial_event_id:
                raise ValueError("A revised query needs a different old value and a later event")
        elif self.old_value is not None or self.latest_event_id != self.initial_event_id:
            raise ValueError("Stable/recent queries have no old value and one source event")
        _template(self.template_family, self.template_index)

    @property
    def answer_byte(self):
        return ord(self.target)

    def to_dict(self):
        return asdict(self)


def render_query(query, template_index=None, template_family=None):
    """Return a label-independent 31-byte prefix ending in ``': '`` or ``'= '``.

    The target letter may occur in ordinary instruction words; the label is
    never interpolated. Changing only a query's target cannot change its text.
    """
    if not isinstance(query, FactQuery):
        raise TypeError("query must be a FactQuery")
    family = query.template_family if template_family is None else template_family
    index = query.template_index if template_index is None else template_index
    _template(family, index)
    text = QUERY_TEMPLATES[family][index].format(key=query.key).encode("ascii")
    if len(text) > QUERY_PREFIX_BYTES or not text.endswith((b": ", b"= ")):
        raise ValueError("Query template violates fixed-width prefix placement")
    return text.rjust(QUERY_PREFIX_BYTES, b" ")


@dataclass(frozen=True)
class Episode:
    seed: int
    facts: int
    delay: int
    split: str
    template_family: str
    revision_fraction: float
    include_recent: bool
    events: tuple[FactRecord, ...]
    queries: tuple[FactQuery, ...]

    def __post_init__(self):
        _integer("seed", self.seed)
        _integer("facts", self.facts, 1)
        _integer("delay", self.delay)
        if self.split not in SPLIT_BOUNDS:
            raise ValueError("split must be train, dev or test")
        _template(self.template_family, 0)
        if self.split == "train" and self.template_family != "seen":
            raise ValueError("Training episodes must use seen templates")
        _fraction(self.revision_fraction)
        if type(self.include_recent) is not bool:
            raise ValueError("include_recent must be Boolean")
        pool = KEY_POOLS[self.split]
        if self.facts + self.delay > len(pool):
            raise ValueError("facts + delay exceeds the split's unique-key capacity")
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "queries", tuple(self.queries))
        revisions = math.floor(self.facts * self.revision_fraction)
        if len(self.events) != self.facts + revisions + self.delay:
            raise ValueError("Online event count does not match facts, revisions and delay")
        initial, latest, revised, distractors = {}, {}, set(), []
        for index, record in enumerate(self.events):
            if not isinstance(record, FactRecord) or record.event_id != index:
                raise ValueError("Online event IDs must be consecutive source indices")
            if record.key not in pool:
                raise ValueError("Event key is outside the split namespace")
            if record.template_family != self.template_family:
                raise ValueError("Event template family disagrees with episode")
            expected = "fact" if index < self.facts else (
                "revision" if index < self.facts + revisions else "distractor")
            if record.kind != expected:
                raise ValueError("Online events must be facts, then revisions, then distractors")
            if record.kind == "revision":
                if record.key not in initial or record.key in revised:
                    raise ValueError("Revisions must update distinct initial keys exactly once")
                if record.value == initial[record.key].value:
                    raise ValueError("A revision must differ from the initial value")
                revised.add(record.key)
            else:
                if record.key in latest:
                    raise ValueError("Initial and distractor keys must be unique with no overlap")
                if record.kind == "fact":
                    initial[record.key] = record
                else:
                    distractors.append(record)
            latest[record.key] = record
            if len(record.encoded) != RECORD_BYTES:
                raise ValueError("Online record is not exactly 32 bytes")
        expected_keys = list(initial)
        if self.include_recent and distractors:
            expected_keys.append(distractors[-1].key)
        if len(self.queries) != len(expected_keys):
            raise ValueError("Queries must cover each initial key and the optional recent key")
        for key, query in zip(expected_keys, self.queries):
            if not isinstance(query, FactQuery) or query.key != key:
                raise ValueError("Queries must follow the initial-key order and optional recent key")
            source = initial.get(key, latest[key])
            final = latest[key]
            category = "recent" if key not in initial else ("revised" if key in revised else "stable")
            old_value = source.value if category == "revised" else None
            if (query.target, query.old_value, query.category, query.initial_event_id, query.latest_event_id) != (
                final.value, old_value, category, source.event_id, final.event_id
            ):
                raise ValueError("Query target/version must be derived only from online events")
            if query.template_family != self.template_family:
                raise ValueError("Query template family disagrees with episode")
            render_query(query)

    def to_dict(self):
        return {
            "format_version": FORMAT_VERSION, "seed": self.seed, "facts": self.facts,
            "delay": self.delay, "split": self.split, "template_family": self.template_family,
            "revision_fraction": self.revision_fraction, "include_recent": self.include_recent,
            "key_pool_scheme": KEY_POOL_SCHEME, "key_pool_sha256": KEY_POOL_HASHES[self.split],
            "events": [record.to_dict() for record in self.events],
            "queries": [query.to_dict() for query in self.queries],
        }

    @classmethod
    def from_dict(cls, data):
        required = {"format_version", "seed", "facts", "delay", "split", "template_family",
                    "revision_fraction", "include_recent", "key_pool_scheme", "key_pool_sha256",
                    "events", "queries"}
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Episode JSON has missing or unexpected fields")
        if type(data["format_version"]) is not int or data["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported episode format_version")
        if (data["key_pool_scheme"] != KEY_POOL_SCHEME or data["split"] not in KEY_POOLS
                or data["key_pool_sha256"] != KEY_POOL_HASHES[data["split"]]):
            raise ValueError("Episode key pool scheme or hash differs from the fixed protocol")
        if not isinstance(data["events"], list) or not isinstance(data["queries"], list):
            raise ValueError("Episode events and queries must be JSON arrays")
        try:
            return cls(**{key: value for key, value in data.items()
                          if key not in ("format_version", "events", "queries", "key_pool_scheme", "key_pool_sha256")},
                       events=tuple(FactRecord(**record) for record in data["events"]),
                       queries=tuple(FactQuery(**query) for query in data["queries"]))
        except (TypeError, KeyError) as error:
            raise ValueError("Invalid episode record or query fields") from error


def make_episode(seed, facts, delay, split, template_family="seen", revision_fraction=0.5, *,
                 include_recent=True):
    """Generate facts -> distinct-key revisions -> unique distractor records.

    Initial values and distractor values are uniform over 16 letters. Revised
    values are uniform over the other 15 letters. Keys have no persistent value
    assignment across episode seeds. ``delay`` counts distractor records. A full
    key permutation is generated before slicing, preserving prefixes across D.
    """
    _integer("seed", seed)
    _integer("facts", facts, 1)
    _integer("delay", delay)
    if split not in SPLIT_BOUNDS:
        raise ValueError("split must be train, dev or test")
    _template(template_family, 0)
    if split == "train" and template_family != "seen":
        raise ValueError("Training episodes must use seen templates")
    _fraction(revision_fraction)
    if type(include_recent) is not bool:
        raise ValueError("include_recent must be Boolean")
    pool = KEY_POOLS[split]
    if facts + delay > len(pool):
        raise ValueError("facts + delay exceeds the split's unique-key capacity")
    keys = _rng(seed, "keys").sample(pool, len(pool))[:facts + delay]
    value_rng, template_rng = _rng(seed, "values"), _rng(seed, "record-templates")
    events = []

    def append(key, value, kind):
        record = FactRecord(len(events), key, value, 2 if kind == "revision" else 1,
                            kind, template_family, template_rng.randrange(2))
        events.append(record)

    for key in keys[:facts]:
        append(key, value_rng.choice(VALUES), "fact")
    for index in _rng(seed, "revisions").sample(range(facts), math.floor(facts * revision_fraction)):
        initial = events[index]
        alternatives = VALUES.replace(initial.value, "")
        append(initial.key, value_rng.choice(alternatives), "revision")
    for key in keys[facts:]:
        append(key, value_rng.choice(VALUES), "distractor")
    latest = {event.key: event for event in events}
    query_rng, queries = _rng(seed, "query-templates"), []
    sources = events[:facts] + ([events[-1]] if include_recent and delay else [])
    for source in sources:
        final = latest[source.key]
        category = "recent" if source.kind == "distractor" else (
            "revised" if final.version == 2 else "stable")
        queries.append(FactQuery(source.key, final.value, source.value if category == "revised" else None,
                                 category, source.event_id, final.event_id, template_family,
                                 query_rng.randrange(2)))
    return Episode(seed, facts, delay, split, template_family, revision_fraction,
                   include_recent, tuple(events), tuple(queries))


@dataclass(frozen=True)
class ScheduledEvent:
    record: FactRecord
    kind: str
    block_index: int
    after_online_event_id: int
    stale: bool

    def to_dict(self):
        return {"source_event_id": self.record.event_id, "key": self.record.key,
                "value": self.record.value, "version": self.record.version,
                "kind": self.kind, "block_index": self.block_index,
                "after_online_event_id": self.after_online_event_id, "stale": self.stale}


@dataclass(frozen=True)
class ReplayPlan:
    events: tuple[ScheduledEvent, ...]
    metadata: dict

    @property
    def records(self):
        return tuple(event.record for event in self.events)

    def to_dict(self):
        """Return a detached JSON-serializable schedule/budget/causality artifact."""
        return json.loads(json.dumps(self.metadata, allow_nan=False))


def _buffer_state(buffer):
    return [{"key": record.key, "version": record.version, "event_id": record.event_id}
            for record in buffer.values()]


def schedule_episode(episode, condition, block_size=4, buffer_size=8, *, schedule_seed=None):
    """Construct a causal schedule with paired continuous/periodic replay IDs.

    At each block start, snapshot the latest-per-key FIFO buffer and draw exactly
    the block's length with replacement, or draw none when the snapshot is empty.
    Only real online records update buffer membership, values and recency. A
    revision refreshes its key's recency. Replay reads never refresh the buffer.

    Continuous replay follows each corresponding online event; periodic replay
    delivers the same ordered selection after the entire block. Repeat-current
    uses the same eligible insertion slots but repeats each current online event.
    The initial empty-snapshot block receives no extras in any condition.
    """
    if not isinstance(episode, Episode):
        raise TypeError("episode must be an Episode")
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    _integer("block_size", block_size, 1)
    _integer("buffer_size", buffer_size, 1)
    schedule_seed = episode.seed if schedule_seed is None else schedule_seed
    _integer("schedule_seed", schedule_seed)
    replay_rng = _rng(schedule_seed, f"replay:{episode.seed}")
    buffer, latest, plan, blocks, reference_ids = OrderedDict(), {}, [], [], []

    def execute(record, kind, block_index, after_online):
        if kind == "online":
            latest[record.key] = record
            buffer.pop(record.key, None)
            buffer[record.key] = record
            if len(buffer) > buffer_size:
                buffer.popitem(last=False)
        elif record.event_id > after_online or record.key not in latest:
            raise AssertionError("Replay cannot read a future online event")
        stale = kind == "replay" and latest[record.key].version > record.version
        plan.append(ScheduledEvent(record, kind, block_index, after_online, stale))

    for block_index, start in enumerate(range(0, len(episode.events), block_size)):
        block = episode.events[start:start + block_size]
        snapshot = tuple(buffer.values())
        selected = tuple(replay_rng.choice(snapshot) for _ in block) if snapshot else ()
        if any(record.event_id >= start for record in selected):
            raise AssertionError("Block-start replay snapshot contains a future event")
        reference_ids.extend(record.event_id for record in selected)
        block_meta = {"block_index": block_index, "online_event_ids": [r.event_id for r in block],
                      "buffer_at_start": _buffer_state(buffer),
                      "reference_replay_event_ids": [r.event_id for r in selected],
                      "selected_replay_event_ids": [], "extra_event_ids": []}
        for index, record in enumerate(block):
            execute(record, "online", block_index, record.event_id)
            if selected and condition in ("continuous", "repeat_current"):
                extra = selected[index] if condition == "continuous" else record
                kind = "replay" if condition == "continuous" else "repeat_current"
                execute(extra, kind, block_index, record.event_id)
                block_meta["extra_event_ids"].append(extra.event_id)
                if kind == "replay":
                    block_meta["selected_replay_event_ids"].append(extra.event_id)
        if condition == "periodic":
            for extra in selected:
                execute(extra, "replay", block_index, block[-1].event_id)
                block_meta["selected_replay_event_ids"].append(extra.event_id)
                block_meta["extra_event_ids"].append(extra.event_id)
        block_meta["buffer_at_end"] = _buffer_state(buffer)
        blocks.append(block_meta)
    online_count = len(episode.events)
    replay_count = sum(event.kind == "replay" for event in plan)
    repeat_count = sum(event.kind == "repeat_current" for event in plan)
    extra_count = replay_count + repeat_count
    record_tokens = RECORD_BYTES * len(plan)
    execution_plan = [{"execution_index": index, **event.to_dict()} for index, event in enumerate(plan)]
    metadata = {
        "format_version": FORMAT_VERSION, "episode_seed": episode.seed,
        "schedule_seed": schedule_seed, "split": episode.split, "condition": condition,
        "key_pool_scheme": KEY_POOL_SCHEME, "key_pool_sha256": KEY_POOL_HASHES[episode.split],
        "key_pool_hashes": dict(KEY_POOL_HASHES),
        "block_size": block_size, "buffer_size": buffer_size, "record_bytes": RECORD_BYTES,
        "query_prefix_bytes": QUERY_PREFIX_BYTES, "online_record_count": online_count,
        "replay_record_count": replay_count, "repeat_current_record_count": repeat_count,
        "extra_record_count": extra_count, "total_record_count": len(plan),
        "online_token_count": RECORD_BYTES * online_count,
        "replay_token_count": RECORD_BYTES * replay_count,
        "repeat_current_token_count": RECORD_BYTES * repeat_count,
        "extra_token_count": RECORD_BYTES * extra_count, "record_token_count": record_tokens,
        "query_prefix_token_count": QUERY_PREFIX_BYTES,
        "input_token_count": record_tokens + QUERY_PREFIX_BYTES,
        "target_token_count": 1, "training_sequence_token_count": record_tokens + QUERY_PREFIX_BYTES + 1,
        "selected_replay_event_ids": [event.record.event_id for event in plan if event.kind == "replay"],
        "extra_event_ids": [event.record.event_id for event in plan if event.kind != "online"],
        "reference_replay_event_ids": reference_ids,
        "stale_replay_count": sum(event.stale for event in plan),
        "final_buffer": _buffer_state(buffer), "blocks": blocks, "execution_plan": execution_plan,
    }
    return ReplayPlan(tuple(plan), metadata)


def encode_example(episode, query, condition, block_size=4, buffer_size=8, *, schedule_seed=None):
    """Return ``(model_prefix_bytes, next_answer_byte_int, JSON_metadata)``.

    The answer is excluded from model inputs. Append ``bytes([answer])`` only
    when constructing a shifted training target. No query is inserted before
    the final query, so evaluating a different key never feeds earlier answers.
    """
    if not isinstance(episode, Episode):
        raise TypeError("episode must be an Episode")
    if not isinstance(query, FactQuery) or query not in episode.queries:
        raise ValueError("query must be one of this episode's validated queries")
    plan = schedule_episode(episode, condition, block_size, buffer_size, schedule_seed=schedule_seed)
    query_prefix = render_query(query)
    prefix = b"".join(record.encoded for record in plan.records) + query_prefix
    query_start = RECORD_BYTES * len(plan.events)
    if len(prefix) != plan.metadata["input_token_count"] or query_start % RECORD_BYTES:
        raise AssertionError("Query must start in its own attention segment")
    last_key_index = max(index for index, event in enumerate(plan.events) if event.record.key == query.key)
    last_key_record = plan.events[last_key_index].record
    metadata = plan.to_dict()
    metadata.update({
        "query": query.to_dict(), "query_prefix": query_prefix.decode("ascii"),
        "answer_byte": query.answer_byte, "query_start_token": query_start,
        "query_attention_segment": len(plan.events),
        "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        "online_events_since_latest_target": len(episode.events) - 1 - query.latest_event_id,
        "records_since_last_target_key_record": len(plan.events) - 1 - last_key_index,
        "last_target_key_record_source_event_id": last_key_record.event_id,
        "last_target_key_record_version": last_key_record.version,
        "target_in_final_buffer": any(row["key"] == query.key for row in metadata["final_buffer"]),
    })
    return prefix, query.answer_byte, metadata
