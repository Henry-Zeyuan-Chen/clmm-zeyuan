"""Matched digit/atomic-key acquisition data for the unchanged 256-byte LM.

Atomic keys use one byte followed by two spaces in the original three-byte
key field. This is an encoding-bundle diagnostic, not a tokenizer change or
a replay adapter. No controller or model receives query targets as input.
"""
from dataclasses import dataclass
import math
import numpy as np
from .fact_streams import Episode, FactRecord, FactQuery, KEY_POOLS, VALUES, render_query
from .sequence_freshness_data import signature, retemplate, swap_assignments

ENCODINGS = ('digits', 'atomic')
CONDITIONS = ('normal', 'reset_memory', 'retrieval_zero')


@dataclass(frozen=True)
class KeyCodec:
    train_keys: tuple[str, ...]
    dev_keys: tuple[str, ...]
    aliases: tuple[int, ...]

    def __post_init__(self):
        for name in ('train_keys', 'dev_keys', 'aliases'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        keys = self.train_keys + self.dev_keys
        if (len(self.train_keys) < 2 or len(self.dev_keys) < 2 or len(keys) > 128
                or len(set(keys)) != len(keys) or len(self.aliases) != len(keys)
                or len(set(self.aliases)) != len(keys)
                or any(type(a) is not int or not 128 <= a <= 255 for a in self.aliases)
                or not set(self.train_keys) <= set(KEY_POOLS['train'])
                or not set(self.dev_keys) <= set(KEY_POOLS['dev'])):
            raise ValueError('Invalid disjoint key pools or byte aliases')

    def to_dict(self):
        return dict(schema_version=1, train_keys=list(self.train_keys),
                    dev_keys=list(self.dev_keys), aliases=list(self.aliases),
                    atomic_field='one alias byte followed by two ASCII spaces')

    @classmethod
    def from_dict(cls, value):
        if (type(value) is not dict or set(value) != {'schema_version', 'train_keys', 'dev_keys', 'aliases', 'atomic_field'}
                or type(value['schema_version']) is not int or value['schema_version'] != 1
                or value['atomic_field'] != 'one alias byte followed by two ASCII spaces'):
            raise ValueError('Unsupported codec state')
        return cls(value['train_keys'], value['dev_keys'], value['aliases'])

    def transform(self, raw, key, encoding):
        if type(raw) is not bytes or encoding not in ENCODINGS:
            raise ValueError('Expected bytes and a supported encoding')
        keys = self.train_keys + self.dev_keys
        if key not in keys:
            raise ValueError('Key outside the fixed codec')
        token = key.encode('ascii')
        if raw.count(token) != 1:
            raise ValueError('Exactly one visible three-digit key field is required')
        if encoding == 'digits':
            return raw
        field = bytes([self.aliases[keys.index(key)]]) + b'  '
        return raw.replace(token, field, 1)


def make_codec(seed=91401):
    aliases = np.random.Generator(np.random.PCG64(seed)).permutation(np.arange(128, 256))
    return KeyCodec(KEY_POOLS['train'][:96], KEY_POOLS['dev'][:32], tuple(map(int, aliases)))


def make_world(seed, facts=2, split='train', family='seen', codec=None):
    codec = make_codec() if codec is None else codec
    if type(facts) is not int or facts not in (1, 2) or split not in ('train', 'dev'):
        raise ValueError('Only K1/K2 train/dev acquisition is supported')
    if family not in ('seen', 'heldout') or split == 'train' and family != 'seen':
        raise ValueError('Training uses seen wording only')
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    pool = codec.train_keys if split == 'train' else codec.dev_keys
    keys = rng.choice(pool, facts, replace=False).tolist()
    values = rng.choice(list(VALUES), facts, replace=False).tolist()
    records = tuple(FactRecord(i, key, value, 1, 'fact', family, int(rng.integers(2)))
                    for i, (key, value) in enumerate(zip(keys, values)))
    query_template = int(rng.integers(2))
    queries = tuple(FactQuery(r.key, r.value, None, 'stable', i, i, family, query_template)
                    for i, r in enumerate(records))
    return Episode(int(seed), facts, 0, split, family, 0., False, records, queries)


def examples(ep, codec, encoding, query_only=False):
    context = b'' if query_only else b''.join(codec.transform(r.encoded, r.key, encoding) for r in ep.events)
    queries = ep.queries if len(ep.queries) == 2 else (ep.queries[0], ep.queries[0])
    return [(context + codec.transform(render_query(q), q.key, encoding), q.answer_byte) for q in queries]


def training_worlds(steps=2000, contexts=8, seed=91402, codec=None):
    if type(steps) is not int or steps < 1 or type(contexts) is not int or contexts < 1:
        raise ValueError('Positive integer steps/contexts required')
    codec = make_codec() if codec is None else codec
    worlds = []
    for i in range(steps):
        stage = 'k1' if i < math.ceil(.128 * steps) else 'k2'
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, i])))
        eps = [make_world(int(s), 1 if stage == 'k1' else 2, codec=codec)
               for s in rng.integers(8_000_000_000, 9_000_000_000, size=contexts)]
        worlds.append(dict(step=i + 1, stage=stage, episodes=[ep.to_dict() for ep in eps]))
    return worlds


def training_arrays(worlds, codec):
    batch = len(worlds[0]['episodes']) * 2
    arrays = {encoding: np.zeros((len(worlds), batch, 95), dtype=np.uint8) for encoding in ENCODINGS}
    targets = np.empty((len(worlds), batch), dtype=np.uint8)
    lengths = np.empty(len(worlds), dtype=np.int16)
    for i, world in enumerate(worlds):
        for encoding in ENCODINGS:
            samples = [sample for row in world['episodes'] for sample in examples(Episode.from_dict(row), codec, encoding)]
            n = len(samples[0][0])
            assert all(len(x) == n for x, _ in samples)
            arrays[encoding][i, :, :n] = np.asarray([list(x) for x, _ in samples], dtype=np.uint8)
            y = np.asarray([y for _, y in samples], dtype=np.uint8)
            if encoding == 'digits':
                targets[i], lengths[i] = y, n
            else:
                assert np.array_equal(y, targets[i]) and lengths[i] == n
    return {**arrays, 'targets': targets, 'lengths': lengths}


def evaluation_worlds(excluded, codec, seed=91423, n=256):
    if type(n) is not int or n < 1:
        raise ValueError('Positive integer evaluation count required')
    rng = np.random.Generator(np.random.PCG64(seed))
    seen = set(excluded)
    groups = {}
    for split in ('train', 'dev'):
        accepted = []
        for _ in range(n):
            for attempt in range(100_000):
                ep = make_world(int(rng.integers(9_000_000_000, 10_000_000_000)), split=split, codec=codec)
                ids = {signature(ep.events), signature(swap_assignments(ep).events)}
                if not ids & seen:
                    seen.update(ids)
                    accepted.append(ep)
                    break
            else:
                raise RuntimeError('Fresh evaluation rejection exhausted')
        groups[f'acquire_{split}_seen'] = accepted
        groups[f'acquire_{split}_swapped'] = [swap_assignments(ep) for ep in accepted]
    groups['acquire_dev_heldout'] = [retemplate(ep, 'heldout') for ep in groups['acquire_dev_seen']]
    groups['acquire_dev_heldout_swapped'] = [retemplate(ep, 'heldout') for ep in groups['acquire_dev_swapped']]
    return {name: [ep.to_dict() for ep in eps] for name, eps in groups.items()}


def make_samples(data, codec):
    rows = []
    for encoding in ENCODINGS:
        for group, eps in data.items():
            for condition in CONDITIONS:
                for i, raw in enumerate(eps):
                    ep = Episode.from_dict(raw)
                    for rank, (x, target) in enumerate(examples(ep, codec, encoding)):
                        rows.append(dict(encoding=encoding, group=group, arm='online', condition=condition,
                                         episode=i, rank=rank, key=ep.queries[rank].key, target=target, old=-1,
                                         prefix_hex=x.hex(), input_bytes=len(x), real_records=2, extra_records=0, stale_extras=0))
        for source_group in ('acquire_train_seen', 'acquire_dev_seen'):
            for i, raw in enumerate(data[source_group]):
                ep = Episode.from_dict(raw)
                for rank, (x, target) in enumerate(examples(ep, codec, encoding, query_only=True)):
                    rows.append(dict(encoding=encoding, group=source_group.replace('acquire_', 'query_only_'), arm='online', condition='normal',
                                     episode=i, rank=rank, key=ep.queries[rank].key, target=target, old=-1,
                                     prefix_hex=x.hex(), input_bytes=len(x), real_records=0, extra_records=0, stale_extras=0))
    return rows
