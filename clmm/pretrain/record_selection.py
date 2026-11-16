"""Paired record selection and value decoding data for the unchanged byte LM.

Each input is two 32-byte records and a 31-byte query.  A visible ``[0]`` or
``[1]`` replaces three leading padding spaces in each record.  Tags describe
the current position, including after a reversal.  Key queries retain the base
world's key identities.  Target labels, query ranks, and external audit metadata
are never model input; the explicit ``decode`` position cue is intentional input.

``decode`` supplies a visible current-position index rather than a key, so it
measures value decoding with an explicit record-selection cue.  Its target
value is never part of the query.  ``select_aligned`` moves the queried identity
to the last query token.  It is a separate alignment-supervision diagnostic, not an equivalent
wording variant of ``select``.  Atomic record fields and ordinary queries retain
the existing alias-byte-plus-two-spaces encoding.  Only aligned atomic queries
use two spaces followed by the alias, keeping all inputs exactly 95 raw bytes.
"""

from dataclasses import replace
from functools import lru_cache
import re

import numpy as np

from .fact_streams import (
    Episode, FactQuery, QUERY_PREFIX_BYTES, QUERY_TEMPLATES, RECORD_BYTES,
    RECORD_TEMPLATES, render_query,
)
from .key_binding import ENCODINGS, KeyCodec, make_codec, make_world
from .sequence_freshness_data import retemplate, signature


TASKS = ('select', 'select_aligned', 'decode', 'value')
VARIANTS = ('original', 'order', 'value', 'order_value')
SELECT_CONDITIONS = ('normal', 'reset_memory', 'retrieval_zero', 'key_erased', 'no_facts')
VALUE_CONDITIONS = ('normal', 'reset_memory', 'retrieval_zero', 'oracle',
                    'wrong_record', 'no_facts', 'key_erased')
DECODE_CONDITIONS = ('normal', 'reset_memory', 'retrieval_zero', 'oracle',
                     'wrong_record', 'no_facts', 'index_erased')
CONDITIONS = {'select': SELECT_CONDITIONS, 'select_aligned': SELECT_CONDITIONS,
              'decode': DECODE_CONDITIONS, 'value': VALUE_CONDITIONS}
INPUT_BYTES = 2 * RECORD_BYTES + QUERY_PREFIX_BYTES
SELECT_QUERY_TEMPLATES = {
    'seen': ('record for key {key}: ', 'which record has {key} = '),
    'heldout': ('locate record for {key}: ', 'record holding key {key} = '),
}
ALIGNED_QUERY_TEMPLATES = {
    'seen': ('matching record key {key}', 'choose record for {key}'),
    'heldout': ('find its record: {key}', 'match record to {key}'),
}
DECODE_QUERY_TEMPLATES = {
    'seen': ('read value from record {slot}', 'decode record {slot}'),
    'heldout': ('retrieve value at slot {slot}', 'which value in record {slot}'),
}
_TASK_TEMPLATES = {'select': SELECT_QUERY_TEMPLATES,
                   'select_aligned': ALIGNED_QUERY_TEMPLATES,
                   'decode': DECODE_QUERY_TEMPLATES,
                   'value': QUERY_TEMPLATES}


def _positive_integer(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _world(value):
    ep = Episode.from_dict(value) if isinstance(value, dict) else value
    if (not isinstance(ep, Episode) or ep.facts != 2 or ep.delay != 0
            or ep.revision_fraction != 0 or ep.include_recent
            or len(ep.events) != 2 or len(ep.queries) != 2
            or len({r.key for r in ep.events}) != 2
            or len({r.value for r in ep.events}) != 2):
        raise ValueError('Record selection requires two distinct keys and values, with no revisions')
    return ep


def variant_records(world, variant='original'):
    """Return the observed record view without revalidating a reversed Episode.

    Record identities and their wording move together.  Value swaps exchange
    assignments at fixed keys, and order reversals then move the entire records.
    """
    ep = _world(world)
    if variant not in VARIANTS:
        raise ValueError('Unknown record-selection variant')
    records = ep.events
    if variant in ('value', 'order_value'):
        a, b = records
        records = (replace(a, value=b.value), replace(b, value=a.value))
    if variant in ('order', 'order_value'):
        records = records[::-1]
    return records


def variant_signatures(world):
    """Canonical ordered key/value signatures for all four full histories."""
    return {signature(variant_records(world, variant)) for variant in VARIANTS}


def collect_variant_signatures(value):
    """Collect all variant histories from worlds, step lists, or group mappings.

    Unlike a generic history collector, this explicitly closes exclusions under
    both order reversal and value reassignment before evaluation generation.
    Template clones intentionally share signatures.
    """
    if isinstance(value, Episode):
        return variant_signatures(value)
    if isinstance(value, dict):
        if {'format_version', 'events', 'queries', 'facts'} <= set(value):
            return variant_signatures(value)
        return set().union(*(collect_variant_signatures(v) for v in value.values()))
    if isinstance(value, (tuple, list)):
        return set().union(*(collect_variant_signatures(v) for v in value))
    return set()


def render_decode_query(slot, family='seen', template_index=0, *, index_erased=False):
    """Render an explicit position cue without any key or value metadata.

    This helper also constructs composed-inference candidates at positions zero
    and one.  ``index_erased=True`` yields the common no-index candidate while
    still requiring the caller to pass a valid slot.
    """
    if type(slot) is not int or slot not in (0, 1):
        raise ValueError('decode requires an explicit current record slot 0 or 1')
    if (family not in DECODE_QUERY_TEMPLATES or type(template_index) is not int
            or template_index not in (0, 1) or type(index_erased) is not bool):
        raise ValueError('Unsupported decode wording or index-erasure flag')
    raw = DECODE_QUERY_TEMPLATES[family][template_index].format(slot=slot).encode('ascii')
    raw = raw.rjust(QUERY_PREFIX_BYTES, b' ')
    if len(raw) != QUERY_PREFIX_BYTES or raw[-1] != ord('0') + slot:
        raise ValueError('Decode query must end in its explicit record-index byte')
    return raw[:-1] + b' ' if index_erased else raw


def render_query_prefix(query, codec, task, encoding, *, key_erased=False, slot=None,
                        index_erased=False):
    """Render a label-independent query from identity/index, wording and task."""
    if not isinstance(query, FactQuery) or task not in TASKS or encoding not in ENCODINGS:
        raise ValueError('Expected a FactQuery, supported task and supported encoding')
    if type(key_erased) is not bool or type(index_erased) is not bool:
        raise ValueError('Erasure flags must be Boolean')
    if task == 'decode':
        if key_erased:
            raise ValueError('decode has an index field, not a key field')
        return render_decode_query(slot, query.template_family, query.template_index,
                                   index_erased=index_erased)
    if index_erased:
        raise ValueError('Only decode has an explicit query-index field')
    if task == 'value':
        raw = render_query(query)
    else:
        raw = _TASK_TEMPLATES[task][query.template_family][query.template_index].format(
            key=query.key).encode('ascii').rjust(QUERY_PREFIX_BYTES, b' ')
    if len(raw) != QUERY_PREFIX_BYTES or raw.count(query.key.encode('ascii')) != 1:
        raise ValueError('Query must contain exactly one three-byte key field')
    start = raw.index(query.key.encode('ascii'))
    # This validates codec membership even when the visible key will be erased.
    encoded = codec.transform(raw, query.key, encoding)
    if task == 'select_aligned' and encoding == 'atomic':
        encoded = encoded[:start] + b'  ' + encoded[start:start + 1] + encoded[start + 3:]
    if key_erased:
        encoded = encoded[:start] + b'   ' + encoded[start + 3:]
    return encoded


def _tagged_record(record, codec, encoding, slot):
    raw = record.encoded
    if len(raw) != RECORD_BYTES or raw[:3] != b'   ':
        raise ValueError('Position tags may replace only three leading record-padding spaces')
    tagged = bytes((ord('['), ord('0') + slot, ord(']'))) + raw[3:]
    return codec.transform(tagged, record.key, encoding)


def render_samples(world, codec, task, encoding, variant='original', condition='normal'):
    """Return two raw query forks in fixed base-query identity order.

    ``prefix`` is the only model input.  ``target``, ``slot``, ``rank``, key and
    history signatures are external labels/audit metadata.  The decode prefix
    intentionally also contains a rendered position cue.  Oracle/wrong-record
    masks preserve the retained record's original slot and replace the other
    32-byte slot with spaces.  Model ablations do not alter input bytes.
    """
    ep = _world(world)
    if task not in TASKS or encoding not in ENCODINGS or condition not in CONDITIONS[task]:
        raise ValueError('Unsupported task, encoding or condition')
    records = variant_records(ep, variant)
    tagged = tuple(_tagged_record(record, codec, encoding, slot)
                   for slot, record in enumerate(records))
    history = signature(records)
    rows = []
    for rank, query in enumerate(ep.queries):
        slot = next(i for i, record in enumerate(records) if record.key == query.key)
        keep = (slot,) if condition == 'oracle' else (
            (1 - slot,) if condition == 'wrong_record' else (
                () if condition == 'no_facts' else (0, 1)))
        context = b''.join(raw if i in keep else b' ' * RECORD_BYTES
                           for i, raw in enumerate(tagged))
        prefix = context + render_query_prefix(query, codec, task, encoding,
                                               key_erased=condition == 'key_erased', slot=slot,
                                               index_erased=condition == 'index_erased')
        target = ord(records[slot].value) if task in ('value', 'decode') else ord('0') + slot
        if len(prefix) != INPUT_BYTES:
            raise ValueError('Every record-selection input must be exactly 95 bytes')
        rows.append(dict(prefix=prefix, target=target, key=query.key, slot=slot, rank=rank,
                         history_signature=history,
                         observed_history_signature=signature([records[i] for i in keep]),
                         real_records=len(keep)))
    return rows


def training_worlds(steps=20000, contexts=8, seed=91602, codec=None, *, shuffle_queries=True):
    """Generate only K2 worlds, shared across every task and key encoding.

    Fork order is randomized once per world, outside the rendered input.  The
    explicit ``query_orders`` metadata allows exact tensor-row reconstruction.
    """
    _positive_integer('steps', steps)
    _positive_integer('contexts', contexts)
    if type(shuffle_queries) is not bool:
        raise ValueError('shuffle_queries must be Boolean')
    codec = make_codec() if codec is None else codec
    worlds = []
    for i in range(steps):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, i])))
        seeds = rng.integers(8_000_000_000, 9_000_000_000, size=contexts)
        eps = [make_world(int(s), facts=2, codec=codec) for s in seeds]
        orders = [[1, 0] if shuffle_queries and int(rng.integers(2)) else [0, 1]
                  for _ in eps]
        worlds.append(dict(step=i + 1, stage='k2', episodes=[ep.to_dict() for ep in eps],
                           query_orders=orders))
    return worlds


def training_arrays(worlds, codec):
    """Build paired uint8 inputs and separate targets/rank/slot metadata.

    Main defaults give 20,000 updates x 16 forks = 320,000 95-byte inputs
    (30,400,000 presented input bytes) in each task/encoding case.
    """
    if not isinstance(worlds, (list, tuple)) or not worlds:
        raise ValueError('At least one training step is required')
    contexts = len(worlds[0]['episodes'])
    _positive_integer('contexts', contexts)
    shape = (len(worlds), 2 * contexts)
    arrays = {f'{task}_{encoding}': np.empty((*shape, INPUT_BYTES), dtype=np.uint8)
              for task in TASKS for encoding in ENCODINGS}
    arrays.update({f'targets_{task}': np.empty(shape, dtype=np.uint8) for task in TASKS})
    arrays['slots'] = np.empty(shape, dtype=np.int8)
    arrays['ranks'] = np.empty(shape, dtype=np.int8)
    arrays['lengths'] = np.full(len(worlds), INPUT_BYTES, dtype=np.int16)
    for i, step in enumerate(worlds):
        if (step.get('stage') != 'k2' or len(step['episodes']) != contexts
                or len(step['query_orders']) != contexts):
            raise ValueError('Every step must have the same positive K2 context count')
        for j, (raw, order) in enumerate(zip(step['episodes'], step['query_orders'])):
            if type(order) is not list or order not in ([0, 1], [1, 0]):
                raise ValueError('Each query order must be an explicit permutation of [0, 1]')
            ep = _world(raw)
            sl = slice(2 * j, 2 * j + 2)
            arrays['ranks'][i, sl] = order
            arrays['slots'][i, sl] = order  # original-view position equals base query identity
            for task in TASKS:
                for encoding in ENCODINGS:
                    rows = render_samples(ep, codec, task, encoding)
                    arrays[f'{task}_{encoding}'][i, sl] = np.stack([
                        np.frombuffer(rows[rank]['prefix'], dtype=np.uint8) for rank in order])
                    targets = [rows[rank]['target'] for rank in order]
                    if encoding == 'digits':
                        arrays[f'targets_{task}'][i, sl] = targets
                    elif not np.array_equal(arrays[f'targets_{task}'][i, sl], targets):
                        raise ValueError('Matched encodings must have identical labels')
    return arrays


def evaluation_worlds(excluded, codec, seed=91623, n=256):
    """Reject all four full variant histories jointly against global exclusions.

    Accepted train/dev worlds also exclude one another.  Heldout wording is an
    intentional semantic clone of dev_seen; it does not consume a fresh world.
    Generate a gate first, then add ``collect_variant_signatures(gate)`` to the
    exclusions before generating the final evaluation.
    """
    _positive_integer('n', n)
    rng = np.random.Generator(np.random.PCG64(seed))
    seen = set(excluded)
    groups = {}
    for split in ('train', 'dev'):
        accepted = []
        for _ in range(n):
            for _attempt in range(100_000):
                ep = make_world(int(rng.integers(9_000_000_000, 10_000_000_000)),
                                facts=2, split=split, codec=codec)
                ids = variant_signatures(ep)
                if not ids & seen:
                    seen.update(ids)
                    accepted.append(ep)
                    break
            else:
                raise RuntimeError('Fresh record-selection evaluation rejection exhausted')
        groups[f'{split}_seen'] = accepted
    groups['dev_heldout'] = [retemplate(ep, 'heldout') for ep in groups['dev_seen']]
    return {name: [ep.to_dict() for ep in eps] for name, eps in groups.items()}


eval_worlds = evaluation_worlds


def make_samples(data, codec, *, tasks=TASKS, encodings=ENCODINGS, conditions=None):
    """Freeze JSON-safe rows; labels and audit fields remain outside input bytes.

    ``conditions`` may map each requested task to its condition subset, e.g.
    selector normal and value normal+oracle for a cheap engineering gate.
    """
    tasks, encodings = tuple(tasks), tuple(encodings)
    if (not tasks or not encodings or len(set(tasks)) != len(tasks)
            or len(set(encodings)) != len(encodings) or not set(tasks) <= set(TASKS)
            or not set(encodings) <= set(ENCODINGS)):
        raise ValueError('Unique supported task and encoding subsets are required')
    source = CONDITIONS if conditions is None else conditions
    if not isinstance(source, dict) or not set(tasks) <= set(source):
        raise ValueError('Conditions must map every requested task to a condition subset')
    selected = {task: tuple(source[task]) for task in tasks}
    for task in tasks:
        values = selected[task]
        if not values or len(set(values)) != len(values) or not set(values) <= set(CONDITIONS[task]):
            raise ValueError('Unsupported or duplicate task conditions')
    rows = []
    for task in tasks:
        for encoding in encodings:
            for group, eps in data.items():
                for variant in VARIANTS:
                    for condition in selected[task]:
                        for i, raw in enumerate(eps):
                            for sample in render_samples(raw, codec, task, encoding, variant, condition):
                                prefix = sample.pop('prefix')
                                rows.append(dict(task=task, encoding=encoding, group=group,
                                                 variant=variant, arm='online', condition=condition,
                                                 episode=i, **sample, old=-1,
                                                 prefix_hex=prefix.hex(), input_bytes=len(prefix),
                                                 extra_records=0, stale_extras=0))
    return rows


@lru_cache(maxsize=None)
def _query_grammars(task, encoding):
    # Independent visible-byte matcher: no Episode, query target, rank or slot.
    if task == 'decode':
        return tuple(re.compile(rb' *' + re.escape(text.encode('ascii')).replace(
            re.escape(b'{slot}'), rb'(?P<slot>[01]| )'))
            for templates in DECODE_QUERY_TEMPLATES.values() for text in templates)
    if encoding == 'digits':
        key = rb'(?P<key>[0-9]{3}| {3})'
    elif task == 'select_aligned':
        key = rb'(?P<key>  [\x80-\xff]| {3})'
    else:
        key = rb'(?P<key>[\x80-\xff]  | {3})'
    return tuple(re.compile(rb' *' + re.escape(text.encode('ascii')).replace(
        re.escape(b'{key}'), key))
        for templates in _TASK_TEMPLATES[task].values() for text in templates)


@lru_cache(maxsize=None)
def _record_grammars(encoding):
    key = rb'(?P<key>[0-9]{3})' if encoding == 'digits' else rb'(?P<key>[\x80-\xff]  )'
    return tuple(re.compile(rb'\[(?P<tag>[01])\] *' + re.escape(text.encode('ascii')).replace(
        re.escape(b'{key}'), key).replace(re.escape(b'{value}'), rb'(?P<value>[a-p])'))
        for templates in RECORD_TEMPLATES.values() for text in templates)


def _visible_identity(field, codec, encoding, *, aligned=False):
    if field == b'   ':
        return None
    keys = codec.train_keys + codec.dev_keys
    if encoding == 'digits':
        key = field.decode('ascii')
        if key not in keys:
            raise ValueError('Visible query/record key is outside the fixed codec')
        return key
    alias = field[-1] if aligned else field[0]
    try:
        return keys[codec.aliases.index(alias)]
    except ValueError as error:
        raise ValueError('Visible alias is outside the fixed codec') from error


def visible_oracle(prefix, codec, task, encoding):
    """Derive a label from visible bytes alone, or None if evidence is absent.

    This data-validation oracle is never fed to the neural model.  It rejects
    malformed records/tags/queries and duplicate keys, and returns None for a
    missing query identity or a queried record removed by an input ablation.
    """
    if type(prefix) is not bytes or len(prefix) != INPUT_BYTES or task not in TASKS or encoding not in ENCODINGS:
        raise ValueError('Expected an exact 95-byte input and supported task/encoding')
    query = prefix[2 * RECORD_BYTES:]
    match = next((m for grammar in _query_grammars(task, encoding)
                  if (m := grammar.fullmatch(query)) is not None), None)
    if match is None:
        raise ValueError('Malformed visible record-selection query')
    if task == 'decode':
        wanted = None if match['slot'] == b' ' else int(match['slot'])
    else:
        wanted = _visible_identity(match['key'], codec, encoding, aligned=task == 'select_aligned')
    observed = {}
    observed_slots = {}
    for slot in range(2):
        raw = prefix[slot * RECORD_BYTES:(slot + 1) * RECORD_BYTES]
        if raw == b' ' * RECORD_BYTES:
            continue
        match = next((m for grammar in _record_grammars(encoding)
                      if (m := grammar.fullmatch(raw)) is not None), None)
        if match is None or int(match['tag']) != slot:
            raise ValueError('Malformed visible record or incorrect position tag')
        key = _visible_identity(match['key'], codec, encoding)
        if key in observed:
            raise ValueError('Visible records must have distinct keys')
        observed[key] = (match['tag'][0], match['value'][0])
        observed_slots[slot] = match['value'][0]
    if task == 'decode':
        return observed_slots.get(wanted)
    if wanted is None or wanted not in observed:
        return None
    tag, value = observed[wanted]
    return value if task == 'value' else tag
