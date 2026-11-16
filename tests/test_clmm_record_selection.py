"""Falsifiable byte, counterfactual, split-isolation and pairing checks."""

from collections import Counter
import copy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from clmm.pretrain.fact_streams import Episode
from clmm.pretrain.key_binding import make_codec as old_make_codec
from clmm.pretrain.record_selection import (
    ALIGNED_QUERY_TEMPLATES, CONDITIONS, ENCODINGS, INPUT_BYTES,
    SELECT_QUERY_TEMPLATES, TASKS, VARIANTS, collect_variant_signatures,
    evaluation_worlds, make_codec, make_samples, make_world, render_decode_query, render_query_prefix,
    render_samples, training_arrays, training_worlds, variant_records,
    variant_signatures, visible_oracle,
)


def _alias(codec, key):
    return codec.aliases[(codec.train_keys + codec.dev_keys).index(key)]


def _independent_signatures(ep):
    # Do not call the module's variant constructor or canonical hash helper.
    (ka, va), (kb, vb) = [(r.key, r.value) for r in ep.events]
    histories = (((ka, va), (kb, vb)), ((kb, vb), (ka, va)),
                 ((ka, vb), (kb, va)), ((kb, va), (ka, vb)))
    return {hashlib.sha256(json.dumps(history, separators=(',', ':')).encode('ascii')).hexdigest()
            for history in histories}


def test_fixed_old_codec_is_reused_without_changes():
    codec = make_codec()
    assert codec == old_make_codec(91401)
    assert len(codec.train_keys) == 96 and len(codec.dev_keys) == 32
    assert set(codec.aliases) == set(range(128, 256))
    assert INPUT_BYTES == 95


@pytest.mark.parametrize('family', ['seen', 'heldout'])
@pytest.mark.parametrize('encoding', ENCODINGS)
@pytest.mark.parametrize('seed', range(4))
def test_tags_replace_only_padding_and_raw_key_fields(family, encoding, seed):
    codec = make_codec()
    ep = make_world(seed, split='dev', family=family)
    for variant in VARIANTS:
        records = variant_records(ep, variant)
        rows = render_samples(ep, codec, 'select', encoding, variant)
        assert rows[0]['prefix'][:64] == rows[1]['prefix'][:64]
        for slot, record in enumerate(records):
            raw = record.encoded
            assert len(raw) == 32 and raw[:3] == b'   '
            expected = (b'[0]' if slot == 0 else b'[1]') + raw[3:]
            start = expected.index(record.key.encode('ascii'))
            if encoding == 'atomic':
                expected = expected[:start] + bytes([_alias(codec, record.key)]) + b'  ' + expected[start + 3:]
            actual = rows[0]['prefix'][32 * slot:32 * (slot + 1)]
            assert actual == expected
            assert actual[-3:] == record.value.encode('ascii') + b'.\n'
        for row in rows:
            assert len(row['prefix']) == 95
            assert visible_oracle(row['prefix'], codec, 'select', encoding) == row['target']
            if encoding == 'atomic':
                assert sum(byte >= 128 for byte in row['prefix']) == 3
                assert bytes.fromhex(row['prefix'].hex()) == row['prefix']


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('encoding', ENCODINGS)
@pytest.mark.parametrize('family', ['seen', 'heldout'])
def test_query_depends_on_identity_and_wording_but_not_label(task, encoding, family):
    codec = make_codec()
    ep = make_world(771, split='dev', family=family)
    for index in range(2):
        for query in ep.queries:
            query = replace(query, template_index=index)
            alternate = replace(query, target='a' if query.target != 'a' else 'p')
            options = {'slot': 0} if task == 'decode' else {}
            raw = render_query_prefix(query, codec, task, encoding, **options)
            assert raw == render_query_prefix(alternate, codec, task, encoding, **options)
            assert len(raw) == 31
            if task == 'decode':
                assert raw[-1:] == b'0' and query.key.encode('ascii') not in raw
                assert not any(byte >= 128 for byte in raw)
                assert render_query_prefix(query, codec, task, encoding, slot=1) == raw[:-1] + b'1'
            elif task == 'select_aligned':
                expected = (query.key.encode('ascii') if encoding == 'digits'
                            else b'  ' + bytes([_alias(codec, query.key)]))
                assert raw[-3:] == expected
            elif encoding == 'atomic':
                assert bytes([_alias(codec, query.key)]) + b'  ' in raw
    # The fixed source wordings themselves do not contain an interpolated label.
    for templates in (SELECT_QUERY_TEMPLATES, ALIGNED_QUERY_TEMPLATES):
        for wording in templates[family]:
            assert '{key}' in wording and '{target}' not in wording and '{slot}' not in wording


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('encoding', ENCODINGS)
@pytest.mark.parametrize('seed', range(4))
def test_counterfactuals_keep_query_identity_and_change_only_correct_labels(task, encoding, seed):
    codec = make_codec()
    ep = make_world(seed + 9100)
    views = {variant: render_samples(ep, codec, task, encoding, variant) for variant in VARIANTS}
    for rank, query in enumerate(ep.queries):
        original = views['original'][rank]
        for variant in VARIANTS:
            row = views[variant][rank]
            assert row['rank'] == rank and row['key'] == query.key
            reverse = variant in ('order', 'order_value')
            swap = variant in ('value', 'order_value')
            assert row['slot'] == (1 - rank if reverse else rank)
            if task == 'decode':
                assert row['prefix'][64:-1] == original['prefix'][64:-1]
                assert row['prefix'][-1] == ord('0') + row['slot']
            else:
                assert row['prefix'][64:] == original['prefix'][64:]
            expected = ord(ep.events[1 - rank if swap else rank].value) if task in ('value', 'decode') else ord('0') + row['slot']
            assert row['target'] == expected
            assert visible_oracle(row['prefix'], codec, task, encoding) == expected
        changed = 'value' if task in ('value', 'decode') else 'order'
        invariant = 'order' if task in ('value', 'decode') else 'value'
        assert views[changed][rank]['target'] != original['target']
        assert views[invariant][rank]['target'] == original['target']


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('encoding', ENCODINGS)
@pytest.mark.parametrize('variant', VARIANTS)
def test_masks_preserve_slots_and_make_absent_evidence_explicit(task, encoding, variant):
    codec = make_codec()
    ep = make_world(888)
    normal = render_samples(ep, codec, task, encoding, variant)
    for condition in CONDITIONS[task]:
        rows = render_samples(ep, codec, task, encoding, variant, condition)
        for rank, row in enumerate(rows):
            base = normal[rank]
            prefix = row['prefix']
            assert len(prefix) == 95
            assert row['target'] == base['target'] and row['slot'] == base['slot']
            if condition in ('normal', 'reset_memory', 'retrieval_zero'):
                assert prefix == base['prefix'] and row['real_records'] == 2
            elif condition == 'no_facts':
                assert prefix == b' ' * 64 + base['prefix'][64:]
                assert row['real_records'] == 0
            elif condition in ('oracle', 'wrong_record'):
                keep = base['slot'] if condition == 'oracle' else 1 - base['slot']
                assert row['real_records'] == 1
                assert prefix[32 * keep:32 * (keep + 1)] == base['prefix'][32 * keep:32 * (keep + 1)]
                assert prefix[32 * (1 - keep):32 * (2 - keep)] == b' ' * 32
                assert prefix[64:] == base['prefix'][64:]
            elif condition == 'key_erased':
                assert prefix[:64] == base['prefix'][:64]
                field = (ep.queries[rank].key.encode('ascii') if encoding == 'digits' else
                         (b'  ' + bytes([_alias(codec, ep.queries[rank].key)]) if task == 'select_aligned'
                          else bytes([_alias(codec, ep.queries[rank].key)]) + b'  '))
                assert prefix[64:] == base['prefix'][64:].replace(field, b'   ')
                assert row['real_records'] == 2
            elif condition == 'index_erased':
                assert prefix == base['prefix'][:-1] + b' '
                assert row['real_records'] == 2
            expected = None if condition in ('no_facts', 'wrong_record', 'key_erased', 'index_erased') else row['target']
            assert visible_oracle(prefix, codec, task, encoding) == expected
        if condition in ('key_erased', 'index_erased'):
            # Same observed input with different labels defeats query-rank or
            # hidden-label leakage.  No row identity is present in prefix.
            assert rows[0]['prefix'] == rows[1]['prefix']
            assert rows[0]['target'] != rows[1]['target']


@pytest.mark.parametrize('encoding', ENCODINGS)
def test_visible_oracle_follows_observations_and_detects_corruption(encoding):
    codec = make_codec()
    ep = make_world(1122)
    row = render_samples(ep, codec, 'value', encoding)[0]
    prefix = row['prefix']
    # Alter a valid observed value without touching any Episode or row label.
    changed_value = ord('p') if prefix[29] != ord('p') else ord('a')
    changed = prefix[:29] + bytes([changed_value]) + prefix[30:]
    assert row['target'] != changed_value
    assert visible_oracle(changed, codec, 'value', encoding) == changed_value
    selector = render_samples(ep, codec, 'select', encoding)[0]['prefix']
    changed = selector[:29] + bytes([changed_value]) + selector[30:]
    assert visible_oracle(changed, codec, 'select', encoding) == ord('0')
    corruptions = [b'[1]' + prefix[3:], prefix[:29] + b'z' + prefix[30:],
                   prefix[:32] + b'[1]' + prefix[3:32] + prefix[64:],
                   prefix[:64] + b'?' * 31, prefix[:-1], bytearray(prefix)]
    if encoding == 'atomic':
        # A text roundtrip expands raw aliases and is explicitly invalid.
        corruptions.append(prefix.decode('latin1').encode('utf8'))
    for raw in corruptions:
        with pytest.raises(ValueError):
            visible_oracle(raw, codec, 'value', encoding)


def test_all_ordered_variant_histories_are_canonical_and_excluded_jointly():
    codec = make_codec()
    ep = make_world(3030)
    assert variant_signatures(ep) == _independent_signatures(ep)
    assert len(variant_signatures(ep)) == 4
    worlds = training_worlds(3, 2, codec=codec)
    expected = set().union(*(_independent_signatures(Episode.from_dict(raw))
                             for step in worlds for raw in step['episodes']))
    assert collect_variant_signatures(worlds) == expected
    gate = evaluation_worlds(expected, codec, seed=91613, n=5)
    assert not collect_variant_signatures(gate) & expected
    final = evaluation_worlds(expected | collect_variant_signatures(gate), codec, seed=91623, n=5)
    assert not collect_variant_signatures(final) & (expected | collect_variant_signatures(gate))
    seen = set()
    for group in ('train_seen', 'dev_seen'):
        for raw in final[group]:
            ids = _independent_signatures(Episode.from_dict(raw))
            assert len(ids) == 4 and not ids & seen
            seen.update(ids)
    for a, b in zip(final['dev_seen'], final['dev_heldout']):
        a, b = Episode.from_dict(a), Episode.from_dict(b)
        assert _independent_signatures(a) == _independent_signatures(b)
        assert a.template_family == 'seen' and b.template_family == 'heldout'
        assert [q.key for q in a.queries] == [q.key for q in b.queries]
    assert evaluation_worlds(expected, codec, seed=91613, n=5) == gate


@pytest.mark.parametrize('variant', VARIANTS)
def test_exclusion_of_any_one_variant_rejects_the_entire_base_world(variant):
    codec = make_codec()
    initial = evaluation_worlds(set(), codec, seed=2277, n=1)
    first = Episode.from_dict(initial['train_seen'][0])
    records = variant_records(first, variant)
    forbidden = hashlib.sha256(json.dumps([(r.key, r.value) for r in records],
                                          separators=(',', ':')).encode('ascii')).hexdigest()
    fresh = evaluation_worlds({forbidden}, codec, seed=2277, n=1)
    assert not variant_signatures(first) & collect_variant_signatures(fresh)
    assert fresh['train_seen'][0] != initial['train_seen'][0]


def test_training_arrays_are_paired_reconstructible_and_all_k2():
    codec = make_codec()
    worlds = training_worlds(7, 3, codec=codec)
    assert training_worlds(7, 3, codec=codec) == worlds
    fixed = training_worlds(7, 3, codec=codec, shuffle_queries=False)
    assert [w['episodes'] for w in worlds] == [w['episodes'] for w in fixed]
    assert all(order == [0, 1] for step in fixed for order in step['query_orders'])
    assert {tuple(order) for step in worlds for order in step['query_orders']} == {(0, 1), (1, 0)}
    arrays = training_arrays(worlds, codec)
    assert set(arrays['lengths']) == {95}
    assert arrays['slots'].dtype == arrays['ranks'].dtype == np.int8
    assert np.array_equal(arrays['slots'], arrays['ranks'])
    assert np.array_equal(arrays['targets_select'], arrays['targets_select_aligned'])
    for i, step in enumerate(worlds):
        assert step['stage'] == 'k2'
        for j, (raw, order) in enumerate(zip(step['episodes'], step['query_orders'])):
            ep = Episode.from_dict(raw)
            assert ep.facts == 2 and len(ep.events) == 2
            assert list(arrays['ranks'][i, 2 * j:2 * j + 2]) == order
            for task in TASKS:
                assert arrays[f'targets_{task}'].dtype == np.uint8
                for encoding in ENCODINGS:
                    arr = arrays[f'{task}_{encoding}']
                    assert arr.shape == (7, 6, 95) and arr.dtype == np.uint8
                    assert arr.size == 7 * 6 * 95
                    rows = render_samples(ep, codec, task, encoding)
                    for offset, rank in enumerate(order):
                        actual = bytes(arr[i, 2 * j + offset])
                        assert actual == rows[rank]['prefix']
                        assert visible_oracle(actual, codec, task, encoding) == arrays[f'targets_{task}'][i, 2 * j + offset]
    corrupt = copy.deepcopy(worlds)
    corrupt[0]['query_orders'][0] = [0, 0]
    with pytest.raises(ValueError):
        training_arrays(corrupt, codec)


def test_frozen_row_counts_metadata_and_raw_byte_roundtrip():
    codec = make_codec()
    data = evaluation_worlds(set(), codec, n=2)
    rows = make_samples(data, codec)
    counts = Counter((r['task'], r['encoding']) for r in rows)
    for encoding in ENCODINGS:
        assert counts['select', encoding] == 240
        assert counts['select_aligned', encoding] == 240
        assert counts['decode', encoding] == 336
        assert counts['value', encoding] == 336
        # Scale to the predeclared N=256 fork budget in each case.
        assert counts['select', encoding] * 128 == 30720
        assert counts['value', encoding] * 128 == 43008
    for row in rows:
        prefix = bytes.fromhex(row['prefix_hex'])
        assert len(prefix) == row['input_bytes'] == 95
        assert row['rank'] in (0, 1) and row['slot'] in (0, 1)
        assert row['extra_records'] == row['stale_extras'] == 0
        expected = None if row['condition'] in ('key_erased', 'index_erased', 'no_facts', 'wrong_record') else row['target']
        assert visible_oracle(prefix, codec, row['task'], row['encoding']) == expected
    assert json.loads(json.dumps(rows)) == rows
    gate = make_samples(data, codec, tasks=('select', 'value'), encodings=('digits',),
                        conditions={'select': ('normal',), 'value': ('normal', 'oracle')})
    assert len(gate) == 144
    assert set((r['task'], r['condition']) for r in gate) == {
        ('select', 'normal'), ('value', 'normal'), ('value', 'oracle')}


@pytest.mark.parametrize('kwargs', [dict(steps=0), dict(contexts=0), dict(steps=True),
                                   dict(shuffle_queries='yes')])
def test_invalid_training_configuration(kwargs):
    with pytest.raises(ValueError):
        training_worlds(**kwargs)


def test_unsupported_worlds_conditions_and_subsets_are_rejected():
    codec = make_codec()
    ep = make_world(811)
    with pytest.raises(ValueError):
        render_samples(make_world(811, facts=1), codec, 'select', 'digits')
    for task, encoding, variant, condition in [('bad', 'digits', 'original', 'normal'),
                                              ('select', 'utf8', 'original', 'normal'),
                                              ('select', 'digits', 'bad', 'normal'),
                                              ('select', 'digits', 'original', 'oracle')]:
        with pytest.raises(ValueError):
            render_samples(ep, codec, task, encoding, variant, condition)
    data = {'train_seen': [ep.to_dict()]}
    for kwargs in (dict(tasks=()), dict(tasks=('select', 'select')),
                   dict(encodings=('utf8',)), dict(conditions={'select': ('oracle',)}),
                   dict(conditions={'select': ('normal', 'normal')}), dict(conditions={})):
        options = {'tasks': ('select',), **kwargs}
        with pytest.raises(ValueError):
            make_samples(data, codec, **options)


@pytest.mark.parametrize('slot', [None, -1, 2, True, '0'])
def test_decode_requires_valid_explicit_position_cue(slot):
    codec = make_codec()
    query = make_world(444).queries[0]
    with pytest.raises(ValueError):
        render_query_prefix(query, codec, 'decode', 'digits', slot=slot)
    with pytest.raises(ValueError):
        render_decode_query(slot, index_erased=True)


@pytest.mark.parametrize('family', ['seen', 'heldout'])
@pytest.mark.parametrize('index', [0, 1])
def test_decode_composition_candidates_use_only_supplied_index(family, index):
    zero = render_decode_query(0, family, index)
    one = render_decode_query(1, family, index)
    erased = render_decode_query(0, family, index, index_erased=True)
    assert len(zero) == len(one) == len(erased) == 31
    assert zero[:-1] == one[:-1] == erased[:-1]
    assert zero[-1:] == b'0' and one[-1:] == b'1' and erased[-1:] == b' '
    assert erased == render_decode_query(1, family, index, index_erased=True)


@pytest.mark.parametrize('encoding', ENCODINGS)
@pytest.mark.parametrize('variant', VARIANTS)
def test_decoder_oracle_obeys_visible_index_even_when_it_disagrees_with_query_identity(encoding, variant):
    codec = make_codec()
    ep = make_world(76331)
    row = render_samples(ep, codec, 'decode', encoding, variant)[0]
    prefix = row['prefix']
    other = 1 - row['slot']
    # Keep the original row's key/target metadata untouched.  Only the visible
    # query index changes, so the correct decoder answer is the other record.
    switched = prefix[:-1] + bytes([ord('0') + other])
    observed_other_value = prefix[32 * other + 29]
    assert observed_other_value != row['target']
    assert visible_oracle(switched, codec, 'decode', encoding) == observed_other_value
    assert visible_oracle(prefix[:-1] + b' ', codec, 'decode', encoding) is None
    with pytest.raises(ValueError):
        visible_oracle(prefix[:-1] + b'2', codec, 'decode', encoding)
