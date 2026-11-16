import copy
import json
import numpy as np
import pytest
from clmm.pretrain.fact_streams import Episode, KEY_POOLS, render_query
from clmm.pretrain.key_binding import (KeyCodec, make_codec, make_world, examples, training_worlds,
    training_arrays, evaluation_worlds, make_samples, CONDITIONS)
from clmm.pretrain.sequence_freshness_data import collect_histories, signature, swap_assignments


def test_codec_identity_and_resume():
    codec = make_codec()
    assert len(codec.train_keys) == 96 and len(codec.dev_keys) == 32
    assert len(set(codec.aliases)) == 128 and set(codec.aliases) == set(range(128, 256))
    assert not set(codec.train_keys) & set(codec.dev_keys)
    assert KeyCodec.from_dict(json.loads(json.dumps(codec.to_dict()))) == codec
    assert make_codec() == codec and make_codec(91499) != codec


@pytest.mark.parametrize('family', ['seen', 'heldout'])
@pytest.mark.parametrize('seed', range(12))
def test_raw_span_and_query_label_exclusion(family, seed):
    codec = make_codec()
    ep = make_world(seed, split='dev', family=family)
    assert ep.events[0].value != ep.events[1].value
    for record, query in zip(ep.events, ep.queries):
        for raw in (record.encoded, render_query(query)):
            transformed = codec.transform(raw, record.key, 'atomic')
            index = raw.index(record.key.encode())
            alias = codec.aliases[(codec.train_keys + codec.dev_keys).index(record.key)]
            assert len(transformed) == len(raw)
            assert transformed[:index] == raw[:index] and transformed[index + 3:] == raw[index + 3:]
            assert transformed[index:index + 3] == bytes([alias]) + b'  '
            assert codec.transform(raw, record.key, 'digits') == raw
        # Targets are not interpolated in the raw query or the conversion.
        altered = copy.copy(query)
        object.__setattr__(altered, 'target', 'p' if query.target != 'p' else 'a')
        assert render_query(altered) == render_query(query)
    a, b = examples(ep, codec, 'atomic')
    assert a[0][:64] == b[0][:64] and len(a[0]) == len(b[0]) == 95
    assert a[1] != b[1]


@pytest.mark.parametrize('raw,key,encoding', [(b'001001', '001', 'atomic'), (b'zero', '001', 'atomic'),
    ('001', '001', 'atomic'), (b'001', '001', 'utf8'), (b'xyz', 'xyz', 'atomic')])
def test_invalid_visible_span(raw, key, encoding):
    with pytest.raises(ValueError):
        make_codec().transform(raw, key, encoding)


def test_codec_corruptions_rejected():
    for name in ('schema_version', 'atomic_field', 'aliases', 'train_keys'):
        data = make_codec().to_dict()
        data[name] = False if name == 'schema_version' else ('bad' if name == 'atomic_field' else [data[name][0]] * len(data[name]))
        with pytest.raises(ValueError):
            KeyCodec.from_dict(data)


def test_training_arrays_pairing_and_actual_lengths():
    codec = make_codec()
    worlds = training_worlds(20, 2, codec=codec)
    arrays = training_arrays(worlds, codec)
    assert [w['stage'] for w in worlds].count('k1') == 3
    assert arrays['digits'].shape == arrays['atomic'].shape == (20, 4, 95)
    assert set(arrays['lengths']) == {63, 95}
    assert arrays['digits'].dtype == arrays['atomic'].dtype == np.uint8
    for i, w in enumerate(worlds):
        n = int(arrays['lengths'][i])
        assert not np.any(arrays['digits'][i, :, n:]) and not np.any(arrays['atomic'][i, :, n:])
        for j, row in enumerate(w['episodes']):
            ep = Episode.from_dict(row)
            for encoding in ('digits', 'atomic'):
                for rank, (x, target) in enumerate(examples(ep, codec, encoding)):
                    assert bytes(arrays[encoding][i, j * 2 + rank, :n]) == x
                    assert arrays['targets'][i, j * 2 + rank] == target
    assert training_worlds(20, 2, codec=codec) == worlds


def test_evaluation_exclusions_pairing_and_frozen_samples():
    codec = make_codec()
    initial = evaluation_worlds(set(), codec, n=4)
    forbidden = collect_histories(initial)
    data = evaluation_worlds(forbidden, codec, n=4)
    assert not collect_histories(data) & forbidden
    for split in ('train', 'dev'):
        for a, b in zip(data[f'acquire_{split}_seen'], data[f'acquire_{split}_swapped']):
            assert swap_assignments(Episode.from_dict(a)).to_dict() == b
    for a, b in zip(data['acquire_dev_seen'], data['acquire_dev_heldout']):
        assert signature(a['events']) == signature(b['events'])
    samples = make_samples(data, codec)
    assert len(samples) == 2 * 4 * 40
    assert set(r['condition'] for r in samples) == set(CONDITIONS)
    for r in samples:
        assert len(bytes.fromhex(r['prefix_hex'])) == r['input_bytes']
        assert r['input_bytes'] == (31 if r['group'].startswith('query_only') else 95)
    assert collect_histories(training_worlds(5, 2, codec=codec))
