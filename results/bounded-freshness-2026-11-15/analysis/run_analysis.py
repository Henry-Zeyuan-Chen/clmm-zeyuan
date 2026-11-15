"""Progressive completed-case arithmetic audit; final stats require all12."""
import argparse
import json
from pathlib import Path

import numpy as np

from independent_raw_audit import audit_evaluation
from independent_statistics import summarize
from independent_tools import read_csv, sha, write_csv

SOURCE_COMMIT = '9ab3901860548d2d87a9afc61e8ce58f950a03ce'
EXPECTED = dict(score_vectors=36096, causal_writes=1099392, extra_writes=460800,
                resource_batches=384, literal_plans=6144, matched_candidate_cells=768)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--model', action='append')
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    root, analysis = args.root.resolve(), Path(__file__).resolve().parent
    packet = root/'packet'
    packet_meta = json.loads((packet/'metadata.json').read_text(encoding='utf-8'))
    assert packet_meta['development_seed'] == 91223 and packet_meta['episodes'] == 128
    assert packet_meta['source_commit'] == SOURCE_COMMIT
    assert sha(packet/'episodes.json') == packet_meta['data_sha256']
    models = json.loads((packet/'models.json').read_text(encoding='utf-8'))
    data = json.loads((packet/'episodes.json').read_text(encoding='utf-8'))
    case_out = analysis/'cases'
    case_out.mkdir(exist_ok=True)
    wanted = {r['name'] for r in models} if args.model is None else set(args.model)
    assert wanted <= {r['name'] for r in models}
    statuses = []
    for record in models:
        name = record['name']
        folder, meta_path = root/'runs'/name, root/'runs'/name/'metadata.json'
        if not meta_path.exists():
            statuses.append(dict(name=name, status='missing'))
            continue
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        statuses.append(dict(name=name, status=meta['status'], failure=meta.get('failure'),
                             metadata_sha256=sha(meta_path), elapsed_seconds=meta.get('elapsed_seconds')))
        if name not in wanted or meta['status'] != 'succeeded':
            continue
        for key in ('name', 'parent_seed', 'policy', 'random_seed', 'config', 'slow_sha256',
                    'checkpoint_sha256', 'route_table', 'original_training'):
            assert meta[key] == record[key], (name, key)
        assert meta['source_commit'] == packet_meta['source_commit']
        assert meta['offline_training'] is False and meta['additional_optimizer_steps'] == 0
        assert meta['data_sha256'] == packet_meta['data_sha256']
        assert meta['environment']['device'] == 'cpu' and meta['environment']['threads'] == 1
        assert meta['environment']['dtype'] == 'float32' and meta['environment']['deterministic'] is True
        assert meta['evaluation']['pure_reads'] and meta['evaluation']['slow_parameters_and_rng_unchanged']
        assert meta['observer']['matched_frozen_candidate_sequences'] == 768
        assert meta['observer']['literal_CP_extra_sequences_match']
        assert meta['observer']['bounded_content_can_differ_by_application_time']
        assert meta['observer']['observer_annotations_only'] and meta['observer']['query_metadata_not_passed_to_writes']
        assert sha(folder/'evaluation/scores.npy') == meta['evaluation']['scores_sha256']
        saved = case_out/f'{name}.json'
        if saved.exists():
            audit = json.loads(saved.read_text(encoding='utf-8'))
            assert audit['metadata_sha256'] == sha(meta_path)
            for file_name, digest in audit['audit']['files'].items():
                assert sha(folder/'evaluation'/file_name) == digest
            assert sha(case_out/f'{name}.npz') == audit['vectors_sha256']
            continue
        result = audit_evaluation(folder/'evaluation', data, record['route_table'])
        assert all(result['audit'][key] == expected for key, expected in EXPECTED.items())
        assert meta['evaluation']['queries'] == result['audit']['score_vectors']
        vector_keys, all_correct_keys = sorted(result['vectors']), sorted(result['all_correct'])
        arrays = {f'v{i}': result['vectors'][key] for i, key in enumerate(vector_keys)}
        arrays.update({f'a{i}': result['all_correct'][key] for i, key in enumerate(all_correct_keys)})
        array_path = case_out/f'{name}.npz'
        np.savez_compressed(array_path, **arrays)
        write_csv(case_out/f'{name}-diagnostics.csv', result['diagnostics'])
        row = dict(name=name, parent_seed=record['parent_seed'],
                   route='content' if record['policy'] == 'content' else f"random-{record['random_seed']}",
                   metadata_sha256=sha(meta_path), vectors_sha256=sha(array_path),
                   vector_keys=vector_keys, all_correct_keys=all_correct_keys, audit=result['audit'])
        saved.write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(dict(audited=name, **{k: result['audit'][k] for k in EXPECTED})), flush=True)
    (analysis/'run_status_inventory.json').write_text(json.dumps(statuses, indent=2)+'\n', encoding='utf-8')
    if not args.final:
        return
    assert len(models) == 12 and all((case_out/f"{r['name']}.json").exists() for r in models), 'All12 audited cases required before final stats'
    assert all(r['status'] == 'succeeded' for r in statuses), 'All12 final runs must succeed; preserve failures separately'
    vectors, all_correct, audits, diagnostics = {}, {}, [], []
    for record in models:
        name = record['name']
        saved = json.loads((case_out/f'{name}.json').read_text(encoding='utf-8'))
        parent, route = saved['parent_seed'], saved['route']
        with np.load(case_out/f'{name}.npz', allow_pickle=False) as arrays:
            for i, key in enumerate(saved['vector_keys']):
                vectors[(parent, route, *key)] = arrays[f'v{i}'].copy()
            for i, key in enumerate(saved['all_correct_keys']):
                all_correct[(parent, route, *key)] = arrays[f'a{i}'].copy()
        audits.append(saved['audit'])
        diagnostics.extend(dict(parent_seed=parent, route=route,
                                **{key: value for key, value in row.items()
                                   if key not in ('penultimate', 'query_module', 'final_applied_value_for_key')})
                           for row in read_csv(case_out/f'{name}-diagnostics.csv'))
    result = summarize(vectors, all_correct)
    total_keys = (*EXPECTED, 'stale_extra_writes', 'refreshed_records', 'changed_values', 'fallback_writes')
    totals = {key: sum(a[key] for a in audits) for key in total_keys}
    assert all(totals[key] == value*12 for key, value in EXPECTED.items())
    for name in ('gates', 'aggregate', 'contrasts', 'per_condition'):
        write_csv(analysis/f'{name}.csv', result.pop(name))
    write_csv(analysis/'diagnostics.csv', diagnostics)
    supplement=json.loads((analysis/'supplemental_exclusion_audit.json').read_text(encoding='utf-8'))
    assert supplement['main_overlap_with_own_extended_launch_exclusions']==0
    result.update(status='passed_with_documented_launch_coverage_omission', saved_output_audit_status='passed',
                  models=12, audited_totals=totals, packet_audit='packet_audit.json',
                  source_commit=packet_meta['source_commit'], data_sha256=packet_meta['data_sha256'],
                  additional_optimizer_steps=0, all_models_included=True,
                  exclusion_coverage_audit='supplemental_exclusion_audit.json',
                  launch_exclusion_collector_scope='episodes.json plus two explicit training arrays',
                  launch_omitted_prior_unique_histories=1, extended_prior_overlap=0,
                  extended_exclusions_leave_every_main_and_engineering_history_unchanged=True,
                  verification_scope='full saved-output arithmetic plus independent data/source reconstruction; no numerical model inference by reviewer',
                  query_fifo_feature_scope='block-start candidate availability; bounded actual targets may refresh within block',
                  tables=['gates.csv', 'aggregate.csv', 'contrasts.csv', 'per_condition.csv', 'diagnostics.csv'])
    (analysis/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(final_statistics='saved', models=12, totals=totals, primary=result['primary'])), flush=True)


if __name__ == '__main__':
    main()
