"""Authenticate the twelve prescribed terminal checkpoint files; no unpickling/inference."""
import argparse
import json
from pathlib import Path

from independent_tools import PARENTS, ROUTES, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trained', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'pretrained_inventory.json')
    args = parser.parse_args()
    rows = []
    for parent in PARENTS:
        for route in ROUTES:
            folder = args.trained/'runs'/f'{parent}-{route}-local_delta'
            case = json.loads((folder/'case.json').read_text(encoding='utf-8'))
            assert case['name'] == folder.name and case['parent_seed'] == parent
            assert case['revision_rule'] == case['config']['revision_rule'] == 'local_delta'
            assert case['config']['inactive_decay'] == 'protected'
            assert case['training']['step'] == 1200 and case['training']['train_seed'] == 90802
            assert case['source_commit'] == 'c9f5c153f786f1b82ac383d01c1d18c6c595946b'
            assert case['status'] == 'succeeded' and case['registered_parameters'] == 81921
            if route == 'content':
                assert case['policy'] == 'content' and case['random_seed'] is None
            else:
                assert case['policy'] == 'random' and case['random_seed'] == int(route.split('-')[1])
            table = case['route_table']
            assert len(table) == 64 and all(type(k) is int and 0 <= k < 4 for k in table)
            assert case['route_cardinalities'] == [table.count(k) for k in range(4)]
            checkpoint = folder/'checkpoint.pt'
            actual_sha = sha(checkpoint)
            assert actual_sha == case['checkpoint_sha256']
            rows.append(dict(name=case['name'], parent_seed=parent, route=route,
                             checkpoint=str(checkpoint.resolve()), checkpoint_sha256=actual_sha,
                             case_sha256=sha(folder/'case.json'), training=case['training'],
                             model_slow_sha256=case['slow_sha256'], route_table=table,
                             route_table_sha256=case['config']['routing_table_sha256'],
                             registered_parameters=81921, retained_fast_bytes_per_stream=65536))
    result = dict(status='passed', cases=rows, checkpoints=len(rows),
                  scope='file hashes and saved case records; checkpoint payload loading/model inference not performed',
                  selection='all prescribed2parents x(content+5fixedrandom) terminal1200step local_delta checkpoints')
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='passed', checkpoints=len(rows), out=str(args.out))))


if __name__ == '__main__':
    main()
