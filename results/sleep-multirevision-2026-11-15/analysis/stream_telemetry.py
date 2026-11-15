"""Summarize model-independent, independently audited literal event telemetry."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from independent_tools import ARMS, PRESSURE_CELLS, read_csv, sha, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source = args.evaluation/'replay_audit.csv'
    rows = read_csv(source)
    grouped = defaultdict(list)
    for row in rows:
        if row['cell'] in PRESSURE_CELLS:
            grouped[(row['cell'], row['arm'])].append(row)
    assert set(grouped) == {(cell, arm) for cell in PRESSURE_CELLS for arm in ARMS}
    result = []
    for (cell, arm), subset in sorted(grouped.items()):
        assert len(subset) == 128 and {int(r['episode']) for r in subset} == set(range(128))
        totals = {key: sum(int(row[key]) for row in subset)
                  for key in ('events', 'extra_writes', 'stale_extra_writes', 'final_obsolete_query_extras')}
        phase, tail = int(cell.split('_p')[1].split('_')[0]), int(cell.rsplit('_d', 1)[1])
        result.append(dict(cell=cell, phase=phase, retention_tail=tail, arm=arm, histories=128,
                           **totals, stale_fraction_of_extras=totals['stale_extra_writes']/totals['extra_writes']
                           if totals['extra_writes'] else None))
    write_csv(args.analysis/'replay_telemetry.csv', result)
    meta = dict(source=str(source.resolve()), source_sha256=sha(source),
                data_sha256=sha(args.evaluation/'episodes.json'),
                audit='The full raw audit checks every event against an independent PCG/FIFO reconstruction.',
                distinction='stale uses world truth at application; final_obsolete refers to earlier versions of final queried keys',
                scope='Model-independent counts over the same128 paired streams, repeated under every checkpoint; do not multiply by12 for independent sample size.')
    (args.analysis/'replay_telemetry.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(cells=6, arms=4, histories=128, out=str(args.analysis/'replay_telemetry.csv'))))


if __name__ == '__main__':
    main()
