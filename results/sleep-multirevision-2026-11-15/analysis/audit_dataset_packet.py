"""Audit the frozen dataset against independently recovered saved exclusions."""
import argparse
import hashlib
import json
from pathlib import Path

from independent_tools import audit_dataset, saved_signatures, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'dataset_audit.json')
    args = parser.parse_args()
    relative = {'training90502': 'trained/old_training_exclusions.npz',
                'training90802': 'trained/training_data.npz',
                'evaluation90723': 'provenance/prior90723/episodes.json',
                'evaluation90823': 'frozen/episodes.json',
                'evaluation90923': 'trained/episodes.json',
                'evaluation91023': 'stress/episodes.json'}
    excluded, sources = set(), {}
    for role, name in relative.items():
        path = args.prior_root/name
        signatures = saved_signatures(path)
        assert signatures
        excluded.update(signatures)
        sources[role] = dict(path=str(path.resolve()), sha256=sha(path), unique_histories=len(signatures))
    datasets = json.loads(args.episodes.read_text(encoding='utf-8'))
    n = len(datasets['gate_k16_r3'])
    result = audit_dataset(datasets, n, excluded)
    result.update(status='passed', episodes=str(args.episodes.resolve()), episodes_sha256=sha(args.episodes),
                  exclusion_sources=sources,
                  sorted_exclusion_set_sha256=hashlib.sha256(('\n'.join(sorted(excluded))+'\n').encode()).hexdigest())
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='passed', cells=result['cells'], histories_per_cell=n,
                         literal_histories=result['literal_histories'], exclusions=len(excluded), out=str(args.out))))


if __name__ == '__main__':
    main()
