"""Broader prior-JSON audit; preserve launch packets and disclose coverage gaps."""
import json
from pathlib import Path
import subprocess

from independent_tools import saved_signatures, sha
from independent_generation import audit_generation


def main():
    out=Path(__file__).resolve().parent
    root=out.parent
    output_root=root.parent
    scan=subprocess.run(['rg','-l','-U',r'"keys"\s*:\s*\[',str(output_root),
                         '-g','*.json','-g','!plans.json','-g','!excluded_signatures.json','-g','!models.json'],
                        capture_output=True,text=True,check=False)
    assert scan.returncode in (0,1),scan.stderr
    excluded=set(json.loads((root/'packet/excluded_signatures.json').read_text(encoding='utf-8')))
    prior_files, prior_union, beyond=set(),set(),[]
    numeric_manifest=[]
    for raw in scan.stdout.splitlines():
        path=Path(raw).resolve()
        if root in path.parents:
            if not any('engineer' in part.lower() for part in path.relative_to(root).parts):
                continue
        ids=saved_signatures(path)
        if not ids:
            continue
        prior_files.add(str(path))
        prior_union.update(ids)
        missing=ids-excluded
        numeric_manifest.append(dict(path=str(path),sha256=sha(path),histories=len(ids),
                                     omitted_from_launch=len(missing)))
        if missing:
            beyond.append(dict(path=str(path),sha256=sha(path),histories=len(ids),
                               omitted_signatures=sorted(missing)))
    missing=prior_union-excluded
    assert len(missing)==1 and len(beyond)==2
    assert missing=={'5671c63e7ecea6474409f7eb758467e43cdb1892c7ffbdb4b9bf296fa7293bca'}
    main_data=json.loads((root/'packet/episodes.json').read_text(encoding='utf-8'))
    main_ids=saved_signatures(root/'packet/episodes.json')
    engineering_data=json.loads((root/'engineering/packet/episodes.json').read_text(encoding='utf-8'))
    engineering_ids=saved_signatures(root/'engineering/packet/episodes.json')
    assert not main_ids.intersection(prior_union)
    assert not engineering_ids.intersection(missing)
    main_regeneration=audit_generation(main_data,91223,128,excluded|missing)
    engineering_excluded=set(json.loads((root/'engineering/packet/excluded_signatures.json').read_text(encoding='utf-8')))
    assert not main_ids.intersection(excluded|missing)
    assert not engineering_ids.intersection(engineering_excluded|missing)
    engineering_regeneration=audit_generation(engineering_data,91203,16,engineering_excluded|missing)
    episode_manifest=json.loads((out/'packet_audit.json').read_text(encoding='utf-8'))['independently_enumerated_prior_episode_files']
    non_numeric=[r for r in episode_manifest if r['histories']==0]
    assert len(episode_manifest)==274 and len(non_numeric)==1 and len(prior_files)==275
    assert {r['path'] for r in episode_manifest if r['histories']>0} | {r['path'] for r in beyond} == prior_files
    superset=out/'audited_signature_superset.json'
    superset.write_text(json.dumps(sorted(excluded|missing),indent=2)+'\n',encoding='utf-8')
    result=dict(status='passed_with_documented_launch_coverage_omission',
                scope='Broader prior JSON embedded-key/value history search; no inference or engineering efficacy inspection',
                original_packets_modified=False,launch_exclusion_json_sha256=sha(root/'packet/excluded_signatures.json'),
                launch_exclusion_count=len(excluded),broader_prior_numeric_files=len(prior_files),
                prior_episode_json_files=274,prior_episode_json_files_with_numeric_histories=273,
                non_numeric_episode_json_files=non_numeric,
                file_count_reconciliation='274 prior episodes.json paths include1 byte/semantic episode file with0 typed numeric histories;273 numeric episode files+2 numeric verification files=275 numeric files.',
                prior_numeric_history_manifest=numeric_manifest,
                omitted_prior_unique_histories=len(missing),omitted_sources=beyond,
                supplemental_signatures=sorted(missing),extended_exclusion_count=len(excluded|missing),
                engineering_own_extended_exclusion_count=len(engineering_excluded|missing),
                reusable_superset_path=str(superset),reusable_superset_sha256=sha(superset),
                reusable_superset_scope='Original launch union plus the one omitted integration history; a future study must also add this study main histories and any later consulted inputs.',
                main_overlap_with_extended_prior_union=0,engineering_overlap_with_supplement=0,
                main_overlap_with_own_extended_launch_exclusions=0,
                engineering_overlap_with_own_extended_launch_exclusions=0,
                exact_main_regeneration_with_extended_exclusions=main_regeneration,
                exact_engineering_regeneration_with_supplement=engineering_regeneration,
                conclusion='One seed90323 integration history embedded in verification.json was absent from the launch union. It has26 arrivals, whereas all new cells have2/3/4/22/72/136/200 arrivals. Extended exclusions reproduce every accepted new history exactly, so no data, score or selection change is needed.',
                interpretation='Provenance coverage omission, not observed train/test history overlap; disclose rather than retroactively rewrite the launch packet.')
    (out/'supplemental_exclusion_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','omitted_prior_unique_histories','main_overlap_with_extended_prior_union','extended_exclusion_count')}))


if __name__=='__main__':
    main()
