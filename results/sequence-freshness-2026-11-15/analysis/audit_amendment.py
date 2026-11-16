"""Verify data-preserving pre-execution source amendment between frozen packets."""
import argparse,json
from pathlib import Path
import numpy as np
from audit_packet import read_json,sha


def compare(original,executed):
    checks={}
    for name in ['training_plan.json.gz','episodes.json','plans.json','samples.json','parents.json']:
        assert read_json(original/name)==read_json(executed/name),name
        checks[name]=dict(semantic_content_identical=True,original_sha256=sha(original/name),executed_sha256=sha(executed/name))
    with np.load(original/'training.npz') as left,np.load(executed/'training.npz') as right:
        assert set(left.files)==set(right.files)
        for name in left.files:assert np.array_equal(left[name],right[name]),name
        checks['training.npz']=dict(all_arrays_identical=True,keys=sorted(left.files),original_sha256=sha(original/'training.npz'),executed_sha256=sha(executed/'training.npz'))
    a=read_json(original/'metadata.json');b=read_json(executed/'metadata.json')
    for name in ['train_seed','development_seed','steps','contexts','episodes','revision_episodes','training_answers','training_input_bytes']:assert a[name]==b[name],name
    changed=[name for name in a['source_sha256'] if a['source_sha256'][name]!=b['source_sha256'][name]]
    assert set(a['source_sha256'])==set(b['source_sha256'])
    assert set(changed)=={'scripts/sequence_freshness_experiment.py','clmm/pretrain/sequence_freshness_data.py'},changed
    old=set(read_json(original/'excluded_signatures.json'));new=set(read_json(executed/'excluded_signatures.json'));assert old<=new
    return dict(status='passed',original_source=a['source_commit'],executed_source=b['source_commit'],
        source_files_changed=changed,original_exclusions=len(old),executed_exclusions=len(new),additional_exclusions=len(new-old),data=checks,
        scope='data-preserving pre-execution source amendment; no model execution')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--original',type=Path,required=True);parser.add_argument('--executed',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();result=compare(args.original,args.executed)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='data'}))

