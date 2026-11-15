"""Exercise independent raw-artifact audit on already produced CPU smokes."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from raw_eval_tools import audit_eval
p=argparse.ArgumentParser()
p.add_argument('--smoke-parent',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
results=[]
for folder in sorted(a.smoke_parent.glob('smoke-*cpu')):
    meta=json.loads((folder/'metadata.json').read_text())
    assert meta['status']=='succeeded'
    episodes=json.loads((folder/'episodes.json').read_text())
    for case in meta['cases']:
        for stage in ('evaluation','initial-evaluation'):
            path=folder/'runs'/case['name']/stage
            if path.exists():
                result=audit_eval(path,episodes,case['route_table'],case['revision_rule'],int(meta['args']['eval_batch_size']))
                results.append(dict(smoke=folder.name,case=case['name'],stage=stage,**result['audit']))
a.out.write_text(json.dumps(dict(status='passed',checks=results),indent=2)+'\n')
print(json.dumps(dict(status='passed',evaluation_batches=len(results),predictions=sum(r['predictions'] for r in results))))
