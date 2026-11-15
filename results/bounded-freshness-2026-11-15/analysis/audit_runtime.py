"""Runtime continuation and preservation audit; reads no engineering scores."""
import argparse
import json
from pathlib import Path

from independent_tools import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--final',action='store_true')
    args=p.parse_args()
    root,out=args.root.resolve(),Path(__file__).resolve().parent
    pipeline_path=root/'pipeline.json'
    meta=json.loads(pipeline_path.read_text(encoding='utf-8'))
    expected=['90511-content-local_delta','90515-random-90701-local_delta']
    assert meta['representatives']==expected
    for key in ('preflight_sequential','preflight_concurrent'):
        assert len(meta[key])==2 and all(r['returncode']==0 for r in meta[key])
        assert [r['command'][r['command'].index('--model')+1] for r in meta[key]]==expected
        assert all(r['command'][r['command'].index('--max-seconds')+1]=='1500' for r in meta[key])
    projected=meta['concurrent_wall_seconds']*8*6*1.25
    assert abs(projected-meta['projected_main_seconds'])<1e-9
    assert meta['preflight_pass'] is (projected<=3600) and meta['preflight_pass']
    engineering=json.loads((root/'engineering/packet/metadata.json').read_text(encoding='utf-8'))
    main_packet=json.loads((root/'packet/metadata.json').read_text(encoding='utf-8'))
    assert engineering['development_seed']==91203 and engineering['episodes']==16
    assert main_packet['development_seed']==91223 and main_packet['episodes']==128
    inventory=[]
    for path in sorted(root.rglob('metadata.json')):
        if 'sources' in path.parts:
            continue
        record=json.loads(path.read_text(encoding='utf-8'))
        inventory.append(dict(path=str(path.resolve()),status=record.get('status'),mode=record.get('mode'),
                              name=record.get('name'),failure=record.get('failure'),sha256=sha(path)))
    if args.final:
        assert meta['status']=='succeeded' and len(meta['main'])==12
        assert len({r['name'] for r in meta['main']})==12 and all(r['returncode']==0 for r in meta['main'])
        assert meta['elapsed_seconds']<=4200
    result=dict(status='passed',pipeline_status=meta['status'],pipeline_sha256=sha(pipeline_path),
                engineering_efficacy_inspected=False,representatives=expected,
                sequential_seconds=[r['elapsed'] for r in meta['preflight_sequential']],
                concurrent_wall_seconds=meta['concurrent_wall_seconds'],
                padded_projection_seconds=projected,prospective_gate_seconds=3600,preflight_pass=True,
                pipeline_hard_limit_seconds=4200,observed_pipeline_seconds=meta.get('elapsed_seconds'),
                final=bool(args.final),main_completed=len(meta.get('main',[])),
                observed_failures=[r for r in inventory if r['status'] in ('failed','interrupted','cancelled')],
                invocation_metadata_inventory=inventory,
                evidence_scope='saved command/runtime/status arithmetic; no engineering model predictions or score arrays read')
    (out/'runtime_gate_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','pipeline_status','padded_projection_seconds','main_completed','final')}))


if __name__=='__main__':
    main()
