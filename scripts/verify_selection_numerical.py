"""Actual CPU inference on original saved GPU batch memberships; no training."""
import argparse,gzip,hashlib,json,os,sys,time,traceback
from contextlib import nullcontext
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--source',required=True,type=Path);P.add_argument('--packet',required=True,type=Path)
P.add_argument('--out',required=True,type=Path);A=P.parse_args();sys.path.insert(0,str(A.source.resolve()))
import numpy as np
import torch
from clmm.pretrain import load_checkpoint,zero_memory_output,reset_memory_at_segments
from scripts.selection_experiment import state_sha,predicted_decode_prefix
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(91602);torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def scope(model,condition):
    if condition=='reset_memory':return reset_memory_at_segments(model)
    if condition=='retrieval_zero':return zero_memory_output(model)
    return nullcontext()
records=[];failures=[];start=time.perf_counter();A.out.mkdir(parents=True,exist_ok=False)
manifest=json.loads((A.packet/'manifest.json').read_text())
for entry in manifest['cases']:
    directory=A.packet/entry['evaluation']
    with gzip.open(directory/'batches.json.gz','rt',encoding='utf8') as f:r=json.load(f)
    for relative,digest in r['source_sha256'].items():assert sha(A.source/relative)==digest,relative
    assert sha(directory/'expected.npz')==r['expected_sha256']
    checkpoint=A.packet/r['checkpoint'];assert sha(checkpoint)==r['checkpoint_sha256']
    model,config,payload=load_checkpoint(checkpoint,map_location='cpu');model.eval()
    assert sum(p.numel() for p in model.parameters())==839248 and config.segment_len==128
    assert payload['source_info']['source_commit']==r['source_commit']
    before=state_sha(model);rng=torch.random.get_rng_state().clone();offset=0;detail=[]
    expected=np.load(directory/'expected.npz')
    assert expected['answer_logits'].shape==(entry['queries'],256)
    if r['compose']:assert expected['selection_logits'].shape==(entry['queries'],256)
    with torch.no_grad():
        for batch in r['batches']:
            rows=batch['rows'];x=torch.tensor([list(bytes.fromhex(row['prefix_hex'])) for row in rows])
            size=len(rows)
            with scope(model,batch['condition']):
                result=model(x)[:,-1]
                selector_error=None;selector_probability_error=None;selector_mismatch=0;route_mismatches=0
                if r['compose']:
                    target=torch.tensor(expected['selection_logits'][offset:offset+size])
                    selector_error=float((result-target).abs().max());selector_mismatch=int((result.argmax(-1)!=target.argmax(-1)).sum())
                    selector_probability_error=float((result.softmax(-1)-target.softmax(-1)).abs().max())
                    prefixes=[predicted_decode_prefix(row,int(pred)) for row,pred in zip(rows,result.argmax(-1))]
                    route_mismatches=sum(prefix.hex()!=row['decode_prefix_hex'] for prefix,row in zip(prefixes,rows))
                    result=model(torch.tensor([list(p) for p in prefixes]))[:,-1]
                target=torch.tensor(expected['answer_logits'][offset:offset+size]);offset+=size
                err=float((result-target).abs().max());prob=float((result.softmax(-1)-target.softmax(-1)).abs().max())
                mismatch=int((result.argmax(-1)!=target.argmax(-1)).sum())
                passed=err<=.002 and prob<=.0002 and mismatch==0 and route_mismatches==0 and (selector_error is None or selector_error<=.002 and selector_probability_error<=.0002 and selector_mismatch==0)
                row=dict(condition=batch['condition'],group=batch['group'],variant=batch['variant'],queries=size,
                    max_logit_error=err,max_probability_error=prob,argmax_mismatches=mismatch,selector_max_logit_error=selector_error,
                    selector_max_probability_error=selector_probability_error,selector_argmax_mismatches=selector_mismatch,route_mismatches=route_mismatches,passed=passed)
                detail.append(row)
                if not passed:failures.append(dict(evaluation=r['evaluation'],**row))
    assert offset==entry['queries'] and before==state_sha(model) and torch.equal(rng,torch.random.get_rng_state())
    record=dict(evaluation=r['evaluation'],queries=offset,checkpoint_sha256=r['checkpoint_sha256'],weight_and_RNG_pure=True,batches=detail)
    records.append(record);(A.out/'progress.json').write_text(json.dumps(dict(completed_cases=len(records),failures=failures),indent=2))
result=dict(status='passed' if not failures else 'failed',source_commit=payload['source_info']['source_commit'],cases=records,failures=failures,
    queries=sum(x['queries'] for x in records),batches=sum(len(x['batches']) for x in records),
    max_logit_error=max(x['max_logit_error'] for r in records for x in r['batches']),
    max_probability_error=max(x['max_probability_error'] for r in records for x in r['batches']),
    selector_max_logit_error=max((x['selector_max_logit_error'] for r in records for x in r['batches'] if x['selector_max_logit_error'] is not None),default=None),
    selector_max_probability_error=max((x['selector_max_probability_error'] for r in records for x in r['batches'] if x['selector_max_probability_error'] is not None),default=None),
    environment=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,device='CPU',threads=2),
    input_manifest_sha256=sha(A.packet/'manifest.json'),script_sha256=sha(Path(__file__)),elapsed_seconds=time.perf_counter()-start,
    scope='Selected original GPU batches only, including each normal group/variant and every control. Does not establish all-vector cross-device equivalence.')
(A.out/'numerical_proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('cases','failures')}))
assert not failures,failures
