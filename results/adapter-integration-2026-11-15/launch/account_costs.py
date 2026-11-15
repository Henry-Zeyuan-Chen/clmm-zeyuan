"""Count saved slow/routing tensors and optimizer backing separately from fast state."""
import json
from pathlib import Path
import torch
ROOT=Path(r'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-adapter-integration-2026-11-15')
def backing(value):
    seen={}
    def visit(item):
        if isinstance(item,torch.Tensor):
            s=item.untyped_storage()
            seen[s._cdata]=s.nbytes()
        elif isinstance(item,dict):
            for v in item.values():visit(v)
        elif isinstance(item,(list,tuple)):
            for v in item:visit(v)
    visit(value)
    return sum(seen.values())
records=[]
for run in sorted(ROOT.iterdir()):
    if not (run/'optimizer.pt').is_file() or run.name.startswith(('smoke','throughput')):continue
    payload=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
    state=payload['model_state']
    slow={k:v for k,v in state.items() if k in ('key_encoder','value_encoder','decoder','W0','alpha','raw_eta')}
    buffers={k:v for k,v in state.items() if k not in slow}
    optimizer=torch.load(run/'optimizer.pt',map_location='cpu',weights_only=True)
    records.append(dict(run=run.name,slow_parameter_elements=sum(v.numel() for v in slow.values()),
        slow_parameter_backing_bytes=backing(slow),fixed_codebook_and_router_backing_bytes=backing(buffers),
        router_backing_bytes=backing(state['router_prototypes']),optimizer_tensor_backing_bytes=backing(optimizer),
        optimizer_file_bytes=(run/'optimizer.pt').stat().st_size,fast_trace_bytes_per_stream=65536,
        note='Optimizer is training-only and not retained by evaluation adapter; tensor backing excludes Python/allocator overhead'))
assert len(records)==8
result=dict(status='passed',scope='Actual saved tensor storage, not disk size or FLOP equivalence',runs=records,
            latency_scope='Run walltimes include imports/provenance; resource batch times include policy/validation/serialization, not bare-kernel inference latency',
            sparse_scope='Top1 updates one active module vsdense4 but vectorized dense candidates are still computed; no4xspeed ormemoryclaim')
(ROOT/'cost_accounting.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(records[0],indent=2))
