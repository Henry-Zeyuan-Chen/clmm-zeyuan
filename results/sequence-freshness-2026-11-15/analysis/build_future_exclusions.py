"""Future-only exclusion union after consulting main/engineering outputs."""
import json
from pathlib import Path
import numpy as np
from audit_packet import read_json,sha
from audit_exclusions import structured_histories
from independent_helpers import context_signature,parse_record
ROOT=Path(__file__).resolve().parents[1]
base_path=ROOT/'analysis/work_smoke_expanded_signatures.json';base=set(read_json(base_path));assert len(base)==81347
all_histories=set(base);inputs=[];counts=[]
for label,packet in [('main',ROOT/'packet-final'),('engineering',ROOT/'engineering/packet-final')]:
    groups=read_json(packet/'episodes.json');worlds=read_json(packet/'training_plan.json.gz')
    real_eval=structured_histories(groups);real_train=structured_histories(worlds)
    rendered=set();raw_contexts=set()
    for row in read_json(packet/'samples.json'):
        raw=bytes.fromhex(row['prefix_hex']);assert (len(raw)-31)%32==0
        if len(raw)>31:raw_contexts.add(raw[:-31])
    with np.load(packet/'training.npz') as saved:
        lengths=saved['lengths']
        for name in ['periodic','periodic_bounded']:
            array=saved[name]
            for index,length in enumerate(lengths):
                for row in array[index,:,:int(length)-31]:raw_contexts.add(bytes(row))
    for context in raw_contexts:
        assert len(context)%32==0 and context
        rendered.add(context_signature([context[i:i+32] for i in range(0,len(context),32)]))
    before=len(all_histories);all_histories.update(real_eval|real_train|rendered)
    counts.append(dict(packet=label,real_training_histories=len(real_train),real_evaluation_histories=len(real_eval),
                      rendered_training_and_evaluation_histories=len(rendered),new_signatures_added=len(all_histories)-before))
    inputs.extend(dict(path=str(packet/name),sha256=sha(packet/name)) for name in ['episodes.json','training_plan.json.gz','samples.json','training.npz'])
union_path=ROOT/'analysis/future_exclusion_signatures.json';union_path.write_text(json.dumps(sorted(all_histories),indent=2)+'\n',encoding='utf8')
result=dict(status='complete_future_only',scope='For subsequent studies only; launch packets/unions remain byte-for-byte unchanged',
    base_histories=len(base),future_union_histories=len(all_histories),additional_now_consulted_histories=len(all_histories-base),
    includes='Chronological real semantic histories plus actual presented raw-record contexts from both trained policies and all evaluation arms; excludes label-free query-only empty contexts',
    units='SHA256 compact-JSON ordered string-key/value pairs, independent of surface template',packets=counts,
    base_sha256=sha(base_path),future_union_sha256=sha(union_path),inputs=inputs)
(ROOT/'analysis/future_exclusion_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='inputs'}))
