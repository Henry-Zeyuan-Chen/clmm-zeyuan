"""Supplement frozen exclusions using old raw training arrays and JSONL episodes."""
from __future__ import annotations
import argparse,collections,gzip,hashlib,json
from pathlib import Path
import numpy as np
from independent_helpers import parse_record,parse_query,context_signature
from audit_packet import read_json,render_record,sha


def structured_histories(value):
    result=set()
    if isinstance(value,list):
        pairs=[]
        for item in value:
            if not isinstance(item,dict):break
            key,observed_value=item.get('key'),item.get('value')
            if not (isinstance(key,str) and len(key)==3 and key.isascii() and key.isdigit() and isinstance(observed_value,str) and len(observed_value)==1 and 'a'<=observed_value<='p'):break
            pairs.append((key,observed_value))
        else:
            if pairs:result.add(hashlib.sha256(json.dumps(pairs,separators=(',',':')).encode()).hexdigest())
        for child in value:result.update(structured_histories(child))
    elif isinstance(value,dict):
        for child in value.values():result.update(structured_histories(child))
    return result


def raw_training_histories(path):
    with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
    raw=arrays['input_bytes'];targets=arrays['targets'];histories=set();answers=0
    if raw.ndim==3:
        chunks=[(raw[step],targets[step]) for step in range(len(raw))]
    else:
        assert raw.ndim==1 and 'step_offsets' in arrays
        offsets=arrays['step_offsets'];assert len(offsets)==len(targets)+1 and offsets[0]==0 and offsets[-1]==len(raw)
        chunks=[]
        for step in range(len(targets)):
            flat=raw[offsets[step]:offsets[step+1]];batch=len(targets[step]);assert len(flat)%batch==0
            chunks.append((flat.reshape(batch,-1),targets[step]))
    for batch,labels in chunks:
        for row,label in zip(batch,labels):
            prefix=bytes(row);assert (len(prefix)-31)%32==0
            records=[prefix[i:i+32] for i in range(0,len(prefix)-31,32)]
            observed=[parse_record(record) for record in records];query=parse_query(prefix[-31:])
            current={item['key']:item['value'] for item in observed}
            assert ord(current[query['key']])==int(label),'Old raw training label inconsistent with observed records'
            histories.add(context_signature(records));answers+=1
    return histories,answers


def run(packet,history_root,engineering=None):
    declared=set(read_json(packet/'excluded_signatures.json'));manifest=read_json(packet/'exclusion_manifest.json')
    new_training=read_json(packet/'training_plan.json.gz')
    train_histories=structured_histories(new_training)
    assert len(train_histories)==manifest['current_training_unique']
    manifest_union=set(train_histories);inputs=[]
    for record in manifest['inputs']:
        path=Path(record['path']);assert sha(path)==record['sha256']
        content=read_json(path)
        ids=set(content) if record.get('kind')=='supplied_exclusions' else structured_histories(content)
        assert len(ids)==record['histories']
        manifest_union.update(ids)
    assert manifest_union==declared and len(declared)==manifest['union_count']
    roots=[history_root/name for name in ['clmm-fact-revision-2026-11-14','clmm-paired-acquisition-2026-11-14','clmm-fixed-set-2026-11-14']]
    extra=set();seen_digests=set()
    for root in roots:
        for path in sorted(root.rglob('evaluation_data.jsonl.gz')):
            digest=sha(path)
            if digest in seen_digests:continue
            seen_digests.add(digest);ids=set();episode_lines=0
            with gzip.open(path,'rt',encoding='utf8') as handle:
                for line in handle:
                    if '"type": "episodes"' not in line:continue
                    item=json.loads(line);assert item['type']=='episodes';ids.update(structured_histories(item['episodes']));episode_lines+=1
            extra.update(ids);inputs.append(dict(path=str(path),sha256=digest,kind='JSONL real episodes',histories=len(ids),episode_lines=episode_lines))
        for path in sorted(root.rglob('training_inputs.npz')):
            digest=sha(path)
            if digest in seen_digests:continue
            seen_digests.add(digest);ids,answers=raw_training_histories(path)
            extra.update(ids);inputs.append(dict(path=str(path),sha256=digest,kind='actual raw training prefixes',histories=len(ids),answer_forks=answers))
    engineering_train=set()
    if engineering:
        engineering_train=structured_histories(read_json(engineering/'training_plan.json.gz'));extra.update(engineering_train)
    expanded=declared|extra
    groups=read_json(packet/'episodes.json');evaluation={context_signature([render_record(r) for r in ep['events']]) for eps in groups.values() for ep in eps}
    overlap=sorted(evaluation&expanded);assert not overlap,'Fixed main evaluation overlaps expanded prior/engineering histories'
    result=dict(status='passed',scope='historical raw data/exclusions only; no inference/training',
        declared_exclusions=len(declared),declared_manifest_inputs=len(manifest['inputs']),new_training_histories=len(train_histories),
        supplemental_unique_histories=len(extra),additional_histories=len(expanded-declared),expanded_exclusions=len(expanded),
        main_evaluation_unique_histories=len(evaluation),overlap_with_main_evaluation=len(overlap),engineering_training_histories=len(engineering_train),
        inputs=inputs,declared_set_sha256=sha(packet/'excluded_signatures.json'))
    return result,expanded

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--history-root',type=Path,required=True);parser.add_argument('--engineering',type=Path)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'supplemental_exclusion_review.json')
    args=parser.parse_args();result,expanded=run(args.packet,args.history_root,args.engineering)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    args.out.with_name(args.out.stem+'_expanded_signatures.json').write_text(json.dumps(sorted(expanded),indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'}))


