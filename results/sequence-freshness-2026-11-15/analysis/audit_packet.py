"""Independent saved-packet audit, with no production imports or model execution."""
from __future__ import annotations
import argparse,collections,csv,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
from independent_helpers import key_pools,parse_record,parse_query,reference_schedule,context_signature

RECORDS={'seen':('fact {key} has value {value}.\n','key {key} has code {value}.\n'),
         'heldout':('value for {key} is {value}.\n','{key} maps to value {value}.\n')}
QUERIES={'seen':('value of key {key}: ','code for key {key} = '),
         'heldout':('recall value for {key}: ','key {key} now has value = ')}
POOLS=key_pools()

def read_json(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8-sig') as handle:return json.load(handle)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def render_record(event):
    return RECORDS[event['template_family']][event['template_index']].format(**event).rjust(32).encode('ascii')

def render_query(query):
    return QUERIES[query['template_family']][query['template_index']].format(**query).rjust(31).encode('ascii')

def validate_episode(ep,training=False):
    assert ep['split'] in POOLS and ep['split']!='test','Reserved-test episode in development packet'
    if training:assert ep['split']=='train' and ep['template_family']=='seen'
    facts,delay=ep['facts'],ep['delay'];rev=math.floor(facts*ep['revision_fraction'])
    assert facts in (1,2) and len(ep['events'])==facts+rev+delay
    initial={};latest={};allkeys=set();revised=set();raw=[]
    for i,event in enumerate(ep['events']):
        assert event['event_id']==i and event['key'] in POOLS[ep['split']]
        assert event['template_family']==ep['template_family']
        kind='fact' if i<facts else 'revision' if i<facts+rev else 'distractor'
        assert event['kind']==kind
        if kind=='revision':
            assert event['key'] in initial and event['key'] not in revised
            assert event['value']!=initial[event['key']]['value'] and event['version']==2
            revised.add(event['key'])
        else:
            assert event['key'] not in allkeys and event['version']==1
            if kind=='fact':initial[event['key']]=event
        allkeys.add(event['key']);latest[event['key']]=event
        current=[latest[k]['value'] for k in initial]
        assert len(set(current))==len(current),'Distinct-current-target constraint failed'
        record=render_record(event);info=parse_record(record)
        assert (info['key'],info['value'],info['family'],info['template'])==(event['key'],event['value'],event['template_family'],event['template_index'])
        raw.append(record)
    assert not ep['include_recent'] and len(ep['queries'])==facts
    assert len({q['template_index'] for q in ep['queries']})==1,'Within-pair query wording differs'
    for query,(key,first) in zip(ep['queries'],initial.items()):
        last=latest[key]
        assert query['key']==key and query['target']==last['value']
        assert query['initial_event_id']==first['event_id'] and query['latest_event_id']==last['event_id']
        assert query['template_family']==ep['template_family']
        assert query['category']==('revised' if key in revised else 'stable')
        assert query['old_value']==(first['value'] if key in revised else None)
        parsed=parse_query(render_query(query));assert parsed['key']==key
    return raw

def expected_samples(ep,arm):
    raw=validate_episode(ep)
    policy_arm=arm.removesuffix('_bounded')
    expected=reference_schedule(raw,policy_arm,ep['seed'],min(2,len(raw)),
        freshness='bounded' if arm.endswith('_bounded') else 'literal')
    context=b''.join(raw[event['source']] for event in expected)
    queries=ep['queries'] if len(ep['queries'])==2 else ep['queries']*2
    return [(context+render_query(query),ord(query['target'])) for query in queries],expected

def extract_exclusions(value):
    if isinstance(value,list) and all(isinstance(item,str) and len(item)==64 for item in value):return set(value)
    if isinstance(value,dict):
        for key in ['excluded_signatures','signatures','excluded_histories']:
            if key in value:return extract_exclusions(value[key])
    raise AssertionError('Unsupported exclusion-set format; supply a JSON SHA256 list')

def audit_packet(training_worlds_path,training_arrays_path,evaluation_worlds_path,exclusions_path=None):
    worlds=read_json(training_worlds_path);groups=read_json(evaluation_worlds_path)
    if isinstance(worlds,dict):worlds=worlds.get('worlds',worlds.get('training_worlds'))
    if isinstance(groups,dict) and 'groups' in groups:groups=groups['groups']
    assert isinstance(worlds,list) and worlds
    with np.load(training_arrays_path,allow_pickle=False) as saved:
        arrays={name:saved[name] for name in saved.files}
    assert set(arrays)=={'periodic','periodic_bounded','targets','lengths'}
    assert arrays['periodic'].shape==arrays['periodic_bounded'].shape
    assert arrays['periodic'].shape[0]==len(worlds) and arrays['periodic'].shape[2]==351
    batch=arrays['periodic'].shape[1];assert batch%2==0
    assert arrays['targets'].shape==(len(worlds),batch) and arrays['lengths'].shape==(len(worlds),)
    train_signatures=set();stage_counts=collections.Counter();byte_total=0;forks=0;real=extras=0;changed_steps=0
    expected_plans=0
    for index,world in enumerate(worlds):
        assert world['step']==index+1 and len(world['episodes'])*2==batch
        stage_counts[world['stage']]+=1
        expected_stage=('k1' if index/len(worlds)<.128 else 'k2' if index/len(worlds)<.64
                        else 'revision' if (index-int(.64*len(worlds)))%4==3 else 'k2')
        assert world['stage']==expected_stage
        raw_eps=[]
        for ep in world['episodes']:
            raw=validate_episode(ep,training=True)
            assert ep['facts']==(1 if world['stage']=='k1' else 2)
            assert ep['delay']==(2 if world['stage']=='revision' else 0)
            assert ep['revision_fraction']==(1 if world['stage']=='revision' else 0)
            train_signatures.add(context_signature(raw));raw_eps.append(raw)
        for arm in ['periodic','periodic_bounded']:
            samples=[]
            for ep in world['episodes']:
                sample,plan=expected_samples(ep,arm);samples.extend(sample);expected_plans+=1
                if arm=='periodic':real+=len(ep['events']);extras+=len(plan)-len(ep['events'])
            lengths={len(x) for x,y in samples};assert len(lengths)==1
            length=lengths.pop();assert length==int(arrays['lengths'][index])
            expected_x=np.array([list(x) for x,y in samples],dtype=np.uint8)
            expected_y=np.array([y for x,y in samples],dtype=np.uint8)
            assert np.array_equal(expected_x,arrays[arm][index,:,:length]),f'Training input mismatch {arm}/{index}'
            assert not arrays[arm][index,:,length:].any(),'Nonzero storage-only padding'
            assert np.array_equal(expected_y,arrays['targets'][index])
        length=int(arrays['lengths'][index]);byte_total+=batch*length;forks+=batch
        changed_steps+=int(not np.array_equal(arrays['periodic'][index],arrays['periodic_bounded'][index]))
        if world['stage']!='revision':assert np.array_equal(arrays['periodic'][index],arrays['periodic_bounded'][index])
    if len(worlds)==2000 and batch==16:
        assert dict(stage_counts)==dict(k1=256,k2=1564,revision=180)
        assert byte_total==3646208 and forks==32000
    expected_groups={'acquire_train_seen','acquire_train_swapped','acquire_dev_seen','acquire_dev_heldout','revision_dev_seen','revision_dev_heldout'}
    assert set(groups)==expected_groups
    historic=extract_exclusions(read_json(exclusions_path)) if exclusions_path else set()
    primary_sigs=set();group_summary=[]
    for name,episodes in groups.items():
        ids=[]
        for ep in episodes:
            raw=validate_episode(ep)
            assert ep['split']==('train' if name.startswith('acquire_train') else 'dev')
            assert ep['template_family']==('heldout' if name.endswith('_heldout') else 'seen')
            assert ep['facts']==2 and ep['delay']==(2 if name.startswith('revision') else 0)
            identifier=context_signature(raw);ids.append(identifier)
            assert identifier not in train_signatures,'Evaluation exact semantic history occurs in training'
            assert identifier not in historic,'Evaluation semantic history occurs in consulted exclusion union'
            if not name.endswith('_heldout'):
                assert identifier not in primary_sigs,'Unexpected duplicated evaluation semantic history'
                primary_sigs.add(identifier)
            # Every replay arm has identical byte prefixes on no-extra acquisition.
            if name.startswith('acquire'):
                assert expected_samples(ep,'periodic')[0]==expected_samples(ep,'periodic_bounded')[0]==expected_samples(ep,'online')[0]
        assert len(ids)==len(set(ids))
        group_summary.append(dict(group=name,episodes=len(episodes),queries=2*len(episodes),unique_semantic_histories=len(set(ids))))
    for left,right in [('acquire_dev_seen','acquire_dev_heldout'),('revision_dev_seen','revision_dev_heldout')]:
        assert len(groups[left])==len(groups[right])
        for a,b in zip(groups[left],groups[right]):
            assert a['seed']==b['seed'] and context_signature([render_record(r) for r in a['events']])==context_signature([render_record(r) for r in b['events']])
            assert [q['key'] for q in a['queries']]==[q['key'] for q in b['queries']]
            assert [q['target'] for q in a['queries']]==[q['target'] for q in b['queries']]
    for a,b in zip(groups['acquire_train_seen'],groups['acquire_train_swapped']):
        assert a['seed']==b['seed']
        assert [r['key'] for r in a['events']]==[r['key'] for r in b['events']]
        assert [r['value'] for r in a['events']]==[r['value'] for r in b['events']][::-1]
        assert [render_query(q) for q in a['queries']]==[render_query(q) for q in b['queries']]
    return dict(status='passed',scope='saved packet reconstruction only; no model inference',training_steps=len(worlds),
        stage_counts=dict(stage_counts),per_case_answer_forks=forks,per_case_input_bytes=byte_total,
        real_context_records=real,extra_context_records=extras,training_histories=len(train_signatures),
        training_revision_steps_with_changed_payloads=changed_steps,independently_reconstructed_context_plans=expected_plans,
        consulted_exclusion_count=len(historic),groups=group_summary,
        input_sha256={str(p):sha(p) for p in [training_worlds_path,training_arrays_path,evaluation_worlds_path]+([exclusions_path] if exclusions_path else [])})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--training-worlds',type=Path,required=True);parser.add_argument('--training-arrays',type=Path,required=True)
    parser.add_argument('--evaluation-worlds',type=Path,required=True);parser.add_argument('--exclusions',type=Path)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'packet_review.json')
    args=parser.parse_args();result=audit_packet(args.training_worlds,args.training_arrays,args.evaluation_worlds,args.exclusions)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


