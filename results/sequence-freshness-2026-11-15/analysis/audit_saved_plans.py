"""Independently check frozen evaluation byte prefixes and chronological replay plans."""
import argparse,collections,hashlib,json
from pathlib import Path
from audit_packet import read_json,render_record,sha
from audit_results import expected_evaluation
from independent_helpers import reference_schedule


def audit(packet):
    groups=read_json(packet/'episodes.json');plans=read_json(packet/'plans.json');samples=read_json(packet/'samples.json')
    expected_samples=expected_evaluation(groups);assert samples==expected_samples
    seen=set();writes=online=extra=stale=0;bounded_refreshed=0;plans_by_world=collections.defaultdict(dict)
    for plan in plans:
        group,index,arm=plan['group'],plan['episode'],plan['arm'];identity=(group,index,arm);assert identity not in seen;seen.add(identity)
        ep=groups[group][index];raw=[render_record(event) for event in ep['events']]
        expected=reference_schedule(raw,arm.removesuffix('_bounded'),ep['seed'],min(2,len(raw)),freshness='bounded' if arm.endswith('_bounded') else 'literal')
        assert len(expected)==len(plan['events'])==plan['total_records']
        latest={};count_stale=0;candidate=[]
        for event,reference in zip(plan['events'],expected):
            assert all(event[k]==reference[k] for k in ['source','kind','after_online'])
            source=ep['events'][event['source']]
            assert event['key']==int(source['key']) and event['value']==ord(source['value'])-97
            assert bytes.fromhex(event['record_hex'])==raw[event['source']]
            assert event['source']<=event['after_online']
            if event['kind']=='online':latest[source['key']]=source['value']
            is_stale=event['kind']=='extra' and latest[source['key']]!=source['value']
            assert event['stale']==is_stale;count_stale+=is_stale
            if arm.endswith('_bounded') and event['kind']=='extra':
                for key in ['sampled_source','refreshed','fallback']:assert event[key]==reference[key]
                assert event['sampled_value']==ord(ep['events'][event['sampled_source']]['value'])-97
                bounded_refreshed+=event['refreshed']
            if event['kind']=='extra':candidate.append(event.get('sampled_source',event['source']))
        assert count_stale==plan['stale_extras']
        assert plan['prime_count']==min(2,len(raw)) and plan['real_records']==len(raw)
        assert plan['extra_records']==len(expected)-len(raw)==(0 if arm=='online' else len(raw)-min(2,len(raw)))
        context=b''.join(raw[event['source']] for event in expected)
        assert plan['context_sha256']==hashlib.sha256(context).hexdigest()
        assert plan['peak_retained_raw_records']<=12 and plan['peak_retained_raw_bytes']==32*plan['peak_retained_raw_records']
        assert plan['transition_raw_record_envelope']<=13 and plan['transition_raw_byte_envelope']==32*plan['transition_raw_record_envelope']
        budget=plan['final_budget']
        assert budget['online_records']==len(raw) and budget['emitted_records']==len(expected)
        assert budget['extra_records']==plan['extra_records'] and budget['prime_records']==min(2,len(raw))
        assert budget['observe_records']==len(raw)-min(2,len(raw)) and budget['snapshot_pairs']==0
        assert budget['raw_payload_bytes']==32*budget['raw_payload_records']
        if arm in ['online','repeat']:assert budget['raw_payload_records']==budget['buffer_pairs']==0
        if arm.endswith('_bounded'):assert count_stale==0,'Current revision grid keeps all targets resident'
        plans_by_world[(group,index)][arm]=candidate
        writes+=len(expected);online+=len(raw);extra+=plan['extra_records'];stale+=count_stale
    expected_count=sum(len(eps)*(6 if name.startswith('revision') else 1) for name,eps in groups.items())
    assert len(seen)==expected_count
    for identity,arms in plans_by_world.items():
        if not identity[0].startswith('revision'):continue
        assert arms['continuous']==arms['periodic']==arms['continuous_bounded']==arms['periodic_bounded']
    return dict(status='passed',scope='frozen bytes and replay-plan reconstruction; no model execution',
        plans=len(plans),samples=len(samples),context_record_presentations=writes,online_record_presentations=online,
        extra_record_presentations=extra,stale_extra_presentations=stale,bounded_source_refreshes=bounded_refreshed,
        samples_input_bytes=sum(s['input_bytes'] for s in samples),
        files_sha256={name:sha(packet/name) for name in ['episodes.json','plans.json','samples.json']})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'saved_execution_review.json')
    args=parser.parse_args();result=audit(args.packet);args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
