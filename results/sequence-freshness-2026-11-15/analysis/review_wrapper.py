"""Independent data-only bridge review. Does not import or execute a model."""
import argparse,copy,hashlib,importlib.util,json,sys,types
from pathlib import Path
from independent_helpers import reference_schedule,parse_record
sys.dont_write_bytecode=True


def run(repo):
    root=Path(__file__).resolve().parent
    # Load exactly two policy modules, bypassing clmm.pretrain's model imports.
    namespace='independent_policy_review'
    package=types.ModuleType(namespace);package.__path__=[str(repo/'clmm/pretrain')]
    sys.modules[namespace]=package
    for name in ['kv_scheduler','sequence_replay']:
        path=repo/f'clmm/pretrain/{name}.py'
        spec=importlib.util.spec_from_file_location(f'{namespace}.{name}',path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    Bridge=sys.modules[f'{namespace}.sequence_replay'].SequenceReplay
    templates=['fact {key} has value {value}.\n','key {key} has code {value}.\n',
               'value for {key} is {value}.\n','{key} maps to value {value}.\n']
    checks=[];cases=0;resumes=0;maxraw=0;raw_change_examples=0
    for count in [4,5,6,8,9,12,17,25]:
        records=[templates[i%4].format(key=f'{i%6:03d}',value=chr(97+(i//6+i)%16)).rjust(32).encode('ascii') for i in range(count)]
        for prime in [1,4]:
            for arm,fresh in [('online','literal'),('repeat','literal'),('continuous','literal'),('periodic','literal'),('continuous','bounded'),('periodic','bounded')]:
                expected=reference_schedule(records,arm,91323,prime,freshness=fresh)
                bridge=Bridge(arm,91323,freshness=fresh);events=[]
                for index,raw in enumerate(records):
                    prior=bridge.export_state()
                    state_copy=copy.deepcopy(prior)
                    restored=Bridge.from_state(state_copy)
                    actual=(bridge.prime(raw) if index<prime else bridge.observe(raw))
                    resumed=(restored.prime(raw) if index<prime else restored.observe(raw))
                    assert resumed==actual and restored.export_state()==bridge.export_state()
                    assert state_copy==prior
                    resumes+=1
                    events.extend(actual)
                    maxraw=max(maxraw,bridge.budget()['raw_payload_records'])
                    assert bridge.budget()['raw_payload_records']<=12
                    refs=bridge.export_state()['controller']
                    union={r[0] for r in refs['buffer']+refs['snapshot']}
                    assert {p['source'] for p in bridge.export_state()['payloads']}==union
                resumed=Bridge.from_state(bridge.export_state())
                final=bridge.finish();assert resumed.finish()==final
                assert bridge.finish()==[]
                events.extend(final)
                assert len(events)==len(expected)
                for event,reference in zip(events,expected):
                    assert all(event[key]==reference[key] for key in ['source','kind','after_online'])
                    assert event['record_bytes']==records[event['source']]
                    info=parse_record(event['record_bytes'])
                    assert (int(info['key']),ord(info['value'])-97)==(event['key'],event['value'])
                    if fresh=='bounded' and event['kind']=='extra':
                        for key in ['sampled_source','refreshed','fallback']:assert event[key]==reference[key]
                        raw_change_examples+=int(event['refreshed'])
                assert bridge.budget()['online_records']==count
                assert bridge.budget()['extra_records']==(0 if arm=='online' else count-prime)
                if arm in ('online','repeat'):assert bridge.budget()['raw_payload_records']==0
                cases+=1
    checks.extend(['all four raw grammars retained byte-for-byte','chronology matches independent raw reference',
        'every transition resumes with identical emissions and state','partial blocks and repeated finish audited',
        'live payload keys exactly equal deduplicated FIFO/snapshot sources','write budgets match real/postprime arrivals'])
    # A real bounded revision refresh proves distinct source bytes are exercised.
    assert raw_change_examples>0
    checks.append('bounded refresh cases exercised actual changed source payloads')
    rejected=0
    def reject(fn):
        nonlocal rejected
        try:fn()
        except (ValueError,RuntimeError,TypeError):rejected+=1;return
        raise AssertionError('Expected corruption rejection')
    bridge=Bridge('periodic',91323,freshness='bounded')
    for i in range(4):bridge.prime(records[i])
    bridge.observe(records[4]);valid=bridge.export_state()
    for mutate in [lambda s:s['counters'].__setitem__('extra_records',1),
                   lambda s:s['payloads'].pop(),
                   lambda s:s['payloads'][0].__setitem__('record_hex',templates[0].format(key='999',value='p').rjust(32).encode().hex()),
                   lambda s:s['controller']['snapshot'][0].__setitem__(0,99),
                   lambda s:s.__setitem__('schema_version',2)]:
        invalid=copy.deepcopy(valid);mutate(invalid);reject(lambda:Bridge.from_state(invalid))
    for bad in [b'value of key 123: '.rjust(31),b'fact 123 has value q.\n'.rjust(32),records[0].decode()]:
        before=bridge.export_state();reject(lambda:bridge.observe(bad));assert before==bridge.export_state()
    checks.append('malformed/schema/count/payload/future-source corruptions rejected atomically')
    return dict(status='passed',scope='data-only wrapper review; no model import/inference',
                cases=cases,transition_resume_checks=resumes,rejected_corruptions=rejected,
                max_retained_raw_records_observed=maxraw,bounded_source_refresh_examples=raw_change_examples,
                checks=checks,sources={str(path.relative_to(repo)):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [repo/'clmm/pretrain/kv_scheduler.py',repo/'clmm/pretrain/sequence_replay.py']})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);args=parser.parse_args()
    result=run(args.repo)
    (Path(__file__).resolve().parent/'wrapper_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))
