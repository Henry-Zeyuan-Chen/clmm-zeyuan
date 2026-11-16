"""CPU data-only selfchecks for independent evidence helpers (no model imports)."""
import sys
sys.dont_write_bytecode=True
import importlib.util,json
from pathlib import Path
import numpy as np
from independent_helpers import parse_record,parse_query,key_pools,reference_schedule,score_logits,paired_bootstrap

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('review_scheduler',Path('C:/Users/henry/Documents/Codex/clmm-replay/clmm/pretrain/kv_scheduler.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
checks=[]
def checked(name,fn):
    fn();checks.append(name)
def expect_failure(fn):
    try:fn()
    except (AssertionError,ValueError,UnicodeError):return
    raise AssertionError('Expected evidence rejection')

for text,family,index in [('fact 123 has value a.\n','seen',0),('key 123 has code a.\n','seen',1),
                          ('value for 123 is a.\n','heldout',0),('123 maps to value a.\n','heldout',1)]:
    result=parse_record(text.rjust(32));assert result==dict(key='123',value='a',family=family,template=index)
checks.append('four raw templates independently parsed')
for text in ['value of key 123: ','code for key 123 = ','recall value for 123: ','key 123 now has value = ']:
    assert parse_query(text.rjust(31))['key']=='123'
checks.append('four target-free query templates independently parsed')
for bad in ['fact 123 has value q.\n'.rjust(32),'fact 123 has value a.\n'.rjust(31),'fact 123 has value a. x'.rjust(32)]:
    expect_failure(lambda:parse_record(bad))
checks.append('malformed raw payloads rejected')
pools=key_pools();assert list(map(len,pools.values()))==[600,200,200]
assert not pools['train']&pools['dev'] and not pools['train']&pools['test'] and not pools['dev']&pools['test']
checks.append('independent key namespace partition')

matrix=0
for count in [4,5,6,9,12,17]:
    raw=[f'fact {i%6:03d} has value {chr(97+(i//6+i)%16)}.\n'.rjust(32) for i in range(count)]
    for prime in [1,4]:
        for capacity,block in [(8,4),(4,2)]:
            for arm,freshness in [('online','literal'),('repeat','literal'),('continuous','literal'),('periodic','literal'),('continuous','bounded'),('periodic','bounded')]:
                expected=reference_schedule(raw,arm,91323,prime,capacity,block,freshness)
                policy=module.StreamingReplay(arm,91323,capacity,block,nkeys=1000,nvalues=16,freshness=freshness)
                observed=[]
                for index,record in enumerate(raw):
                    decoded=parse_record(record)
                    args=(int(decoded['key']),ord(decoded['value'])-97)
                    observed.extend([policy.prime(*args)] if index<prime else policy.step(*args))
                observed.extend(policy.finish())
                assert [(r['source'],r['kind'],r['after_online']) for r in observed]==[(r['source'],r['kind'],r['after_online']) for r in expected]
                if freshness=='bounded':
                    for actual,reference in zip(observed,expected):
                        if actual['kind']=='extra':
                            for key in ['sampled_source','refreshed','fallback']:assert actual[key]==reference[key]
                matrix+=1
checks.append(f'{matrix} independent reference schedules match policy on partial/full blocks and revisions')
logits=np.zeros((2,256));targets=np.array([97,112]);scores=score_logits(logits,targets)
assert np.allclose(scores['nll'],np.log(256)) and np.allclose(scores['forced_nll'],np.log(16))
assert np.allclose(scores['legal_probability_mass'],16/256) and np.array_equal(scores['prediction'],[0,0])
checks.append('raw 256-logit score formulas checked at uniform distribution')
for bad in [np.full((2,256),np.nan),np.zeros((2,16))]:expect_failure(lambda:score_logits(bad,targets))
checks.append('nonfinite/wrong-vocabulary logits rejected')
a=paired_bootstrap(np.array([[1,0],[0,1]]),seed=42,draws=100)
b=paired_bootstrap(np.array([[1,0],[0,1]]),seed=42,draws=100)
assert all(np.array_equal(a[k],b[k]) for k in ['estimate','lower','upper'])
checks.append('whole-episode paired bootstrap deterministic')
result=dict(status='passed',checks=checks,schedule_cases=matrix,scope='data/statistics helpers only; no model inference')
(ROOT/'helper_selfchecks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
