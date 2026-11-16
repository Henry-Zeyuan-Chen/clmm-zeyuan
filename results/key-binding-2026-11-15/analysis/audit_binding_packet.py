"""Independent pure-data packet audit. Never imports torch or model code."""
from __future__ import annotations
import argparse, copy, gzip, hashlib, json, math, pathlib, sys
import numpy as np

RECORD = {'seen': ('fact {key} has value {value}.\n', 'key {key} has code {value}.\n'),
          'heldout': ('value for {key} is {value}.\n', '{key} maps to value {value}.\n')}
QUERY = {'seen': ('value of key {key}: ', 'code for key {key} = '),
         'heldout': ('recall value for {key}: ', 'key {key} now has value = ')}
VALUES = tuple('abcdefghijklmnop')
SCHEME = 'sha256(clmm.fact.keys.v1:+key), key tie-break; ranked 600/200/200'
ORDER = sorted((f'{i:03d}' for i in range(1000)), key=lambda k: (hashlib.sha256(('clmm.fact.keys.v1:'+k).encode('ascii')).digest(), k))
POOLS = {'train': ORDER[:600], 'dev': ORDER[600:800]}
POOLHASH = {k: hashlib.sha256('\n'.join(v).encode('ascii')).hexdigest() for k,v in POOLS.items()}

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    with (gzip.open(path,'rt',encoding='utf8') if path.suffix=='.gz' else path.open(encoding='utf8')) as f: return json.load(f)
def signature(records): return hashlib.sha256(json.dumps([(r['key'],r['value']) for r in records],separators=(',',':')).encode()).hexdigest()
def collect(value):
    found=set()
    if isinstance(value,list):
        if value and all(isinstance(r,dict) and isinstance(r.get('key'),str) and len(r['key'])==3 and r['key'].isascii() and r['key'].isdigit() and r.get('value') in VALUES for r in value): found.add(signature(value))
        for child in value: found.update(collect(child))
    elif isinstance(value,dict):
        for child in value.values(): found.update(collect(child))
    return found

def codec():
    return dict(schema_version=1,train_keys=POOLS['train'][:96],dev_keys=POOLS['dev'][:32],
                aliases=list(map(int,np.random.Generator(np.random.PCG64(91401)).permutation(np.arange(128,256)))),
                atomic_field='one alias byte followed by two ASCII spaces')

def world(seed,facts=2,split='train',family='seen'):
    rng=np.random.Generator(np.random.PCG64(int(seed)))
    keys=rng.choice(POOLS[split][:96 if split=='train' else 32], facts,replace=False).tolist()
    values=rng.choice(list(VALUES),facts,replace=False).tolist()
    events=[dict(event_id=i,key=k,value=v,version=1,kind='fact',template_family=family,template_index=int(rng.integers(2))) for i,(k,v) in enumerate(zip(keys,values))]
    qi=int(rng.integers(2))
    queries=[dict(key=r['key'],target=r['value'],old_value=None,category='stable',initial_event_id=i,latest_event_id=i,template_family=family,template_index=qi) for i,r in enumerate(events)]
    return dict(format_version=1,seed=int(seed),facts=facts,delay=0,split=split,template_family=family,revision_fraction=0.,include_recent=False,key_pool_scheme=SCHEME,key_pool_sha256=POOLHASH[split],events=events,queries=queries)

def swap(ep):
    ep=copy.deepcopy(ep)
    ep['events'][0]['value'],ep['events'][1]['value']=ep['events'][1]['value'],ep['events'][0]['value']
    for i,r in enumerate(ep['events']): ep['queries'][i]['target']=r['value']
    return ep

def reword(ep):
    ep=copy.deepcopy(ep);ep['template_family']='heldout'
    for row in ep['events']+ep['queries']: row['template_family']='heldout'
    return ep

def train_worlds(steps,contexts,seed):
    worlds=[]
    for step in range(steps):
        k1=step<math.ceil(.128*steps)
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,step])))
        eps=[world(int(s),1 if k1 else 2) for s in rng.integers(8_000_000_000,9_000_000_000,size=contexts)]
        worlds.append(dict(step=step+1,stage='k1' if k1 else 'k2',episodes=eps))
    return worlds

def eval_worlds(excluded,seed,n):
    rng=np.random.Generator(np.random.PCG64(seed));seen=set(excluded);groups={}
    for split in ('train','dev'):
        accepted=[]
        while len(accepted)<n:
            ep=world(int(rng.integers(9_000_000_000,10_000_000_000)),split=split)
            ids={signature(ep['events']),signature(swap(ep)['events'])}
            if ids&seen: continue
            seen.update(ids);accepted.append(ep)
        groups[f'acquire_{split}_seen']=accepted
        groups[f'acquire_{split}_swapped']=[swap(ep) for ep in accepted]
    groups['acquire_dev_heldout']=[reword(ep) for ep in groups['acquire_dev_seen']]
    groups['acquire_dev_heldout_swapped']=[reword(ep) for ep in groups['acquire_dev_swapped']]
    return groups

def field(raw,key,encoding,c):
    assert raw.count(key.encode('ascii'))==1
    if encoding=='digits': return raw
    i=(c['train_keys']+c['dev_keys']).index(key)
    at=raw.index(key.encode('ascii'));result=raw.replace(key.encode('ascii'),bytes([c['aliases'][i]])+b'  ')
    assert len(result)==len(raw) and raw[:at]==result[:at] and raw[at+3:]==result[at+3:]
    return result

def examples(ep,c,encoding,query_only=False):
    context=b''
    if not query_only:
        for r in ep['events']:
            raw=RECORD[r['template_family']][r['template_index']].format(**r).encode('ascii').rjust(32,b' ')
            assert len(raw)==32
            context+=field(raw,r['key'],encoding,c)
    queries=ep['queries'] if len(ep['queries'])==2 else ep['queries']*2
    answer=[]
    for q in queries:
        raw=QUERY[q['template_family']][q['template_index']].format(**q).encode('ascii').rjust(31,b' ')
        assert len(raw)==31
        answer.append((context+field(raw,q['key'],encoding,c),ord(q['target'])))
    return answer

def samples(data,c):
    rows=[]
    for enc in ('digits','atomic'):
        for group,eps in data.items():
            for cond in ('normal','reset_memory','retrieval_zero'):
                for i,ep in enumerate(eps):
                    for rank,(x,y) in enumerate(examples(ep,c,enc)):
                        rows.append(dict(encoding=enc,group=group,arm='online',condition=cond,episode=i,rank=rank,key=ep['queries'][rank]['key'],target=y,old=-1,prefix_hex=x.hex(),input_bytes=len(x),real_records=2,extra_records=0,stale_extras=0))
        for group in ('acquire_train_seen','acquire_dev_seen'):
            for i,ep in enumerate(data[group]):
                for rank,(x,y) in enumerate(examples(ep,c,enc,True)):
                    rows.append(dict(encoding=enc,group=group.replace('acquire_','query_only_'),arm='online',condition='normal',episode=i,rank=rank,key=ep['queries'][rank]['key'],target=y,old=-1,prefix_hex=x.hex(),input_bytes=len(x),real_records=0,extra_records=0,stale_extras=0))
    return rows

def audit(packet):
    meta=read(packet/'metadata.json');c=read(packet/'key_codec.json');assert c==codec()
    for name,sha in meta['files_sha256'].items(): assert digest(packet/name)==sha,name
    for name,sha in meta['source_sha256'].items(): assert digest(packet/'sources'/name)==sha,name
    assert 'clmm/pretrain/key_binding.py' in meta['source_sha256'] and 'scripts/key_binding_experiment.py' in meta['source_sha256']
    worlds=read(packet/'training_plan.json.gz')
    assert worlds==train_worlds(meta['steps'],meta['contexts'],meta['train_seed'])
    own=collect(worlds);exclusions=read(packet/'excluded_signatures.json');excluded=set(exclusions)
    assert len(exclusions)==len(excluded)
    manifest=read(packet/'exclusion_manifest.json');expected=set(own)
    for item in manifest['inputs']:
        path=pathlib.Path(item['path']);assert digest(path)==item['sha256'],str(path)
        value=read(path)
        ids=set(value) if isinstance(value,list) and all(isinstance(v,str) and len(v)==64 for v in value) else collect(value)
        assert len(ids)==item['histories'];expected.update(ids)
    assert expected==excluded and len(own)==manifest['current_training_unique'] and len(excluded)==manifest['union_count']
    data=read(packet/'episodes.json');assert data==eval_worlds(excluded,meta['development_seed'],meta['episodes'])
    eval_ids=collect(data);assert not eval_ids&excluded
    assert len(eval_ids)==meta['episodes']*4
    source=read(packet/'samples.json');assert source==samples(data,c)
    assert len(source)==meta['episodes']*80
    budgets={'answers':0,'bytes':0,'k1_steps':0,'k2_steps':0,'unique_train_keys':set(),'targets':set()}
    with np.load(packet/'training.npz') as arrays:
        assert set(arrays.files)=={'digits','atomic','targets','lengths'}
        for enc in ('digits','atomic'):
            assert arrays[enc].dtype==np.uint8 and arrays[enc].shape==(meta['steps'],meta['contexts']*2,95)
        assert arrays['targets'].dtype==np.uint8 and arrays['lengths'].dtype==np.int16
        for i,w in enumerate(worlds):
            n=63 if w['stage']=='k1' else 95
            assert arrays['lengths'][i]==n
            budgets[w['stage']+'_steps']+=1
            for enc in ('digits','atomic'):
                inp=arrays[enc][i];assert not np.any(inp[:,n:])
                for j,ep in enumerate(w['episodes']):
                    for rank,(x,y) in enumerate(examples(ep,c,enc)):
                        assert bytes(inp[j*2+rank,:n])==x and arrays['targets'][i,j*2+rank]==y
            budgets['answers']+=len(arrays['targets'][i]);budgets['bytes']+=n*len(arrays['targets'][i])
            budgets['unique_train_keys'].update(r['key'] for ep in w['episodes'] for r in ep['events'])
            budgets['targets'].update(int(v) for v in arrays['targets'][i])
    assert budgets['answers']==meta['training_answers'] and budgets['bytes']==meta['training_input_bytes']
    assert meta['case_evaluation_vectors']==meta['episodes']*40
    assert budgets['unique_train_keys']<=set(c['train_keys'])
    for enc in ('digits','atomic'):
        chosen=[r for r in source if r['encoding']==enc]
        assert len(chosen)==meta['case_evaluation_vectors']
        assert sum(r['condition']=='normal' for r in chosen)==meta['episodes']*16
        assert sum(r['condition']=='reset_memory' for r in chosen)==sum(r['condition']=='retrieval_zero' for r in chosen)==meta['episodes']*12
        buckets={}
        for i,r in enumerate(chosen): buckets.setdefault((r['condition'],r['input_bytes']),[]).append((i,r))
        normal=[r['prefix_hex'] for _,r in buckets['normal',95]]
        assert normal==[r['prefix_hex'] for _,r in buckets['reset_memory',95]]
    budgets['unique_train_keys']=len(budgets['unique_train_keys']);budgets['targets']=sorted(budgets['targets'])
    result=dict(status='passed',packet=str(packet),source_commit=meta['source_commit'],models_constructed=False,training_arrays_exact=True,training_worlds_independently_regenerated=True,evaluation_worlds_independently_regenerated=True,evaluation_samples_exact=True,codec_exact=True,hashes_verified=True,design_hash_bound='design.json' in meta['files_sha256'],exclusion_union=len(excluded),training_unique_histories=len(own),evaluation_unique_histories=len(eval_ids),evaluation_rows_all_encodings=len(source),per_case_vectors=meta['case_evaluation_vectors'],budget=budgets)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('packets',nargs='+',type=pathlib.Path);parser.add_argument('--out',required=True,type=pathlib.Path);args=parser.parse_args()
    result={'status':'passed','numpy':np.__version__,'audits':[audit(p) for p in args.packets]}
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
