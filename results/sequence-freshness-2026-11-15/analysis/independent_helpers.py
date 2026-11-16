"""Independent raw-record, replay and statistics helpers; never imports model code."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
import re
import numpy as np

RECORD_BYTES=32
VALUES='abcdefghijklmnop'
RECORD_PATTERNS=tuple(re.compile(p) for p in (
    r'fact (\d{3}) has value ([a-p])\.\n',
    r'key (\d{3}) has code ([a-p])\.\n',
    r'value for (\d{3}) is ([a-p])\.\n',
    r'(\d{3}) maps to value ([a-p])\.\n'))
QUERY_PATTERNS=tuple(re.compile(p) for p in (
    r'value of key (\d{3}): ',r'code for key (\d{3}) = ',
    r'recall value for (\d{3}): ',r'key (\d{3}) now has value = '))


def parse_record(raw):
    raw=raw.encode('ascii') if isinstance(raw,str) else bytes(raw)
    if len(raw)!=32:raise AssertionError('Record must contain 32 bytes')
    text=raw.decode('ascii').lstrip(' ')
    hits=[(i,p.fullmatch(text)) for i,p in enumerate(RECORD_PATTERNS)]
    hits=[(i,m) for i,m in hits if m]
    if len(hits)!=1:raise AssertionError('Unrecognized or ambiguous raw record')
    index,match=hits[0]
    return dict(key=match[1],value=match[2],family='seen' if index<2 else 'heldout',template=index%2)


def parse_query(raw):
    raw=raw.encode('ascii') if isinstance(raw,str) else bytes(raw)
    if len(raw)!=31:raise AssertionError('Query prefix must contain 31 bytes')
    text=raw.decode('ascii').lstrip(' ')
    hits=[(i,p.fullmatch(text)) for i,p in enumerate(QUERY_PATTERNS)]
    hits=[(i,m) for i,m in hits if m]
    if len(hits)!=1:raise AssertionError('Unrecognized or ambiguous raw query')
    index,match=hits[0]
    return dict(key=match[1],family='seen' if index<2 else 'heldout',template=index%2)


def key_pools():
    ordered=sorted((f'{i:03d}' for i in range(1000)),key=lambda key:(
        hashlib.sha256(('clmm.fact.keys.v1:'+key).encode('ascii')).digest(),key))
    return dict(train=set(ordered[:600]),dev=set(ordered[600:800]),test=set(ordered[800:]))


def context_signature(records):
    semantic=[(p['key'],p['value']) for p in map(parse_record,records)]
    return hashlib.sha256(json.dumps(semantic,separators=(',',':')).encode('ascii')).hexdigest()


def reference_schedule(records,arm,seed,prime_count,capacity=8,block_size=4,freshness='literal'):
    """Reconstruct StreamingReplay semantics from raw records without production imports.

    Return source indices for each chronological online/extra presentation, plus
    frozen sampled sources. Prime count is explicit because integration policies
    may differ from the older empty-first-block batch scheduler.
    """
    if arm not in ('online','repeat','continuous','periodic'):raise ValueError(arm)
    if freshness not in ('literal','bounded'):raise ValueError(freshness)
    if freshness=='bounded' and arm not in ('continuous','periodic'):raise ValueError(arm)
    if not 0<=prime_count<=len(records):raise ValueError(prime_count)
    decoded=[parse_record(record) for record in records]
    fifo=OrderedDict();snapshot=[];offset=0;events=[]
    rng=np.random.default_rng(np.random.SeedSequence([seed,77]))
    peak=0
    def arrive(source):
        if arm in ('continuous','periodic'):
            fifo.pop(decoded[source]['key'],None)
            fifo[decoded[source]['key']]=source
            while len(fifo)>capacity:fifo.popitem(last=False)
    def extra(sampled,after):
        current=fifo.get(decoded[sampled]['key'])
        source=sampled if freshness=='literal' or current is None else current
        return dict(source=source,kind='extra',after_online=after,
                    sampled_source=sampled,fallback=current is None,
                    refreshed=source!=sampled)
    for source in range(prime_count):
        arrive(source);events.append(dict(source=source,kind='online',after_online=source))
    for source in range(prime_count,len(records)):
        if arm in ('continuous','periodic') and offset==0:
            if not fifo:raise AssertionError('Replay requires prefill')
            available=list(fifo.values())
            snapshot=[int(rng.choice(available)) for _ in range(block_size)]
        arrive(source)
        events.append(dict(source=source,kind='online',after_online=source))
        if arm=='repeat':events.append(dict(source=source,kind='extra',after_online=source))
        elif arm in ('continuous','periodic'):
            if arm=='continuous':events.append(extra(snapshot[offset],source))
            offset+=1
            if offset==block_size:
                if arm=='periodic':events.extend(extra(sample,source) for sample in snapshot)
                snapshot=[];offset=0
        peak=max(peak,len(set(fifo.values())|set(snapshot)))
    if arm=='periodic' and offset:
        events.extend(extra(sample,len(records)-1) for sample in snapshot[:offset])
    if peak>capacity+block_size:raise AssertionError('Retained-payload bound exceeded')
    return events


def score_logits(logits,targets):
    """Recompute full-256 metrics using float64 arithmetic from saved logits."""
    logits=np.asarray(logits,dtype=np.float64);targets=np.asarray(targets,dtype=np.int64)
    if logits.ndim!=2 or logits.shape[1]!=256 or len(targets)!=len(logits):raise AssertionError('Logit geometry')
    if not np.isfinite(logits).all():raise AssertionError('Nonfinite logits')
    if not ((0<=targets)&(targets<256)).all():raise AssertionError('Invalid target')
    shifted=logits-logits.max(axis=1,keepdims=True)
    exp=np.exp(shifted);denominator=exp.sum(axis=1)
    legal=logits[:,ord('a'):ord('p')+1]
    legal_shift=legal-legal.max(axis=1,keepdims=True)
    legal_denominator=np.exp(legal_shift).sum(axis=1)
    target_legal=targets-ord('a')
    if not ((0<=target_legal)&(target_legal<16)).all():raise AssertionError('Nonlegal target')
    pred=logits.argmax(axis=1)
    return dict(prediction=pred,forced_prediction=legal.argmax(axis=1)+ord('a'),
        correct=(pred==targets).astype(np.int64),
        nll=np.log(denominator)-shifted[np.arange(len(logits)),targets],
        forced_nll=np.log(legal_denominator)-legal_shift[np.arange(len(logits)),target_legal],
        legal_probability_mass=exp[:,ord('a'):ord('p')+1].sum(axis=1)/denominator)


def paired_bootstrap(values,seed=91304,draws=5000,confidence=.95):
    """Values have one row per whole episode and columns for jointly paired arms."""
    values=np.asarray(values,dtype=float)
    if values.ndim==1:values=values[:,None]
    if not len(values) or not np.isfinite(values).all():raise AssertionError('Invalid bootstrap units')
    rng=np.random.default_rng(seed)
    estimates=np.empty((draws,values.shape[1]),dtype=float)
    for start in range(0,draws,200):
        stop=min(start+200,draws)
        indices=rng.integers(len(values),size=(stop-start,len(values)))
        estimates[start:stop]=values[indices].mean(axis=1)
    alpha=(1-confidence)/2
    return dict(estimate=values.mean(axis=0),lower=np.quantile(estimates,alpha,axis=0),
                upper=np.quantile(estimates,1-alpha,axis=0),units=len(values),draws=draws,
                seed=seed,confidence=confidence)

