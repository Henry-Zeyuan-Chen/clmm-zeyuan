import copy,runpy,sys
from pathlib import Path
import numpy as np
import pytest
from clmm.pretrain import format_training as t
from clmm.pretrain.record_selection import make_codec,make_world,VARIANTS


def test_one_indexed_lr_boundaries_and_budget():
    assert t.learning_rate(1,'warmup')==pytest.approx(.0000006)
    assert t.learning_rate(499,'warmup')==pytest.approx(.0002994)
    for step in (500,501,6000):assert t.learning_rate(step,'warmup')==.0003
    constant=np.array([t.learning_rate(s) for s in range(1,6001)])
    warmup=np.array([t.learning_rate(s,'warmup') for s in range(1,6001)])
    assert np.all(constant>0) and np.all(warmup>0) and np.array_equal(constant[499:],warmup[499:])
    assert constant.sum()==pytest.approx(1.8) and warmup.sum()==pytest.approx(1.72515)
    assert np.prod(1-.01*warmup)>np.prod(1-.01*constant)
    for step in (0,-1,True,1.5):
        with pytest.raises(ValueError):t.learning_rate(step,'warmup')
    with pytest.raises(ValueError):t.learning_rate(1,'unknown')


@pytest.mark.parametrize('encoding',['digits','atomic'])
@pytest.mark.parametrize('variant',VARIANTS)
def test_new_wording_independent_blocks_and_targets(encoding,variant):
    c=make_codec();w=make_world(923456,codec=c)
    old=t.render(w,c,encoding,'F0Q0',variant)
    facts=t.render(w,c,encoding,'F3Q0',variant);queries=t.render(w,c,encoding,'F0Q3',variant);both=t.render(w,c,encoding,'F3Q3',variant)
    for a,f,q,b in zip(old,facts,queries,both):
        raw=bytes.fromhex(b['prefix_hex']);assert len(raw)==95 and raw[:3]==b'[0]' and raw[32:35]==b'[1]'
        assert f['prefix_hex'][128:]==a['prefix_hex'][128:] and q['prefix_hex'][:128]==a['prefix_hex'][:128]
        assert raw[:64]==bytes.fromhex(f['prefix_hex'])[:64] and raw[64:]==bytes.fromhex(q['prefix_hex'])[64:]
        assert a['target']==f['target']==q['target']==b['target']
    erased=t.render(w,c,encoding,'F3Q3',variant,'cue_erased')
    assert erased[0]['prefix_hex']==erased[1]['prefix_hex'] and erased[0]['target']!=erased[1]['target']


def test_old_families_keep_identity_and_new_cells_are_sealed():
    c=make_codec();w=make_world(934567,codec=c).to_dict()
    assert len(t.templates()['facts'])==3 and len(t.templates(True)['facts'])==4
    assert set(t.templates(True)['facts'][3]).isdisjoint(sum(t.templates()['facts'],[]))
    assert set(t.templates(True)['queries'][3]).isdisjoint(sum(t.templates()['queries'],[]))
    rows=t.evaluation_samples({'train_keys':[w]},c,cells=t.WARMUP_CELLS)
    assert len(rows)==152 and {r['cell'] for r in rows}==set(t.WARMUP_CELLS)
    assert all('2' not in r['cell'] for r in rows)
    with pytest.raises(ValueError):t.evaluation_samples({'train_keys':[w]},c,confirmation=False,cells=t.WARMUP_CELLS)
    old=dict(status='all_eight_terminal_locked',packet_metadata_sha256='packet',initializers=[dict(checkpoint_sha256=str(i)) for i in range(4)],
             terminals=[dict(parent=p,encoding=e,arm=a,status='succeeded',completed_steps=6000,checkpoint_sha256=f'{p}{e}{a}') for p in (68101,68105) for e in ('digits','atomic') for a in ('constant','warmup')])
    assert t.validate_confirmation_lock(old,'packet','0',arms=['constant','warmup'])
    with pytest.raises(ValueError):t.validate_confirmation_lock(old,'packet','0')


@pytest.mark.parametrize('devices,expected_peak',[(['7'],1),(['3','5'],2)])
def test_gpu_worker_ownership_and_serialization(monkeypatch,tmp_path,devices,expected_peak):
    path=Path(__file__).resolve().parents[1]/'scripts/format_training_pipeline.py'
    monkeypatch.setattr(sys,'argv',[str(path),'--base',str(tmp_path)])
    module=runpy.run_path(str(path),run_name='ownership_test');subprocess=module['subprocess'];active=0;peak=0;visible=[]
    class FinishedProcess:
        returncode=0
        def __init__(self,*args,**kw):
            nonlocal active,peak
            active+=1;peak=max(peak,active);visible.append(kw['env']['CUDA_VISIBLE_DEVICES'])
        def poll(self):return 0
        def wait(self,**kw):
            nonlocal active
            active-=1;return 0
    monkeypatch.setattr(subprocess,'Popen',FinishedProcess)
    module['parallel']('main',devices)
    assert peak==expected_peak and active==0 and visible==([devices[0]]*2 if len(devices)==1 else devices)
