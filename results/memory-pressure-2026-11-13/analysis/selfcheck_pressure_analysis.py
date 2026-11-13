"""Small deterministic checks of the independent pressure analyzer; no neural training."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=Path(r'C:/Users/henry/Documents/Codex/clmm-replay')
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(HERE))
import pressure_analysis as analysis
from scripts.memory_pressure_probe import generate_data,evaluate_batch
spec=importlib.util.spec_from_file_location('pressure_test_models',REPO/'tests/test_clmm_memory_pressure.py')
fixture=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)

checked=0
for facts in (8,16,64,256):
    data=generate_data(3,facts,256,8,[0,8,64],91371,32,8)
    metadata=dict(nkeys=256,nvals=8,working_capacity=32,snapshot_capacity=8,
                  delays=[0,8,64],eval_seed=91371)
    analysis.verify_trace(data,3,facts,metadata)
    common,specific=analysis.reconstruct_rows(data)
    for arm in analysis.ARMS:
        rows=evaluate_batch(fixture.RecordingReader(nkeys=256),data,arm)
        for row in rows:
            di=[0,8,64].index(row['delay'])
            episode=row['episode']
            col=row['query_rank']+{'stable':0,'revised':2,'recent':4}[row['condition']]
            for fields in (common,specific[arm]):
                for name,array in fields.items():
                    expected=array[di,episode,col]
                    if name in ('future_writes_since_aux','future_writes_since_replay') and expected<0:
                        assert row[name]==''
                    else:
                        assert row[name]==expected,(facts,arm,name,row[name],expected)
            checked+=1
    corrupt={name:(value.copy() if isinstance(value,np.ndarray) else value) for name,value in data.items()}
    corrupt['sample_value_sources'][0,0,0]=facts+2
    try:
        analysis.verify_trace(corrupt,3,facts,metadata)
    except AssertionError:
        pass
    else:
        raise AssertionError('Future value source escaped reconstruction')

indices=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.int32)
ratios,valid=analysis.bootstrap_ratios(np.array([[0,2],[0,0]]),np.array([0,2]),indices)
assert valid.tolist()==[False,True,True,True]
assert np.isnan(ratios[:,0]).all()
np.testing.assert_array_equal(ratios[:,1:],[[100,100,100],[0,0,0]])
assert analysis.interval(ratios[0]-ratios[1])==(100.,100.)
ratios,_=analysis.bootstrap_ratios(np.array([[1,0],[0,1]]),np.array([1,3]),np.array([[0,1]]))
np.testing.assert_array_equal(ratios[:,0],[25.,25.])
empty,valid=analysis.bootstrap_ratios(np.zeros((2,2)),np.zeros(2),indices)
assert not valid.any() and analysis.interval(empty[0])==(None,None)

# A deliberately unequal eligible-query count must still use ratio-of-sums.
correct=np.zeros((4,1,2,8),dtype=np.int8)
correct[3,0,0,0]=1
correct[1,0,1,1]=1
eligible=np.zeros((1,2,8),dtype=bool)
eligible[0,0,0]=True
eligible[0,1,1:4]=True
expected={64:({'working_resident':~eligible,'shadow_ever_selected_for_replay':eligible},None)}
meta=dict(episodes=2,delays=[256],checkpoints=[dict(index=0,init_seed=48101,train_seed=48102)])
cells,contrasts,episodes,hash_value=analysis.analyze_results(Path('synthetic'),meta,{0:{64:correct}},expected)
primary=next(row for row in contrasts if row['condition']=='old' and row['stratum']=='nonresident_selected' and row['left']=='periodic' and row['right']=='repeat')
assert primary['eligible_queries']==4 and primary['eligible_episodes']==2
assert primary['difference_pp']==0
assert sum(row['eligible_queries'] for row in episodes if row['condition']=='old')==4

result=dict(status='passed',metadata_rows_compared_to_recording_reader=checked,
    fact_counts=[8,16,64,256],arms=list(analysis.ARMS),future_source_corruption_rejected=True,
    bootstrap_zero_denominators_checked=True,bootstrap_ratio_of_sums_checked=True,
    subgroup_ratio_of_sums_end_to_end_checked=True,neural_inference_or_training=False,
    analyzer_sha256=analysis.sha256(HERE/'pressure_analysis.py'))
(HERE/'pressure_analysis_selfcheck.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
