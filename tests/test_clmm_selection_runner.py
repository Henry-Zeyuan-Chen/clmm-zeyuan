from argparse import Namespace
import json
import pytest
import torch
from scripts.selection_experiment import score_logits,predicted_decode_prefix,guard_final,summarize


def test_full_head_is_primary_for_selector():
    logits=torch.zeros(2,256);logits[0,200]=9;logits[0,48]=8;logits[1,49]=8
    scores=score_logits(logits,torch.tensor([48,49]),'select')
    assert scores['prediction'].tolist()==[200,49]
    assert scores['forced_prediction'].tolist()==[48,49]
    with pytest.raises(ValueError):score_logits(logits,torch.tensor([97,49]),'select')
    with pytest.raises(ValueError):score_logits(logits[:,:16],torch.tensor([48,49]),'select')


def test_route_uses_predicted_byte_without_truth_metadata():
    row={'decode0_hex':b'zero'.hex(),'decode1_hex':b'one'.hex(),'decode_erased_hex':b'erased'.hex(),
         'slot':999,'target':-999,'selection_target':999}
    assert predicted_decode_prefix(row,48)==b'zero'
    assert predicted_decode_prefix(row,49)==b'one'
    for invalid in [0,47,50,255]:assert predicted_decode_prefix(row,invalid)==b'erased'


def test_final_seal_is_packet_and_source_bound(tmp_path):
    from scripts.fact_revision_experiment import sha256
    (tmp_path/'metadata.json').write_text('{}')
    args=Namespace(dataset='final',decision_lock=None,packet=tmp_path)
    packet={'kind':'main','source_commit':'sealed'}
    with pytest.raises(ValueError):guard_final(args,packet)
    lock={'source_commit':'sealed','packet_metadata_sha256':sha256(tmp_path/'metadata.json'),
          'phase':'confirmation_authorized','adaptive_training_complete':True}
    args.decision_lock=tmp_path/'lock.json';args.decision_lock.write_text(json.dumps(lock))
    guard_final(args,packet)
    lock['adaptive_training_complete']=False;args.decision_lock.write_text(json.dumps(lock))
    with pytest.raises(AssertionError):guard_final(args,packet)


def test_joint_metric_requires_both_questions_and_orders():
    rows=[]
    for variant in ['original','order','value','order_value']:
        for rank in [0,1]:rows.append(dict(group='dev_seen',variant=variant,condition='normal',episode=0,
            rank=rank,correct=int(not(variant=='order' and rank==1)),prediction=48+rank,nll=0.,legal_probability_mass=1.))
    summary=summarize(rows)
    assert summary['joint'][0]['joint4']==0
    assert summary['joint'][0]['joint4_swapped']==1
    assert summary['joint'][0]['joint8']==0
