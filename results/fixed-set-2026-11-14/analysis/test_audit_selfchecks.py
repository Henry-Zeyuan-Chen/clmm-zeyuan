"""Synthetic scoring, corruption and semantic-pairing checks; no experiment data."""
from __future__ import annotations
import copy
import csv
import gzip
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
import audit_fixed_set as audit


def episode(seed,keys=('001','002'),values=('a','b'),family='seen'):
    data=dict(seed=seed,keys=list(keys),values=list(values),family=family)
    events=[types.SimpleNamespace(key=k,value=v,event_id=i,
        encoded=(f'fact {k} has value {v}.\n' if family=='seen' else f'value for {k} is {v}.\n').encode().rjust(32,b' '))
        for i,(k,v) in enumerate(zip(keys,values))]
    queries=[types.SimpleNamespace(key=k,target=v,answer_byte=ord(v)) for k,v in zip(keys,values)]
    return types.SimpleNamespace(seed=seed,events=events,queries=queries,to_dict=lambda:copy.deepcopy(data))

class FakeStreams:
    KEY_POOLS=dict(train=('001','002'),dev=('601','602'))
    Episode=types.SimpleNamespace(from_dict=lambda d:episode(d['seed'],d['keys'],d['values'],d['family']))
    @staticmethod
    def render_query(q):return f'value of key {q.key}: '.encode().rjust(31,b' ')

class FakeData:
    @staticmethod
    def fresh_same(bank,seed,count):return [episode(seed+i,values=('c','d')) for i in range(count)]
    @staticmethod
    def fresh_heldout(seed,count,family='seen'):return [episode(seed+i,keys=('601','602'),values=('e','f'),family=family) for i in range(count)]

def fixture(folder):
    bank=dict(episodes=[episode(0).to_dict()]);args=dict(regime='fixed',dev_seed=10,eval_episodes=4)
    groups=audit.group_data(bank,args,FakeData,FakeStreams)
    with gzip.open(folder/'episodes.json.gz','wt',encoding='utf8') as f:
        json.dump([dict(group=g,episodes=[e.to_dict() for e in eps]) for g,eps in groups.items()],f)
    rows=[];logits=[];summaries=[]
    for group,episodes in groups.items():
        for condition in audit.CONDS:
            for i,ep in enumerate(episodes):
                for rank,q in enumerate(ep.queries):
                    prefix=FakeStreams.render_query(q) if group=='train_query_only' else b''.join(e.encoded for e in ep.events)+FakeStreams.render_query(q)
                    z=np.zeros(256,dtype=np.float32);z[q.answer_byte if rank==0 else 255]=5
                    s=audit.score(z[None],[q.answer_byte])
                    row=dict(group=group,episode=i,base_pair=0 if group=='novel_values' else i,rank=rank,condition=condition,key=q.key,
                        target=q.answer_byte,prefix=prefix.decode(),**{k:(int(v[0]) if k in ('prediction','forced_prediction','correct') else float(v[0])) for k,v in s.items()})
                    rows.append(row);logits.append(z)
            subset=[r for r in rows if r['group']==group and r['condition']==condition]
            summaries.append(dict(group=group,condition=condition,episodes=len(episodes),queries=2*len(episodes),
                accuracy=.5,both_correct=0.,first_accuracy=1.,second_accuracy=0.,prediction_changes=1.,
                nll=float(np.mean([r['nll'] for r in subset])),legal_probability_mass=float(np.mean([r['legal_probability_mass'] for r in subset]))))
    audit.csvwrite(folder/'predictions.csv',rows);np.save(folder/'answer_logits.npy',np.array(logits));audit.write(folder/'summary.json',summaries)
    return bank,args

class ScoreChecks(unittest.TestCase):
    def test_uniform_full_vocabulary(self):
        s=audit.score(np.zeros((1,256),dtype=np.float32),[97])
        self.assertAlmostEqual(float(s['nll'][0]),math_log(256));self.assertAlmostEqual(float(s['forced_nll'][0]),math_log(16))
        self.assertAlmostEqual(float(s['legal_probability_mass'][0]),1/16)
    def test_correct_full_vocabulary(self):
        z=np.zeros((1,256));z[0,112]=10;s=audit.score(z,[112]);self.assertEqual(s['prediction'][0],112);self.assertEqual(s['correct'][0],1)
    def test_full_and_forced_choice_differ(self):
        z=np.zeros((1,256));z[0,255]=9;z[0,97]=8;s=audit.score(z,[97]);self.assertEqual(s['prediction'][0],255);self.assertEqual(s['forced_prediction'][0],97)
    def test_nonfinite_rejected(self):
        z=np.zeros((1,256));z[0,0]=np.nan
        with self.assertRaises(audit.AuditError):audit.score(z,[97])
    def test_wrong_shape_rejected(self):
        with self.assertRaises(audit.AuditError):audit.score(np.zeros((1,16)),[97])
    def test_illegal_target_rejected(self):
        with self.assertRaises(audit.AuditError):audit.score(np.zeros((1,256)),[0])
    def test_near_rejects_tampered_score(self):
        with self.assertRaises(audit.AuditError):audit.near(.1,.2,'score')
    def test_cluster_replicates_are_not_independent_units(self):
        p=[dict(base_pair=k,both_correct=value) for k in (0,1) for value in (0,1,0,1)]
        self.assertTrue(np.array_equal(audit.unit_vector(p,'both_correct','novel_values'),[.5,.5]))
    def test_unbalanced_clusters_rejected(self):
        with self.assertRaises(audit.AuditError):audit.unit_vector([dict(base_pair=0,both_correct=1),dict(base_pair=1,both_correct=0),dict(base_pair=1,both_correct=1)],'both_correct','novel_values')
    def test_pair_identity_not_incidental_order(self):
        a=[dict(identity=1,base_pair=0,both_correct=1)];b=[dict(identity=2,base_pair=0,both_correct=1)]
        with self.assertRaises(audit.AuditError):audit.compare(a,b,'both_correct','train',20,1)
    def test_paired_zero_difference(self):
        a=[dict(identity=i,base_pair=i,both_correct=i%2) for i in range(4)]
        r=audit.compare(a,a,'both_correct','train',100,1);self.assertEqual((r['difference'],r['lower'],r['upper']),(0.,0.,0.))
    def test_bootstrap_deterministic(self):
        self.assertEqual(audit.bootstrap([0,1,1],100,2),audit.bootstrap([0,1,1],100,2))

def math_log(x):return float(np.log(x))

class ArtifactChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='_selfcheck_',dir=Path(__file__).resolve().parent)
        self.folder=Path(self.temp.name).resolve()
        self.assertTrue(self.folder.is_relative_to(Path(__file__).resolve().parent));self.bank,self.args=fixture(self.folder)
    def tearDown(self):
        self.assertTrue(self.folder.resolve().is_relative_to(Path(__file__).resolve().parent));self.temp.cleanup()
    def audit(self):return audit.evaluation_audit(self.folder,self.args,self.bank,FakeData,FakeStreams,128)
    def mutate_rows(self,change):
        rows=audit.csvread(self.folder/'predictions.csv');change(rows);audit.csvwrite(self.folder/'predictions.csv',rows)
    def test_valid_fixture(self):
        vectors,summaries=self.audit();self.assertEqual(len(summaries),10);self.assertEqual(len(vectors[('novel_values','online')]),4)
    def test_prefix_corruption_rejected(self):
        self.mutate_rows(lambda r:r[0].__setitem__('prefix','bad'))
        with self.assertRaises(audit.AuditError):self.audit()
    def test_target_corruption_rejected(self):
        self.mutate_rows(lambda r:r[0].__setitem__('target',112))
        with self.assertRaises(audit.AuditError):self.audit()
    def test_duplicate_identity_rejected(self):
        self.mutate_rows(lambda r:r[1].__setitem__('rank',0))
        with self.assertRaises(audit.AuditError):self.audit()
    def test_prediction_corruption_rejected(self):
        self.mutate_rows(lambda r:r[0].__setitem__('prediction',255))
        with self.assertRaises(audit.AuditError):self.audit()
    def test_nll_corruption_rejected(self):
        self.mutate_rows(lambda r:r[0].__setitem__('nll',100))
        with self.assertRaises(audit.AuditError):self.audit()
    def test_summary_corruption_rejected(self):
        s=audit.read(self.folder/'summary.json');s[0]['both_correct']=1;audit.write(self.folder/'summary.json',s)
        with self.assertRaises(audit.AuditError):self.audit()
    def test_reset_non_noop_rejected(self):
        rows=audit.csvread(self.folder/'predictions.csv');z=np.load(self.folder/'answer_logits.npy');index=next(i for i,r in enumerate(rows) if r['condition']=='reset_memory')
        # Add a constant: softmax scores and predictions remain unchanged, but logits differ.
        z[index]+=1;np.save(self.folder/'answer_logits.npy',z)
        with self.assertRaises(audit.AuditError):self.audit()
    def test_truncated_logits_rejected(self):
        z=np.load(self.folder/'answer_logits.npy');np.save(self.folder/'answer_logits.npy',z[:-1])
        with self.assertRaises(audit.AuditError):self.audit()

class DisjointData:
    @staticmethod
    def fresh_same(bank,seed,count):
        variants=(('c','d'),('c','d'),('e','f'),('g','h'),('i','j'),('k','l'))
        for i in range(count):
            original=bank['episodes'][i%32]
            yield episode(seed+i,keys=original['keys'],values=variants[(i//32)%len(variants)])

class DisjointChecks(unittest.TestCase):
    def setUp(self):
        self.bank=dict(episodes=[episode(i,keys=(f'{2*i+1:03d}',f'{2*i+2:03d}')).to_dict() for i in range(32)])
    def selected(self,excluded=None):
        return audit.reconstruct_disjoint(self.bank,DisjointData,FakeStreams,excluded or set())
    def saved(self):
        selected,counters,contexts=self.selected()
        selection=dict(seed=90205,candidate_cap=8192,variants_per_base_pair=4,base_pairs=32,**counters)
        return dict(episodes=copy.deepcopy(selected),selection=copy.deepcopy(selection)),selection,selected,counters
    def test_training_overlap_is_rejected(self):
        first=next(DisjointData.fresh_same(self.bank,90205,8192));context=b''.join(e.encoded for e in first.events)
        selected,counters,contexts=self.selected({context})
        self.assertNotIn(context,contexts);self.assertGreater(counters['rejected_training'],0)
        self.assertNotEqual(selected[0]['source_candidate_index'],0)
    def test_duplicate_candidates_are_rejected(self):
        selected,counters,contexts=self.selected();self.assertEqual(len(contexts),128);self.assertEqual(len(set(contexts)),128)
        self.assertEqual(counters['rejected_duplicate'],32)
    def test_source_candidate_corruption_rejected(self):
        saved,selection,selected,counters=self.saved();saved['episodes'][0]['source_candidate_index']+=32
        with self.assertRaises(audit.AuditError):audit.check_disjoint_selection(saved,selection,selected,counters)
    def test_base_pair_corruption_rejected(self):
        saved,selection,selected,counters=self.saved();saved['episodes'][0]['base_pair']=1
        with self.assertRaises(audit.AuditError):audit.check_disjoint_selection(saved,selection,selected,counters)
    def test_counter_corruption_rejected(self):
        saved,selection,selected,counters=self.saved();selection['visited']+=1;saved['selection']=copy.deepcopy(selection)
        with self.assertRaises(audit.AuditError):audit.check_disjoint_selection(saved,selection,selected,counters)
    def test_selection_passes_without_corruption(self):
        saved,selection,selected,counters=self.saved();audit.check_disjoint_selection(saved,selection,selected,counters)
    def test_wrong_candidate_keys_rejected(self):
        bad=types.SimpleNamespace(fresh_same=lambda bank,seed,count:[episode(0,keys=('999','998'),values=('c','d'))])
        with self.assertRaises(audit.AuditError):audit.reconstruct_disjoint(self.bank,bad,FakeStreams,set())
    def test_original_values_in_candidate_rejected(self):
        bad=types.SimpleNamespace(fresh_same=lambda bank,seed,count:[episode(0,keys=self.bank['episodes'][0]['keys'])])
        with self.assertRaises(audit.AuditError):audit.reconstruct_disjoint(self.bank,bad,FakeStreams,set())
    def test_exhausted_candidate_stream_rejected(self):
        short=types.SimpleNamespace(fresh_same=lambda bank,seed,count:DisjointData.fresh_same(bank,seed,32))
        with self.assertRaises(audit.AuditError):audit.reconstruct_disjoint(self.bank,short,FakeStreams,set())

if __name__=='__main__':unittest.main(verbosity=2)
