"""Secondary fixed-checkpoint diagnostics from audited whole-episode rows."""
import argparse,collections,csv,json
from pathlib import Path
import numpy as np
from audit_results import load_csv,write_csv
from independent_helpers import paired_bootstrap


def interval_row(values,seed=91304,confidence=.95,draws=5000):
    result=paired_bootstrap(values,seed,draws,confidence)
    return dict(estimate=float(result['estimate'][0]),lower=float(result['lower'][0]),upper=float(result['upper'][0]),
                episodes=result['units'],confidence=confidence,draws=draws,seed=seed)


def run(audited,draws=5000):
    rows=load_csv(audited/'episode_metrics.csv')
    for row in rows:
        for key in ['parent','segment','episode','first_correct','second_correct','both_correct','prediction_changes','first_prediction','second_prediction','first_target','second_target']:
            row[key]=int(row[key])
        for key in ['accuracy','nll','legal_probability_mass','obsolete_rate']:row[key]=float(row[key])
    finals=[row for row in rows if row['mode']=='train']
    cases=sorted({(row['case'],row['parent'],row['segment'],row['training_arm']) for row in finals})
    counterfactual=[];memory=[];revision=[];gates=[]
    for case,parent,segment,training_arm in cases:
        subset=[row for row in finals if row['case']==case]
        index={(r['group'],r['arm'],r['condition'],r['episode']):r for r in subset}
        episodes=sorted({r['episode'] for r in subset if r['group']=='acquire_train_seen'})
        joint=[];changed=[]
        for episode in episodes:
            a=index[('acquire_train_seen','online','normal',episode)]
            b=index[('acquire_train_swapped','online','normal',episode)]
            assert (a['first_target'],a['second_target'])==(b['second_target'],b['first_target'])
            joint.append(a['both_correct']*b['both_correct'])
            changed.append(((a['first_prediction']!=b['first_prediction'])+(a['second_prediction']!=b['second_prediction']))/2)
        base=dict(case=case,parent=parent,segment=segment,training_arm=training_arm)
        counterfactual.append(dict(base,metric='joint_four_answer_correct',correct_count=sum(joint),
            arbitrary_relabel_prediction_change_rate=float(np.mean(changed)),**interval_row(joint,draws=draws)))
        for group in ['acquire_train_seen','acquire_dev_seen','acquire_dev_heldout']:
            units=sorted({r['episode'] for r in subset if r['group']==group})
            normal=[index[(group,'online','normal',i)]['both_correct'] for i in units]
            reset=[index[(group,'online','reset_memory',i)]['both_correct'] for i in units]
            contrast=interval_row(np.array(normal)-reset,draws=draws)
            gate=bool(np.mean(normal)>=.8)
            gates.append(dict(base,group=group,both_correct=sum(normal),episodes=len(units),accuracy=float(np.mean(normal)),acquisition_pass=gate,
                memory_dependence_required=segment==32,memory_dependence_pass=contrast['lower']>0 if segment==32 else '',
                normal_minus_reset=contrast['estimate'],lower=contrast['lower'],upper=contrast['upper']))
            memory.append(dict(base,group=group,contrast='normal-minus-segment-reset both-correct',**contrast))
        for group in ['revision_dev_seen','revision_dev_heldout']:
            units=sorted({r['episode'] for r in subset if r['group']==group})
            for contrast,high,low in [('boundedP-minus-literalP','periodic_bounded','periodic'),
                                     ('boundedC-minus-literalC','continuous_bounded','continuous'),
                                     ('boundedP-minus-repeat','periodic_bounded','repeat'),
                                     ('boundedC-minus-repeat','continuous_bounded','repeat'),
                                     ('boundedP-minus-boundedC','periodic_bounded','continuous_bounded')]:
                for metric in ['both_correct','accuracy','obsolete_rate']:
                    differences=[index[(group,high,'normal',i)][metric]-index[(group,low,'normal',i)][metric] for i in units]
                    revision.append(dict(base,group=group,contrast=contrast,metric=metric,**interval_row(differences,draws=draws)))
    write_csv(audited/'counterfactual_metrics.csv',counterfactual);write_csv(audited/'acquisition_gates.csv',gates)
    write_csv(audited/'memory_dependence.csv',memory);write_csv(audited/'revision_contrasts.csv',revision)
    result=dict(status='passed',scope='secondary development diagnostics; fixed checkpoints and paired histories',
        final_cases=len(cases),counterfactual_rows=len(counterfactual),acquisition_gate_rows=len(gates),
        passed_acquisition_gates=sum(row['acquisition_pass'] for row in gates),revision_contrast_rows=len(revision),draws=draws,seed=91304,
        all_final_case_gate_sets_pass=bool(gates) and all(row['acquisition_pass'] for row in gates))
    (audited/'secondary_analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audited',type=Path,required=True);parser.add_argument('--draws',type=int,default=5000)
    args=parser.parse_args();run(args.audited,args.draws)
