"""Descriptive key-switch/value-swap and training-trajectory audit from saved bytes."""
import csv,hashlib,json,pathlib,numpy as np
OUT=pathlib.Path(r'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-key-binding-2026-11-15')
read=lambda p:json.loads(p.read_text());audit=read(OUT/'analysis/main_output_audit.json');results=[]
for c in audit['cases']:
 if c['mode']!='train':continue
 folder=pathlib.Path(c['run'])
 with (folder/'evaluation/predictions.csv').open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
 index={(r['group'],r['condition'],int(r['episode']),int(r['rank'])):r for r in rows}
 groups=[]
 for group in ('acquire_train_seen','acquire_train_swapped','acquire_dev_seen','acquire_dev_swapped','acquire_dev_heldout','acquire_dev_heldout_swapped'):
  pairs=[[index[group,'normal',i,rank] for rank in (0,1)] for i in range(256)]
  preds=np.array([[int(r['prediction']) for r in pair] for pair in pairs]);targets=np.array([[int(r['target']) for r in pair] for pair in pairs])
  present=(preds==targets[:,:1])|(preds==targets[:,1:]);zero=np.array([[int(index[group,'retrieval_zero',i,rank]['prediction']) for rank in (0,1)] for i in range(256)])
  groups.append(dict(group=group,answers=512,present_value_answers=int(present.sum()),first_record_value_answers=int((preds==targets[:,:1]).sum()),second_record_value_answers=int((preds==targets[:,1:]).sum()),same_prediction_pairs=int((preds[:,0]==preds[:,1]).sum()),key_change_pairs=int((preds[:,0]!=preds[:,1]).sum()),normal_vs_retrieval_zero_answer_changes=int((preds!=zero).sum()),normal_accuracy=float((preds==targets).mean()),retrieval_zero_accuracy=float((zero==targets).mean())))
 swaps=[]
 for group,swapped in (('acquire_train_seen','acquire_train_swapped'),('acquire_dev_seen','acquire_dev_swapped'),('acquire_dev_heldout','acquire_dev_heldout_swapped')):
  p=np.array([[int(index[group,'normal',i,r]['prediction']) for r in (0,1)] for i in range(256)]);s=np.array([[int(index[swapped,'normal',i,r]['prediction']) for r in (0,1)] for i in range(256)])
  swaps.append(dict(original_group=group,swapped_group=swapped,prediction_changes_across_value_swap=int((p!=s).sum()),answers=512,scope='A value-swap response alone is not requested-key binding; joint-four remains co-primary'))
 with (folder/'training.csv').open(newline='',encoding='utf8') as f:logs=list(csv.DictReader(f))
 windows=[]
 for label,subset in (('last32_K1',logs[224:256]),('last100_K2',logs[-100:])):
  windows.append(dict(window=label,updates=len(subset),first_step=int(subset[0]['step']),last_step=int(subset[-1]['step']),**{key:float(np.mean([float(r[key]) for r in subset])) for key in ('loss','accuracy','first_accuracy','second_accuracy','prediction_changes')}))
 results.append(dict(parent=c['parent'],encoding=c['encoding'],groups=groups,value_swap_responses=swaps,training_windows=windows))
codec=OUT/'packet/key_codec.json'
result=dict(status='passed',scope='Descriptive secondary analysis; no checkpoint selection or new efficacy claim',cases=results,codec_state=dict(serialized_bytes=codec.stat().st_size,sha256=hashlib.sha256(codec.read_bytes()).hexdigest(),key_identities=128,aliases=128,scope='Fixed external input preprocessing state, separate from839248 registered model parameters'))
(OUT/'analysis/behavior_controls.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
for c in results:print(json.dumps(dict(parent=c['parent'],encoding=c['encoding'],train_seen=c['groups'][0],swap=c['value_swap_responses'][0],training_windows=c['training_windows'])))
print('codec',result['codec_state'])
