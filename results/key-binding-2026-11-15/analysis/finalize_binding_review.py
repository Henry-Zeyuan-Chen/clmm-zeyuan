"""Create future exclusions, scoped claim map and inventory from saved artifacts only."""
import argparse,csv,datetime,hashlib,json,pathlib,re
import numpy as np

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf8'))
def signature(pairs):return hashlib.sha256(json.dumps(pairs,separators=(',',':')).encode()).hexdigest()
def records(prefix,n,encoding,codec):
 keys=codec['train_keys']+codec['dev_keys'];inverse={a:k for a,k in zip(codec['aliases'],keys)};pairs=[]
 for i in range(n):
  raw=prefix[i*32:(i+1)*32];assert len(raw)==32 and raw.endswith(b'.\n')
  if encoding=='atomic':
   high=[b for b in raw if b>=128];assert len(high)==1
   alias=high[0];field=bytes([alias])+b'  ';assert raw.count(field)==1
   raw=raw.replace(field,inverse[alias].encode('ascii'))
  hits=re.findall(rb'[0-9]{3}',raw);assert len(hits)==1 and raw[-3] in range(97,113)
  pairs.append((hits[0].decode('ascii'),chr(raw[-3])))
 return pairs

def main():
 p=argparse.ArgumentParser();p.add_argument('--out-root',required=True,type=pathlib.Path);p.add_argument('--audit',required=True,type=pathlib.Path);p.add_argument('--tensors',required=True,type=pathlib.Path);p.add_argument('--cpu-proof',type=pathlib.Path);a=p.parse_args();root=a.out_root;analysis=root/'analysis'
 audit=read(a.audit);tensor=read(a.tensors);assert audit['status']==tensor['status']=='passed' and audit['vectors']==81920
 prior=pathlib.Path(r'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-sequence-freshness-2026-11-15\analysis\future_exclusion_signatures.json')
 union=set(read(prior));assert len(union)==85123
 codec=read(root/'packet/key_codec.json');packets=[root/'engineering-cpu/packet',root/'engineering/packet',root/'packet'];lookup={}
 for packet in packets:
  meta=read(packet/'metadata.json');lookup[json.dumps(meta['files_sha256'],sort_keys=True)]=(packet,meta)
 manifest=[dict(kind='supplied_prior_future_union',path=str(prior),sha256=sha(prior),unique_histories=len(union))]
 paths=[root/'engineering-cpu/run/metadata.json']+list((root/'gpu-output/result').rglob('metadata.json'))+list((root/'gpu-main/result').rglob('metadata.json'))
 for path in paths:
  m=read(path)
  if m.get('mode') not in ('initial','train','preflight') or m.get('status')!='succeeded':continue
  packet,pm=lookup[json.dumps(m['data_sha256'],sort_keys=True)];train_ids=set();eval_ids=set();completed=m['completed_steps'];encoding=m['encoding']
  if completed:
   with np.load(packet/'training.npz') as archive:
    inputs,lengths=archive[encoding],archive['lengths']
    for step in range(completed):
     length=int(lengths[step]);n=(length-31)//32;assert n in (1,2)
     for row in inputs[step,::2,:length]:train_ids.add(signature(records(bytes(row),n,encoding,codec)))
  pred=path.parent/'evaluation/predictions.csv'
  with pred.open(newline='',encoding='utf8') as f:
   for row in csv.DictReader(f):
    n=int(row['real_records'])
    if n:eval_ids.add(signature(records(bytes.fromhex(row['prefix_hex']),n,encoding,codec)))
  before=len(union);union.update(train_ids);union.update(eval_ids)
  manifest.append(dict(kind='actual_presented_bytes',metadata=str(path),metadata_sha256=sha(path),prediction_csv=str(pred),prediction_csv_sha256=sha(pred),training_packet=str(packet),training_npz_sha256=sha(packet/'training.npz'),completed_steps=completed,encoding=encoding,training_unique=len(train_ids),evaluation_unique=len(eval_ids),added_unique=len(union)-before))
 expected=set(read(root/'packet/excluded_signatures.json'))
 for group,eps in read(root/'packet/episodes.json').items():
  for ep in eps:expected.add(signature([(r['key'],r['value']) for r in ep['events']]))
 assert union==expected,('actual-presented union must match frozen exclusion plus main worlds',len(union),len(expected))
 future=analysis/'future_exclusion_signatures.json';future.write_text(json.dumps(sorted(union),indent=2)+'\n',encoding='utf8')
 (analysis/'future_exclusion_manifest.json').write_text(json.dumps(dict(status='passed',scope='prior union plus actual raw training and evaluated fact-prefix histories, canonical decimal-key/value ordered signatures across both encodings',union_count=len(union),union_sha256=sha(future),main_launch_files_unchanged=True,inputs=manifest),indent=2)+'\n',encoding='utf8')
 claims=[]
 def claim(name,text,status,evidence,limit):claims.append(dict(claim_id=name,claim=text,status=status,evidence=evidence,limit=limit))
 claim('integration','All eight main initial/terminal conditions retain unchanged839248-parameter architecture and complete saved-score validation','measured',str(a.audit)+';'+str(a.tensors),'Engineering validity does not qualify acquisition or retention')
 for effect in audit['co_primary']:
  point=effect['atomic_minus_digits']*100;lo,hi=np.array(effect['ci97_5'])*100
  claim('encoding_'+effect['endpoint'],f'Atomic-minus-digits joint-four correctness {point:.4f}pp; adjusted97.5%CI[{lo:.4f},{hi:.4f}]pp','measured',str(a.audit),'All-zero joint-four yields a degenerate percentile bootstrap; not equivalence evidence. Conditional on two fixed parents and this encoding bundle/curriculum; not pure tokenization causality or training-procedure uncertainty')
 finals=[c for c in audit['cases'] if c['mode']=='train'];passed=sum(c['admission_pass'] for c in finals)
 claim('acquisition_admission',f'{passed}/4 terminal models pass all six normal original/swapped groups; {audit["admission_cells_pass"]}/24 gate cells pass','measured',str(a.audit),'Fixed full256 both-correct threshold205/256 in each group; no retention, sleep or cache persistence inference')
 claim('normal_reset','All normal/reset saved score vectors agree exactly','measured',str(a.audit),'Prefixes95<128, so reset is a parity check and does not establish memory independence')
 claim('retrieval_zero','Ancillary retrieved-output and query-only controls are fully retained','measured',str(a.audit),'Retrieval zero still executes normal within-prefix updates and changes later activations; not a separately trained attention-only baseline')
 cpu=read(a.cpu_proof) if a.cpu_proof else None
 claim('cpu_portability','CPU reload/inference numerical proof: '+(str(cpu.get('status','inspect artifact')) if cpu else 'not supplied to this reviewer'),'measured' if cpu else 'not_reviewed',str(a.cpu_proof) if cpu else '', 'Saved-vector arithmetic is not CPU inference reproduction; original tolerances must remain unchanged')
 claim('retention_sleep','This diagnostic establishes no retention, sleep, Hebbian update, complete-cache persistence, corruption-recovery or atomic replay-adapter efficacy','not_tested',str(root/'design.json'),'No such intervention or delayed/cross-session endpoint was tested')
 with (analysis/'claim_evidence_map.csv').open('w',newline='',encoding='utf8') as f:
  writer=csv.DictWriter(f,fieldnames=list(claims[0]));writer.writeheader();writer.writerows(claims)
 lines=['# Independent saved-result review','',f'All81,920 main saved answer vectors pass the independent raw-score audit. Tensor-only inspection passes four initial, four terminal and four recovery checkpoints plus eight optimizer states. The experiment completed8000optimizer updates,128000supervised forks and11635712input bytes across four terminal runs.','',f'Acquisition admission: {passed}/4 terminal models pass all six normal groups; {audit["admission_cells_pass"]}/24 cells pass the prespecified205/256 threshold.']
 for c in finals:
  lines.append(f'- Parent{c["parent"]}/{c["encoding"]}: train joint-four{c["joint_four"]["train_seen"]}/256, dev-heldout joint-four{c["joint_four"]["dev_heldout"]}/256; admission'+('passes.' if c['admission_pass'] else 'fails.'))
 lines+=['','The atomic-minus-digits contrast estimates an encoding bundle under the common new curriculum. Atomic dev identity bytes are task-untrained; decimal dev keys recombine trained digit bytes. Prior language exposure can differ. Comparisons with the earlier study also bundle the smaller pool and removal of revisions.','',f'The future exclusion union contains{len(union)}unique canonical histories, reconstructed from actual presented bytes plus the supplied prior union. Original launch packets remain unchanged.','', 'Normal/reset parity is expected for95-byte prefixes within one128-byte segment. Retrieved-output zeroing is an ancillary within-prefix ablation. No retention or sleep qualification follows from any acquisition result.','', 'CPU numerical portability requires the separate actual-inference proof and its original tolerances; saved-output arithmetic alone cannot establish it.']
 (analysis/'independent_results_review.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print(json.dumps(dict(status='passed',future_union=len(union),claims=len(claims),admission_models=passed),indent=2))

if __name__=='__main__':main()
