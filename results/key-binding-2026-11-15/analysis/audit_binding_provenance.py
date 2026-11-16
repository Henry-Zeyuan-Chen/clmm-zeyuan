"""Inspect frozen Git blobs, exact packet source and parent tensors; no model construction."""
import hashlib,json,pathlib,subprocess,torch
OUT=pathlib.Path(r'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-key-binding-2026-11-15')
REPO=pathlib.Path(r'C:\Users\henry\Documents\Codex\2026-10-22\i\work\clmm-key-binding\source-frozen')
COMMIT='5e8c61829ff85ecea3c9314d4ccd8dfef3268b68'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==COMMIT
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def state_sha(state):
 h=hashlib.sha256()
 for name,t in sorted(state.items()):
  h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
results=[];parents_seen={}
for packet in (OUT/'engineering-cpu/packet',OUT/'engineering/packet',OUT/'packet'):
 meta=json.loads((packet/'metadata.json').read_text());assert meta['source_commit']==COMMIT
 for name,digest in meta['source_sha256'].items():
  blob=subprocess.check_output(['git','cat-file','blob',COMMIT+':'+name],cwd=REPO)
  assert hashlib.sha256(blob).hexdigest()==digest==sha(packet/'sources'/name)==sha(REPO/name),name
 parents=json.loads((packet/'parents.json').read_text());assert {p['seed'] for p in parents}=={68101,68105}
 details=[]
 for p in parents:
  file=packet/p['path'];assert sha(file)==p['sha256']
  payload=torch.load(file,map_location='cpu',weights_only=True)
  assert payload['training_info']['init_seed']==p['seed'] and payload['training_info'].get('task') is None
  assert payload['config']==p['config']
  assert (p['config']['segment_len'],p['config']['memory_segment_len'],p['config']['memory_batch_size'])==(32,8,32)
  assert payload['config']['num_tokens']==256
  tensors=payload['model_state'];assert all(isinstance(t,torch.Tensor) and torch.isfinite(t).all() for t in tensors.values())
  h=state_sha(tensors);assert h==p['state_sha256']
  count=sum(t.numel() for t in tensors.values());assert count==839248,count
  before=parents_seen.setdefault(p['seed'],(h,p['sha256']));assert before==(h,p['sha256'])
  details.append({'seed':p['seed'],'tensor_elements':count,'state_sha256':h,'checkpoint_sha256':p['sha256'],'task_unadapted':True})
 results.append({'packet':str(packet),'source_files_exact_git_blobs':len(meta['source_sha256']),'parents':details,'design_hash_bound':sha(packet/'design.json')==meta['files_sha256']['design.json']})
test=REPO/'tests/test_clmm_key_binding.py';assert test.read_bytes()==subprocess.check_output(['git','cat-file','blob',COMMIT+':tests/test_clmm_key_binding.py'],cwd=REPO)
log=REPO.parent/'tests.log';assert log.is_file();testlog=log.read_text(encoding='utf8',errors='replace');assert '109 passed' in testlog
protected=pathlib.Path(r'C:\Users\henry\Documents\Codex\clmm-replay\docs\SLEEP_MULTIREVISION.md');assert sha(protected)=='d8d69a9c6872ae86859bf6952814d06cc8d7b43f841f5993d5be1ef30335f26c'
result={'status':'passed','kind':'git_blob_source_and_tensor_only_parent_inspection','source_commit':COMMIT,'models_constructed':False,'model_inference':False,'torch':torch.__version__,'packets':results,'test_source_git_exact':True,'reported_scoped_tests_passed':109,'tests_log_sha256':sha(log),'protected_user_doc_sha256':sha(protected)}
(OUT/'analysis/frozen_provenance_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
