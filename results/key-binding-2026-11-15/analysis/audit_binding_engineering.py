import pathlib,runpy,json,torch
out=pathlib.Path(r'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-key-binding-2026-11-15');torch.set_num_threads(1)
a=runpy.run_path(str(out/'analysis/audit_binding_outputs.py'));t=runpy.run_path(str(out/'analysis/audit_binding_tensors.py'))
cases=[(out/'engineering-cpu/run',out/'engineering-cpu/packet')]
for phase in ('gpu-output','gpu-main'):
 for enc in ('digits','atomic'):cases.append((out/phase/'result/engineering'/enc,out/'engineering/packet'))
records=[]
for path,packet in cases:
 pm=a['read'](packet/'metadata.json');r,j=a['audit_case'](path,packet,pm,a['read'](packet/'samples.json'),pm['episodes'])
 for key in ('metrics','joint_four','admission_pass','admission_threshold'):r.pop(key,None)
 m=a['read'](path/'metadata.json');parent=next(p for p in a['read'](packet/'parents.json') if p['seed']==m['parent_seed']);pp=t['load'](packet/parent['path']);config=dict(pp['config']);config['segment_len']=128
 payload,h=t['check_state'](path/'checkpoint.pt',config,pp['model_state']);assert h==m['final_state_sha256']
 assert payload['source_info']['source_commit']==pm['source_commit'] and payload['source_info']['source_sha256']==pm['source_sha256']
 r['optimizer']=t['optimizer'](path/'optimizer.pt',m['completed_steps'],payload['model_state']);r['tensor_state_sha256']=h;r['scope']='engineering validity only; no efficacy selection';records.append(r)
result={'status':'passed','kind':'separate_saved_engineering_raw_vector_tensor_optimizer_validity','models_constructed':False,'model_inference':False,'gpu_calibration_vectors':sum(r['vectors'] for r in records[1:]),'cpu_smoke_vectors':records[0]['vectors'],'total_vectors':sum(r['vectors'] for r in records),'records':records}
(out/'analysis/engineering_validity_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
