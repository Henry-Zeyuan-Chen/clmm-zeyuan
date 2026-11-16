import ast,hashlib,json,pathlib,math
out=pathlib.Path(r'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-key-binding-2026-11-15');work=out.parents[1]/'work/clmm-key-binding'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
amend=json.loads((out/'execution-amendment.json').read_text());pipeline=json.loads((out/'gpu-output/result/pipeline.json').read_text())
assert pipeline['status']=='preflight_only' and pipeline['initial']==[] and pipeline['cases']==[] and not (out/'gpu-output/result/initial').exists() and not (out/'gpu-output/result/runs').exists()
assert 'main_started_epoch' not in pipeline
metas=list((out/'gpu-output/result').rglob('metadata.json'));assert len(metas)==2
for p in metas:
 m=json.loads(p.read_text());assert m['mode']=='preflight' and m['completed_steps']==20 and m['status']=='succeeded'
 r=next(r for r in pipeline['preflight'] if r['encoding']==m['encoding'])
 assert m['training_elapsed_seconds']==r['training_seconds'] and m['evaluation']['elapsed_seconds']==r['evaluation_seconds']
 assert m['checkpoint_sha256']==sha(p.parent/'checkpoint.pt')
t=max(r['training_seconds'] for r in pipeline['preflight']);e=max(r['evaluation_seconds'] for r in pipeline['preflight']);projection=1.25*(4*100*t+8*16*e)
assert projection==pipeline['projected_main_seconds']==amend['original_projection'] and projection>1800 and not pipeline['runtime_gate']
assert amend['new_threshold_seconds']==2300 and amend['new_pipeline_hard_seconds']==2400 and amend['max_total_allocations']==2 and amend['no_further_resource_escalation']
assert sha(out/'packet/metadata.json')==amend['main_packet_metadata_sha256']
for packet in (out/'packet',out/'engineering/packet'):
 m=json.loads((packet/'metadata.json').read_text());assert m['source_commit']=='5e8c61829ff85ecea3c9314d4ccd8dfef3268b68'
 for name,h in m['files_sha256'].items():assert sha(packet/name)==h
 for name,h in m['source_sha256'].items():assert sha(packet/'sources'/name)==h
launcher=(work/'cluster_parallel.py').read_text();remote=(work/'remote_parallel.py').read_text();ast.parse(launcher);ast.parse(remote)
assert "ROOT=WORK.parents[1]/'outputs/clmm-key-binding-2026-11-15'" in launcher
assert "ROOT/'gpu-main/result/pipeline.json'" in launcher and "ROOT/'gpu-output/result/pipeline.json'" in launcher
assert 'input-upload-parallel.tar.gz' in launcher and 'result-download-parallel.tar.gz' in launcher
assert 'main_packet_metadata_sha256' in launcher and 'execution_amendment_sha256' in launcher and 'execution_amendment_sha256' in remote
assert "original['status']=='preflight_only'" in launcher and "not original['initial']" in launcher and "not original['cases']" in launcher
assert '--gres=gpu:A6000:2' in launcher and '--time=00:45:00' in launcher and '-c8 --mem=16G' in launcher
assert 'LIMIT=2400' in remote and 'projection<=2300' in remote and 'max_workers=2' in remote and "enumerate((68101,68105))" in remote
assert '4*100*train+8*16*evaluation)/2' in remote
scheduler=(out/'cluster/scheduler-final.txt').read_text();assert '13926582|zc2819|CANCELLED' in scheduler
closure=json.loads((out/'cluster/closure.json').read_text());assert closure['job_closed'] and closure['job_id']==13926582 and closure['allocation_elapsed_seconds']==148 and closure['no_active_user_jobs_at_confirmation']
assert '13926582|zc2819|CANCELLED' in (out/'cluster/accounting-confirmed.txt').read_text()
result={'status':'passed','kind':'prospective_execution_amendment_audit','models_constructed':False,'model_inference':False,'source_commit':'5e8c61829ff85ecea3c9314d4ccd8dfef3268b68','old_pipeline_status':pipeline['status'],'old_main_updates':0,'old_main_evaluation_cases':0,'old_preflight_cases':2,'training_calibration_max_seconds':t,'evaluation_calibration_max_seconds':e,'original_padded_projection_seconds':projection,'original_gate_seconds':1800,'projection_halved_seconds':projection/2,'new_gate_seconds':2300,'new_hard_pipeline_seconds':2400,'new_allocation':'one45min2A6000/8CPU/16GiB;2parentworkers;max2totalallocations','scientific_packet_hashes_unchanged':True,'main_packet_metadata_sha256':sha(out/'packet/metadata.json'),'execution_amendment_sha256':sha(out/'execution-amendment.json'),'launcher_sha256':sha(work/'cluster_parallel.py'),'remote_pipeline_sha256':sha(work/'remote_parallel.py'),'path_and_preservation_guards_verified':True,'syntax_ast_passed':True,'old_scheduler_evidence':'Later accounting-confirmed.txt: CANCELLED148seconds/1GPU; closure.json records no active user jobs at confirmation', 'closure_sha256':sha(out/'cluster/closure.json'),'note':'Halved old calibration is a feasibility calculation only; replacement pipeline must independently pass new parallel runtime gate before any main case. Original one-GPU protocol was amended explicitly; no architecture/data/selection change is implied.'}
(out/'analysis/execution_amendment_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))

