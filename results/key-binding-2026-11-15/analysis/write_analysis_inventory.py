"""Write the final analysis-only inventory after all reviewer-owned artifacts."""
import datetime,hashlib,json,pathlib
A=pathlib.Path(__file__).resolve().parent;target=A/'analysis_inventory.json';entries=[]
for path in sorted(A.rglob('*')):
 if path.is_file() and path!=target and '__pycache__' not in path.parts:
  entries.append(dict(path=path.relative_to(A).as_posix(),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
protected=pathlib.Path(r'C:\Users\henry\Documents\Codex\clmm-replay\docs\SLEEP_MULTIREVISION.md');h=hashlib.sha256(protected.read_bytes()).hexdigest();assert h=='d8d69a9c6872ae86859bf6952814d06cc8d7b43f841f5993d5be1ef30335f26c'
result=dict(schema_version=1,status='complete_analysis_snapshot',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(A),file_count=len(entries),excluded=['analysis_inventory.json','__pycache__/**'],scope='Reviewer-owned analysis files only; coordinator may continue changing artifacts outside analysis',protected_user_document_sha256=h,entries=entries)
target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
assert all((A/e['path']).stat().st_size==e['bytes'] and hashlib.sha256((A/e['path']).read_bytes()).hexdigest()==e['sha256'] for e in entries)
print(json.dumps(dict(status='passed',files=len(entries),inventory_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),future_union_sha256=hashlib.sha256((A/'future_exclusion_signatures.json').read_bytes()).hexdigest(),review_sha256=hashlib.sha256((A/'independent_results_review.md').read_bytes()).hexdigest()),indent=2))
