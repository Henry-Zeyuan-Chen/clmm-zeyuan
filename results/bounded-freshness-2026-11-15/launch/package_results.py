"""Compact evidence index plus complete checksum-verified recovery archive."""
import argparse,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-bounded-freshness-2026-11-15'
REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay');DEST=REPO/'results/bounded-freshness-2026-11-15'
SOURCE='9ab3901860548d2d87a9afc61e8ce58f950a03ce';BASELINE='35a549f137548d48aecc26e33c6f8a34a6efc97c'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
def prepare():
 assert read(ROOT/'pipeline.json')['status']=='succeeded'
 assert read(ROOT/'validation.json')['status']=='passed_with_documented_launch_coverage_omission'
 assert read(ROOT/'cpu-recheck/verification.json')['status']=='passed'
 assert read(ROOT/'analysis/packet_audit.json')['status']=='passed'
 for p in WORK.glob('*.py'):copy(p,ROOT/'launch'/p.name)
 for name,target in [('tests.log','validation/tests.log'),('pipeline.log','pipeline.log'),('cpu-recheck.log','cpu-recheck/execution.log'),('coordinator-check.log','validation/coordinator-check.log')]:copy(WORK/name,ROOT/target)
 copy(WORK/'protocol.bundle',ROOT/'provenance/executable-source.bundle')
 copy(ROOT.parent/'clmm-revision-local-2026-11-15/trained/metadata.json',ROOT/'provenance/checkpoint_training_metadata.json')
 for name in ('BOUNDED_FRESHNESS.md','HANDOFF.md','MEMBER_C_STATUS.md','EXPERIMENT_LOG.jsonl'):copy(REPO/'docs'/name,ROOT/name)
 for p in ROOT.iterdir():
  if p.is_file() and p.suffix in ('.json','.log','.txt','.md'):copy(p,DEST/p.name)
 for name in ('analysis','validation','launch','cpu-recheck','provenance','session-example'):
  for p in (ROOT/name).rglob('*'):
   if p.is_file() and p.name!='audited_signature_superset.json' and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.npy','.npz','.gz','.bundle','.zip','.pt','.patch'):copy(p,DEST/name/p.relative_to(ROOT/name))
 for name in ('metadata.json','models.json','design.json','exclusion_manifest.json'):copy(ROOT/'packet'/name,DEST/'packet'/name)
 for folder in (ROOT/'runs').iterdir():
  copy(folder/'metadata.json',DEST/'runs'/folder.name/'metadata.json')
  for name in ('summary.csv','resources.csv'):copy(folder/'evaluation'/name,DEST/'runs'/folder.name/'evaluation'/name)
 (DEST/'.gitattributes').write_text('* -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n*.svg -diff -merge -text\n',encoding='utf8')
 (DEST/'.gitignore').write_text('!*.log\n',encoding='utf8')
 (DEST/'README.md').write_text('''# Bounded freshness versus literal replay

Read [study/next steps](../../docs/BOUNDED_FRESHNESS.md), [handoff](../../docs/HANDOFF.md), model_size_decision.md, analysis/summary.json and analysis/result_interpretation.md. All twelve original local-delta checkpoints and five random maps per parent are retained, with no additional training. Four replay policies match frozen candidate draws/key order/writes/FIFO8+snapshot4/64KiB traces. Bounded actual application source/value can differ by schedule; it is not an identical-content timing comparison. Literal v1 exports remain compatible; bounded v2 exports identify freshness.

This compact Git index omits full weights, source snapshots, excluded signature lists and raw score/prediction/plan arrays. Full root: C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-bounded-freshness-2026-11-15. Adjacent CLMM_Bounded_Freshness_2026-11-15.zip contains those inputs, engineering cases, analysis code, source snapshots, twelve exact checkpoints and full branch bundle. It also preserves the pre-existing user document patch outside Git; that edit was not included in our commits.

Recover with the bundle, check out executable commit9ab3901860548d2d87a9afc61e8ce58f950a03ce with core.autocrlf=false, then relocate the recorded paths. Run scripts/freshness_experiment.py evaluate --out NEW_FOLDER --packet PACKET --model NAME. The runner validates prepared Git/source/data/model hashes. Regenerate data with clmm.sleep.multirevision.make_datasets(91223,128,set(json.load(open(PACKET/excluded_signatures.json)))) and the project's JSON writer. Full historical exclusion-provenance re-audits additionally need prior original inputs, as stated in analysis/README.md; frozen exclusions and saved data suffice for current inference and raw audits.

These are fixed-codebook component results, not a full Titans/MAC or biological novelty claim. Model-size reasoning distinguishes the81,921-parameter component from the existing839,248-parameter complete sequence approximation. The current split becomes consulted development data after this study.
''',encoding='utf8')
 print('Prepared compact index',flush=True)
def package():
 prior=read(WORK/'preexisting_worktree.json')
 assert sha(REPO/prior['path'])==prior['sha256']
 assert git('status','--porcelain')=='M docs/SLEEP_MULTIREVISION.md'
 assert git('rev-parse','baseline-v0.6.0^{}')==BASELINE
 head=git('rev-parse','HEAD');bundle=ROOT/'clmm-bounded-freshness.bundle'
 subprocess.run(['git','bundle','create',str(bundle),'sleep-bounded-freshness','baseline-v0.6.0'],cwd=REPO,check=True)
 subprocess.run(['git','bundle','verify',str(bundle)],cwd=REPO,check=True)
 write(ROOT/'package-provenance.json',dict(source_commit=SOURCE,handoff_commit=head,baseline_tag=BASELINE,bundle_sha256=sha(bundle),preserved_preexisting_user_edit=prior,our_tracked_changes_committed=True))
 files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256' and '__pycache__' not in p.parts and p.suffix!='.pyc')
 manifest=ROOT/'MANIFEST.sha256';manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files),encoding='utf8')
 archive=ROOT.parent/'CLMM_Bounded_Freshness_2026-11-15.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files+[manifest]:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for p in files:
   h=hashlib.sha256()
   with z.open(ROOT.name+'/'+p.relative_to(ROOT).as_posix()) as f:
    for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
   assert h.hexdigest()==sha(p)
 result=dict(status='passed',archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1,sha256=sha(archive),handoff_commit=head,every_artifact_sha_and_crc_verified=True,user_edit_preserved=True)
 write(ROOT.parent/'CLMM_Bounded_Freshness_2026-11-15.verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','package']);a=p.parse_args();(prepare if a.mode=='prepare' else package)()
