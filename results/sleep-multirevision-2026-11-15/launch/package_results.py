"""Prepare compact Git evidence and a complete, verified recovery archive."""
import argparse,csv,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
WORK=Path(__file__).resolve().parent;BASE=WORK.parents[1]
ROOT=BASE/'outputs/clmm-sleep-multirevision-2026-11-15'
REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
DEST=REPO/'results/sleep-multirevision-2026-11-15'
SOURCE='06c0a35b543c25437659765128b81718e5bfa1c5'
BASELINE='35a549f137548d48aecc26e33c6f8a34a6efc97c'
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
 assert read(ROOT/'validation/coordinator_check.json')['status']=='passed'
 assert read(ROOT/'validation.json')['status']=='passed'
 assert read(ROOT/'cpu-recheck/verification.json')['status']=='passed'
 assert read(ROOT/'analysis/packet_audit.json')['status']=='passed'
 assert len(list((ROOT/'runs').glob('*/metadata.json')))==12
 for p in WORK.glob('*.py'):copy(p,ROOT/'launch'/p.name)
 copy(WORK/'tests.log',ROOT/'validation/tests.log')
 copy(WORK/'pipeline.log',ROOT/'pipeline.log')
 copy(WORK/'cpu-recheck.log',ROOT/'cpu-recheck/execution.log')
 copy(WORK/'coordinator-check.log',ROOT/'validation/coordinator-check.log')
 copy(WORK/'cluster-recheck.log',ROOT/'cluster/controller.log')
 copy(WORK/'portable-proof.tar.gz',ROOT/'cluster/portable-proof-upload.tar.gz')
 copy(BASE/'outputs/clmm-revision-local-2026-11-15/trained/metadata.json',ROOT/'provenance/checkpoint_training_metadata.json')
 for name in ('SLEEP_MULTIREVISION.md','HANDOFF.md','MEMBER_C_STATUS.md','EXPERIMENT_LOG.jsonl'):
  copy(REPO/'docs'/name,ROOT/name)
 for p in ROOT.iterdir():
  if p.is_file() and p.suffix in ('.json','.log','.txt'):copy(p,DEST/p.name)
 for name in ('analysis','validation','launch','cpu-recheck','gpu-recheck','provenance','cluster'):
  for p in (ROOT/name).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.npy','.npz','.gz','.bundle','.zip'):copy(p,DEST/name/p.relative_to(ROOT/name))
 for name in ('metadata.json','models.json','design.json','exclusion_manifest.json'):
  copy(ROOT/'packet'/name,DEST/'packet'/name)
 for folder in (ROOT/'runs').iterdir():
  copy(folder/'metadata.json',DEST/'runs'/folder.name/'metadata.json')
  for name in ('summary.csv','resources.csv'):copy(folder/'evaluation'/name,DEST/'runs'/folder.name/'evaluation'/name)
 (DEST/'.gitattributes').write_text('* -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n*.svg -diff -merge -text\n',encoding='utf8')
 (DEST/'.gitignore').write_text('!*.log\n',encoding='utf8')
 (DEST/'README.md').write_text('''# K16 multiple-revision replay timing

Read [results guide](../../docs/SLEEP_MULTIREVISION.md), [handoff](../../docs/HANDOFF.md), and analysis/summary.json. Twelve existing local-delta checkpoints, both parents, content plus all five random maps, no additional training. Continuous/periodic samples and extra writes match exactly; only application timing changes. Keep admission failures and the distinction between stale-at-application and any obsolete final query version.

This compact Git index omits raw scores/predictions/plans, excluded signatures, full source snapshots and weights. Full root: C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-sleep-multirevision-2026-11-15. Adjacent CLMM_Sleep_Multirevision_2026-11-15.zip includes complete raw data, all twelve checkpoints, frozen source, engineering runs, launch scripts and the full branch bundle. Recover with the bundle, check out executable commit06c0a35b543c25437659765128b81718e5bfa1c5 with core.autocrlf=false, and relocate paths in recorded commands. No original team Google Doc or Git remote was modified.

To reproduce inference, use the archived packet directly with scripts/sleep_multirevision_experiment.py evaluate --out NEW_FOLDER --packet PACKET --model NAME. For data-only reconstruction, call clmm.sleep.multirevision.make_datasets(91123,128,set(json.load(open(PACKET/excluded_signatures.json)))); the exact captured JSON bytes/hash can be regenerated with the project's json_file writer. Dataset and observer audit scripts are model-independent. Source snapshots and packet hashes must match the executable source, regardless of later documentation commits.

Reference component evaluation only: no full Titans/MAC qualification or biological novelty claim. Content/random differences also reflect separately learned representations; within-checkpoint replay timing remains the controlled contrast.
''',encoding='utf8')
 print('Prepared compact index',flush=True)
def package():
 assert not git('status','--porcelain')
 assert git('rev-parse','baseline-v0.6.0^{}')==BASELINE
 head=git('rev-parse','HEAD');bundle=ROOT/'clmm-sleep-multirevision.bundle'
 subprocess.run(['git','bundle','create',str(bundle),'sleep-k16-multirevision','baseline-v0.6.0'],cwd=REPO,check=True)
 subprocess.run(['git','bundle','verify',str(bundle)],cwd=REPO,check=True)
 write(ROOT/'package-provenance.json',dict(source_commit=SOURCE,handoff_commit=head,baseline_tag=BASELINE,bundle_sha256=sha(bundle),excludes=['__pycache__','*.pyc']))
 files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256' and '__pycache__' not in p.parts and p.suffix!='.pyc')
 manifest=ROOT/'MANIFEST.sha256';manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files),encoding='utf8')
 archive=ROOT.parent/'CLMM_Sleep_Multirevision_2026-11-15.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files+[manifest]:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for p in files:
   h=hashlib.sha256()
   with z.open(ROOT.name+'/'+p.relative_to(ROOT).as_posix()) as f:
    for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
   assert h.hexdigest()==sha(p)
 result=dict(status='passed',archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1,sha256=sha(archive),handoff_commit=head,every_file_sha_and_crc_verified=True)
 write(ROOT.parent/'CLMM_Sleep_Multirevision_2026-11-15.verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','package']);a=p.parse_args();(prepare if a.mode=='prepare' else package)()
