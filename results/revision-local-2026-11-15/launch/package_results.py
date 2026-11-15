"""Compact Git index and complete verified revision-study recovery package."""
import argparse,csv,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
ROOT=Path(r'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-revision-local-2026-11-15');REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay');WORK=Path(__file__).parent;DEST=REPO/'results/revision-local-2026-11-15'
SOURCE='c9f5c153f786f1b82ac383d01c1d18c6c595946b';BASELINE='35a549f137548d48aecc26e33c6f8a34a6efc97c'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO,text=True).strip()
def prepare():
 assert read(ROOT/'validation.json')['status']=='passed'
 assert read(ROOT/'training-pipeline.json')['status']=='succeeded'
 assert read(ROOT/'stress/metadata.json')['status']=='succeeded'
 assert read(ROOT/'validation/stress_coverage.json')['status']=='passed'
 registry=[]
 for phase in ('frozen','trained'):
  with (ROOT/'analysis'/phase/'final_gates.csv').open() as h:gates=list(csv.DictReader(h))
  initialgates=[]
  if phase=='trained':
   with (ROOT/'analysis'/phase/'initial_gates.csv').open() as h:initialgates=list(csv.DictReader(h))
  for group in (gates,initialgates):
   for row in group:
    for key,value in list(row.items()):
     if value in ('True','False'):row[key]=(value=='True')
  meta=read(ROOT/phase/'metadata.json')
  for case in meta['cases']:
   folder=ROOT/phase/'runs'/case['name']
   for stage in (('final',) if phase=='frozen' else ('initial','final')):
    path=folder/('initial_checkpoint.pt' if stage=='initial' else 'checkpoint.pt')
    own=[g for g in (initialgates if stage=='initial' else gates) if g['rule']==case['revision_rule'] and int(g['parent_seed'])==case['parent_seed'] and g['route']==('content' if case['policy']=='content' else f"random-{case['random_seed']}")]
    assert len(own)==1
    registry.append(dict(phase=phase,case=case['name'],stage=stage,checkpoint=path.relative_to(ROOT).as_posix(),sha256=sha(path),source_commit=SOURCE,config=case['config'],parent='parents/'+case['parent'],parent_sha256=case['parent_sha256'],slow_sha256=case['initial_slow_sha256'] if stage=='initial' else case['slow_sha256'],additional_training_steps=0 if phase=='frozen' or stage=='initial' else 1200,parent_training_steps=case['parent_training']['step'],own_admission=own[0],full_sequence_model_qualified=False,teammate_final_method=False,scope='Reference component; pair fast state with exact slowSHA, not configfingerprint alone'))
 write(ROOT/'checkpoint_admission.json',dict(scope='48frozen +24untrained +24terminaltrained weightsets; not a best-model selection',registry=registry))
 copy(ROOT.parent/'clmm-routing-decay-2026-11-15/main/episodes.json',ROOT/'provenance/prior90723/episodes.json')
 for p in WORK.glob('*.py'):copy(p,ROOT/'launch'/p.name)
 for name in ('smoke-frozen-cpu','smoke-train-cpu','smoke-exclusions-cpu'):
  shutil.copytree(WORK/name,ROOT/'engineering-cpu'/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for p in WORK.glob('smoke*.log'):copy(p,ROOT/'engineering-cpu'/p.name)
 for name in ('REVISION_LOCAL.md','HANDOFF.md','MEMBER_C_STATUS.md','EXPERIMENT_LOG.jsonl'):copy(REPO/'docs'/name,ROOT/name)
 for p in ROOT.iterdir():
  if p.is_file() and p.suffix in ('.json','.log','.txt'):copy(p,DEST/p.name)
 for name in ('analysis','validation','launch','gpu-recheck','cpu-recheck','stress-cpu-recheck'):
  for p in (ROOT/name).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':copy(p,DEST/name/p.relative_to(ROOT/name))
 for phase in ('frozen','trained','engineering-gpu'):
  copy(ROOT/phase/'metadata.json',DEST/phase/'metadata.json')
  for folder in (ROOT/phase/'runs').iterdir():
   for name in ('case.json','training.csv'):
    if (folder/name).is_file():copy(folder/name,DEST/phase/'runs'/folder.name/name)
   for stage in ('initial-evaluation','evaluation'):
    for name in ('summary.csv','resources.csv'):
     if (folder/stage/name).is_file():copy(folder/stage/name,DEST/phase/'runs'/folder.name/stage/name)
 copy(ROOT/'stress/metadata.json',DEST/'stress/metadata.json')
 copy(ROOT/'stress/construction.json',DEST/'stress/construction.json')
 copy(ROOT/'stress/driver_snapshot.py',DEST/'stress/driver_snapshot.py')
 for folder in (ROOT/'stress/runs').iterdir():
  copy(folder/'case.json',DEST/'stress/runs'/folder.name/'case.json')
  for name in ('summary.csv','resources.csv'):copy(folder/'evaluation'/name,DEST/'stress/runs'/folder.name/'evaluation'/name)
 (DEST/'.gitattributes').write_text('* -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n*.svg -diff -merge -text\n');(DEST/'.gitignore').write_text('!*.log\n')
 (DEST/'README.md').write_text("""# Key-local revision reference study

Read [results guide](../../docs/REVISION_LOCAL.md), [handoff](../../docs/HANDOFF.md), [checkpoint/admission registry](checkpoint_admission.json), and phase analysis under analysis/frozen and analysis/trained.

This compact Git index omits full weights/optimizer states, raw scores/predictions/plans and source snapshots. Full root is C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-revision-local-2026-11-15; adjacent CLMM_Revision_Local_2026-11-15.zip contains the complete recoverable study and branch bundle.

48 frozen factorial conditions isolate key-local erasure and active bulk decay.24 matched training runs start from exact paired untrained tensors, with1200steps perrun and initial/final evaluations. Both routing families and all5fixedmaps are retained. No best-checkpoint/map selection. This is a delta-rule-inspired component control, not a new biological rule or full Titans model. Scientific success and artifact validation are distinct.
The stress/ index is a fresh post-outcome reinforced-old-value study using all24terminal checkpoints without training. It preserves the original primary failure. Prior90723 input is packaged under provenance/prior90723 for relocation of the stress --prior argument; full raw data and exact driver snapshot are in the recovery ZIP.
""")
 print('Prepared compact revision index.')
def package():
 assert not git('status','--porcelain');assert git('rev-parse','baseline-v0.6.0^{}')==BASELINE;head=git('rev-parse','HEAD');bundle=ROOT/'clmm-revision-local.bundle'
 subprocess.run(['git','bundle','create',str(bundle),'hebbian-revision-local','baseline-v0.6.0'],cwd=REPO,check=True);subprocess.run(['git','bundle','verify',str(bundle)],cwd=REPO,check=True)
 write(ROOT/'package-provenance.json',dict(source_commit=SOURCE,handoff_commit=head,baseline_tag=BASELINE,bundle_sha256=sha(bundle),excluded=['source.bundle','__pycache__','*.pyc']))
 files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in ('source.bundle','MANIFEST.sha256') and '__pycache__' not in p.parts and p.suffix!='.pyc');manifest=ROOT/'MANIFEST.sha256';manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files));archive=ROOT.parent/'CLMM_Revision_Local_2026-11-15.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files+[manifest]:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for p in files:assert hashlib.sha256(z.read(ROOT.name+'/'+p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
 result=dict(status='passed',archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1,sha256=sha(archive),handoff_commit=head,every_file_sha_and_crc_verified=True);write(ROOT.parent/'CLMM_Revision_Local_2026-11-15.verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','package'));a=p.parse_args();(prepare if a.mode=='prepare' else package)()
