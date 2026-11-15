"""Compact review index and complete SHA-verified recovery package."""
import argparse,csv,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
ROOT=Path(r'C:\Users\henry\Documents\Codex\2026-10-22\i\outputs\clmm-routing-decay-2026-11-15')
REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
WORK=Path(__file__).parent;DEST=REPO/'results/routing-decay-2026-11-15'
SOURCE='61212289d6029da65eeb6a3735ce5ec3894866a9'
BASELINE='35a549f137548d48aecc26e33c6f8a34a6efc97c'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO,text=True).strip()
def prepare():
 assert read(ROOT/'pipeline-cpu.json')['status']=='succeeded'
 assert read(ROOT/'analysis/raw_artifact_audit.json')['fully_audited_complete_main']
 assert read(ROOT/'validation.json')['status']=='passed'
 assert read(ROOT/'gpu-recheck/verification.json')['verified_models']==48
 assert read(ROOT/'cpu-recheck/verification.json')['verified_models']==48
 assert 'RESULTS_PENDING' not in (REPO/'docs/ROUTING_DECAY.md').read_text()
 with (ROOT/'analysis/pressure_parent_gates.csv').open() as h:gates=list(csv.DictReader(h))
 registry=[]
 for case in read(ROOT/'main/metadata.json')['cases']:
  folder=ROOT/'main/runs'/case['name'];checkpoint=folder/'checkpoint.pt'
  registry.append(dict(case=case['name'],checkpoint=checkpoint.relative_to(ROOT).as_posix(),sha256=sha(checkpoint),
    source_commit=SOURCE,config=case['config'],slow_sha256=case['slow_sha256'],parent_sha256=case['parent_sha256'],
    parent='parents/'+case['parent'],route_table=case['route_table'],own_admission=[g for g in gates if g['case']==case['name']],
    offline_training=False,full_sequence_model_qualified=False,teammate_final_method=False,
    construction='Dense trained parent; identical learned tensors; new routing/decay config and zero H. No optimizer or adaptation.',
    state_binding='Fast fingerprint binds policy/decay/codebook/table; additionally keep exact slow checkpoint SHA.'))
 write(ROOT/'checkpoint_admission.json',dict(scope='48 frozen transfer conditions, not new training or selected final systems',registry=registry))
 for p in WORK.glob('*.py'):copy(p,ROOT/'launch'/p.name)
 for name in ('ROUTING_DECAY.md','HANDOFF.md','MEMBER_C_STATUS.md','EXPERIMENT_LOG.jsonl'):copy(REPO/'docs'/name,ROOT/name)
 for p in ROOT.iterdir():
  if p.is_file() and p.suffix in ('.json','.log','.txt'):copy(p,DEST/p.name)
 for folder in ('analysis','validation','launch','gpu-recheck','cpu-recheck'):
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':copy(p,DEST/folder/p.relative_to(ROOT/folder))
 for name in ('metadata.json',):copy(ROOT/'main'/name,DEST/'main'/name)
 for folder in (ROOT/'main/runs').iterdir():
  copy(folder/'case.json',DEST/'main/runs'/folder.name/'case.json')
  for name in ('summary.csv','resources.csv'):copy(folder/'evaluation'/name,DEST/'main/runs'/folder.name/'evaluation'/name)
 for folder in ('engineering','engineering-cpu'):
  copy(ROOT/folder/'metadata.json',DEST/folder/'metadata.json')
  for p in (ROOT/folder/'runs').rglob('summary.csv'):copy(p,DEST/folder/p.relative_to(ROOT/folder))
 (DEST/'.gitattributes').write_text('* -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n')
 (DEST/'.gitignore').write_text('!*.log\n')
 (DEST/'README.md').write_text("""# Frozen routing and inactive-decay diagnostic

Read [results and reproduction guide](../../docs/ROUTING_DECAY.md), [handoff](../../docs/HANDOFF.md), [paired contrasts](analysis/contrasts.csv), [raw audit](analysis/raw_artifact_audit.json), and [checkpoint admission](checkpoint_admission.json).

This compact Git index omits the 48 checkpoints, complete raw predictions/scores, observed plans and source snapshots. Full root is `C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-routing-decay-2026-11-15`; adjacent `CLMM_Routing_Decay_2026-11-15.zip` includes the recoverable branch bundle and every main/engineering artifact.

Four previously dense-trained parents are frozen. One content map and five preselected fixed random maps cross protected/global inactive decay. Main uses CPU after a prospective runtime-only amendment; selected GPU/CPU inference checks cover all48 conditions. No new training or final model selection. These reference components are not full Titans or final teammate methods.
""")
 print('Prepared compact index.')
def package():
 assert not git('status','--porcelain')
 assert git('rev-parse','baseline-v0.6.0^{}')==BASELINE
 head=git('rev-parse','HEAD');bundle=ROOT/'clmm-routing-decay.bundle'
 subprocess.run(['git','bundle','create',str(bundle),'routing-decay-controls','baseline-v0.6.0'],cwd=REPO,check=True)
 subprocess.run(['git','bundle','verify',str(bundle)],cwd=REPO,check=True)
 write(ROOT/'package-provenance.json',dict(source_commit=SOURCE,handoff_commit=head,baseline_tag=BASELINE,bundle_sha256=sha(bundle),excluded=['source.bundle','__pycache__','*.pyc']))
 files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in ('source.bundle','MANIFEST.sha256') and '__pycache__' not in p.parts and p.suffix!='.pyc')
 manifest=ROOT/'MANIFEST.sha256';manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files))
 archive=ROOT.parent/'CLMM_Routing_Decay_2026-11-15.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files+[manifest]:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for p in files:assert hashlib.sha256(z.read(ROOT.name+'/'+p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
 result=dict(status='passed',archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1,sha256=sha(archive),handoff_commit=head,every_file_sha_and_crc_verified=True)
 write(ROOT.parent/'CLMM_Routing_Decay_2026-11-15.verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','package'));a=p.parse_args();(prepare if a.mode=='prepare' else package)()
