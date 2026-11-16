"""Compact Git evidence and a full hash/CRC-verified sequence recovery archive."""
import argparse,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]/'outputs/clmm-sequence-freshness-2026-11-15'
REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
DEST=REPO/'results/sequence-freshness-2026-11-15'
SOURCE='95da59531ccf5d5f3fe2729553e6d980da598352'
BASELINE='35a549f137548d48aecc26e33c6f8a34a6efc97c'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()

def prepare():
    assert read(ROOT/'gpu-output/result/pipeline.json')['status']=='succeeded'
    assert read(ROOT/'validation.json')['status']=='complete_with_documented_limitations'
    assert read(ROOT/'cpu-recheck/verification.json')['status'] in ('passed','numerical_review_required')
    assert read(ROOT/'analysis/audited/results_audit.json')['total_logit_vectors']==52224
    assert read(ROOT/'cluster/metadata.json')['cancel_requested']
    for p in WORK.glob('*.py'):copy(p,ROOT/'launch'/p.name)
    for p in WORK.glob('*.log'):copy(p,ROOT/'launch/logs'/p.name)
    for name,target in [('protocol.bundle','original-protocol.bundle'),('protocol-executed.bundle','executed-source.bundle'),
                        ('preexisting-user.patch','preexisting-user.patch'),('preexisting_worktree.json','preexisting_worktree.json')]:
        copy(WORK/name,ROOT/'provenance'/target)
    for name in ('SEQUENCE_FRESHNESS.md','HANDOFF.md','MEMBER_C_STATUS.md','EXPERIMENT_LOG.jsonl'):
        copy(REPO/'docs'/name,ROOT/name)
    excluded_suffixes=('.pyc','.npy','.npz','.gz','.bundle','.zip','.pt','.patch')
    for folder in ('analysis','validation','launch','cpu-recheck','provenance','cluster'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in excluded_suffixes and not p.name.endswith('signatures.json') and p.name!='selected_rows.json':
                copy(p,DEST/folder/p.relative_to(ROOT/folder))
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.json','.md','.jsonl','.txt'):copy(p,DEST/p.name)
    for section in ('initial','runs','engineering'):
        for folder in (ROOT/'gpu-output/result'/section).iterdir():
            for name in ('metadata.json','training.csv','evaluation/summary.csv','evaluation/resources.csv'):
                p=folder/name
                if p.exists():copy(p,DEST/'gpu-output/result'/section/folder.name/name)
    copy(ROOT/'gpu-output/result/pipeline.json',DEST/'gpu-output/result/pipeline.json')
    for name in ('metadata.json','parents.json','design.json','exclusion_manifest.json'):
        copy(ROOT/'packet-final'/name,DEST/'packet-final'/name)
    (DEST/'.gitattributes').write_text('* -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n*.svg -diff -merge -text\n',encoding='utf8')
    (DEST/'.gitignore').write_text('!*.log\n',encoding='utf8')
    (DEST/'README.md').write_text('''# Complete-sequence bounded freshness acquisition pilot

Read [study and interpretation](../../docs/SEQUENCE_FRESHNESS.md), [Member C status](../../docs/MEMBER_C_STATUS.md) and [handoff](../../docs/HANDOFF.md). The original 839,248-registered-parameter byte-MAC approximation stays unchanged in size. Eight fixed adaptations cross two language parents, native32/wide128 attention training and literal/bounded periodic record replay. All terminal step2000 checkpoints are retained; midpoint1000 is recovery-only. Matched per-case budget:32,000answer forks/3,646,208input bytes.

This compact index includes training traces, aggregate and episode analysis, figures, source/data/provenance audits and launch scripts. It omits full checkpoints/optimizers, raw256-way predictions/logits, complete prepared arrays/worlds/source snapshots, original exclusion lists and the pre-existing user document patch. Full root: C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-sequence-freshness-2026-11-15. Adjacent CLMM_Sequence_Freshness_2026-11-15.zip preserves every input, original/amended/failed preparation, CPU smoke, source bundle, run, audit and full branch history with a SHA256 manifest and verified archive CRC.

Recover a source checkout from the bundle using core.autocrlf=false and exact executed commit95da59531ccf5d5f3fe2729553e6d980da598352. Relocate packet/model paths in commands. scripts/sequence_freshness_experiment.py initial/train --packet PACKET_FINAL --out NEW_FOLDER --parent 68101 --segment 32 --arm periodic_bounded executes the frozen protocol. The runner checks exact source/data/parent hashes. Launch scripts own bounded cluster coordination; analysis/README.md describes independent audits. Historical exclusion-manifest rediscovery needs original prior artifacts; the supplied frozen union suffices for current data regeneration/inference. The later81347union includes the archived CPU smoke; current consulted evaluation must also be excluded for future studies.

Replay retains exact observed32-byte records, FIFO8/snapshot4, <=384payload bytes at rest plus metadata (temporary416B). It traverses the whole transformer. The provided parser recognizes the heldout grammars; neural inputs retain raw wording. This is a sequence replay reference, not memory-only sleep, a biological method or full paper-faithful Titans. Co-primary acquisition probes have no replay extras and estimate adaptation spillover. Failed acquisition prevents retention/sleep efficacy claims. No size increase is part of this study.
''',encoding='utf8')
    print('Prepared compact index',flush=True)

def package():
    prior=read(WORK/'preexisting_worktree.json')
    assert sha(REPO/prior['path'])==prior['sha256']
    assert git('status','--porcelain')=='M docs/SLEEP_MULTIREVISION.md'
    assert git('rev-parse','baseline-v0.6.0^{}')==BASELINE
    head=git('rev-parse','HEAD');bundle=ROOT/'clmm-sequence-freshness.bundle'
    subprocess.run(['git','bundle','create',str(bundle),'sequence-bounded-freshness','baseline-v0.6.0'],cwd=REPO,check=True)
    subprocess.run(['git','bundle','verify',str(bundle)],cwd=REPO,check=True)
    write(ROOT/'package-provenance.json',dict(source_commit=SOURCE,handoff_commit=head,baseline_tag=BASELINE,
          bundle_sha256=sha(bundle),preserved_preexisting_user_edit=prior,our_tracked_changes_committed=True))
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256' and '__pycache__' not in p.parts and p.suffix!='.pyc')
    manifest=ROOT/'MANIFEST.sha256'
    manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files),encoding='utf8')
    archive=ROOT.parent/'CLMM_Sequence_Freshness_2026-11-15.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[manifest]:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in files:
            h=hashlib.sha256()
            with z.open(ROOT.name+'/'+p.relative_to(ROOT).as_posix()) as f:
                for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==sha(p)
    result=dict(status='passed',archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1,
                sha256=sha(archive),handoff_commit=head,every_artifact_sha_and_crc_verified=True,user_edit_preserved=True)
    write(ROOT.parent/'CLMM_Sequence_Freshness_2026-11-15.verification.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','package']);args=p.parse_args()
    (prepare if args.mode=='prepare' else package)()
