"""Compact Git evidence index and complete verified recovery ZIP."""
import argparse,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-key-binding-2026-11-15'
REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay');DEST=REPO/'results/key-binding-2026-11-15'
SOURCE='5e8c61829ff85ecea3c9314d4ccd8dfef3268b68';BASELINE='35a549f137548d48aecc26e33c6f8a34a6efc97c'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()

def prepare():
    assert read(ROOT/'gpu-main/result/pipeline.json')['status']=='succeeded'
    assert read(ROOT/'validation.json')['status'] in ('complete','complete_with_documented_limitations')
    assert read(ROOT/'validation/coordinator-check.json')['totals']['score_vectors']==81920
    assert read(ROOT/'cpu-recheck/verification.json')['status'] in ('passed','numerical_review_required')
    for p in WORK.glob('*.py'):copy(p,ROOT/'launch'/p.name)
    for p in WORK.glob('*.log'):copy(p,ROOT/'launch/logs'/p.name)
    for name,target in [('protocol.bundle','executed-source.bundle'),('preexisting_worktree.json','preexisting_worktree.json'),('preexisting-user.patch','preexisting-user.patch')]:
        copy(WORK/name,ROOT/'provenance'/target)
    for name in ('KEY_BINDING.md','HANDOFF.md','MEMBER_C_STATUS.md','EXPERIMENT_LOG.jsonl'):copy(REPO/'docs'/name,ROOT/name)
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.json','.md','.jsonl','.txt'):copy(p,DEST/p.name)
    omitted=('.pyc','.npy','.npz','.gz','.bundle','.zip','.pt','.patch')
    for folder in ('analysis','validation','cpu-recheck','cluster','cluster-v2','launch','provenance'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in omitted and not p.name.endswith('signatures.json') and p.name!='selected_rows.json':
                copy(p,DEST/folder/p.relative_to(ROOT/folder))
    for tree in ('gpu-main','gpu-output'):
        base=ROOT/tree/'result';copy(base/'pipeline.json',DEST/tree/'result/pipeline.json')
        for section in ('initial','runs','engineering'):
            if not (base/section).exists():continue
            for folder in (base/section).iterdir():
                for name in ('metadata.json','training.csv','evaluation/summary.csv','evaluation/resources.csv'):
                    if (folder/name).exists():copy(folder/name,DEST/tree/'result'/section/folder.name/name)
    for name in ('metadata.json','parents.json','key_codec.json','design.json','exclusion_manifest.json'):copy(ROOT/'packet'/name,DEST/'packet'/name)
    (DEST/'.gitattributes').write_text('* -text whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol\n*.svg -diff -merge -text\n',encoding='utf8')
    (DEST/'.gitignore').write_text('!*.log\n',encoding='utf8')
    (DEST/'README.md').write_text('''# Fixed-size key binding diagnostic

Read [study](../../docs/KEY_BINDING.md), [Member C status](../../docs/MEMBER_C_STATUS.md), [handoff](../../docs/HANDOFF.md) and analysis/independent_results_review.md. The unchanged839,248-registered-parameter byte model is independently trained from two original language parents with digit or atomic key representations, wide128 attention only. Four fixed terminal2000-step models and all four task-unadapted initial conditions are retained. Each adaptation gets32,000answer forks/2,908,928input bytes. Original+value-swap joint-four correctness and six acquisition gates prevent interpreting value-copy shortcuts as binding.

The original one-GPU runtime gate stopped before all main efficacy; its preflight-only output is gpu-output, resource records cluster. The separately hash-bound execution-amendment.json records the one explicit two-GPU replacement, with unchanged scientific data/source/updates/endpoints and its own runtime gate/hard cap. Main results are gpu-main/result, resource records cluster-v2. No automatic renewal or further resource escalation. Source5e8c61829ff85ecea3c9314d4ccd8dfef3268b68 is frozen; later documentation HEAD is not its strict packet Git identity.

This compact Git index omits full weights/optimizers, raw predictions/logits, full prepared worlds/arrays/source snapshots, signature lists and the pre-existing user document patch. Full root:C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-key-binding-2026-11-15. Adjacent CLMM_Key_Binding_2026-11-15.zip contains these inputs, both cluster attempts, CPU smoke/recheck, source and branch bundles, audits and figures. An adjacent verification file binds final commit, every artifact SHA256 and archive CRC. The original user edit is preserved outside our commits.

Recover from the bundle with core.autocrlf=false at exact executed5e8c618, relocate paths, then run scripts/key_binding_experiment.py initial/train --packet PACKET --out NEW_FOLDER --parent 68101 --encoding atomic --device cuda:0. Code validates source/data/protocol/parent hashes. Preparation uses the archived prior signature union plus archived smoke/engineering histories and actual new training worlds; reruns must keep those exclusions. Analysis/README.md records independent audit commands. Codec bytes must be used directly, never UTF8-decoded/re-encoded.

This is a supplied-key encoding bundle, including changed segmentation/token identity/prior exposure, not a pure tokenizer mechanism claim. Static codec metadata is separate from learned model parameters. Normal/reset parity within95<128 is not memory-independence evidence; retrieval-zero preserves writes while changing downstream activations. No cross-session memory, retention/sleep/Hebbian efficacy, full paper-Titans or biological novelty claim follows. Current digit-only SequenceReplay does not accept atomic records. Existing qualified typed-memory controls remain available for A/B. Current histories are consulted development data.
''',encoding='utf8')
    print('Prepared compact index',flush=True)

def package():
    prior=read(WORK/'preexisting_worktree.json')
    assert sha(REPO/prior['path'])==prior['sha256']
    assert git('status','--porcelain')=='M docs/SLEEP_MULTIREVISION.md'
    assert git('rev-parse','baseline-v0.6.0^{}')==BASELINE
    head=git('rev-parse','HEAD');bundle=ROOT/'clmm-key-binding.bundle'
    subprocess.run(['git','bundle','create',str(bundle),'sequence-key-binding','baseline-v0.6.0'],cwd=REPO,check=True)
    subprocess.run(['git','bundle','verify',str(bundle)],cwd=REPO,check=True)
    write(ROOT/'package-provenance.json',dict(source_commit=SOURCE,handoff_commit=head,baseline_tag=BASELINE,bundle_sha256=sha(bundle),preserved_preexisting_user_edit=prior,our_changes_committed=True))
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256' and '__pycache__' not in p.parts and p.suffix!='.pyc')
    manifest=ROOT/'MANIFEST.sha256';manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files),encoding='utf8')
    archive=ROOT.parent/'CLMM_Key_Binding_2026-11-15.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[manifest]:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in files:
            h=hashlib.sha256()
            with z.open(ROOT.name+'/'+p.relative_to(ROOT).as_posix()) as f:
                for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
            assert h.hexdigest()==sha(p)
    result=dict(status='passed',archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1,sha256=sha(archive),handoff_commit=head,every_artifact_sha_and_crc_verified=True,user_edit_preserved=True)
    write(ROOT.parent/'CLMM_Key_Binding_2026-11-15.verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','package']);a=p.parse_args();(prepare if a.mode=='prepare' else package)()
