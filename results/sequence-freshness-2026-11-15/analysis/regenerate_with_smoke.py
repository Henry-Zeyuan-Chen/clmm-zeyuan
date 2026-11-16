"""Regenerate unchanged fixed evaluation using actual frozen producer and wider exclusions."""
import hashlib,importlib.util,json,sys,types
from pathlib import Path
from audit_packet import read_json,sha
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent.parent/'work/clmm-sequence-freshness/source-checkout'
if not SOURCE.exists():SOURCE=Path('C:/Users/henry/Documents/Codex/2026-10-22/i/work/clmm-sequence-freshness/source-checkout')
packet=ROOT/'packet-final';meta=read_json(packet/'metadata.json')
package=types.ModuleType('audited_frozen_producer');package.__path__=[str(SOURCE/'clmm/pretrain')];sys.modules[package.__name__]=package
source_hashes={}
for name in ['fact_streams','kv_scheduler','sequence_replay','sequence_freshness_data']:
    path=SOURCE/f'clmm/pretrain/{name}.py';relative=path.relative_to(SOURCE).as_posix();source_hashes[relative]=sha(path)
    assert source_hashes[relative]==meta['source_sha256'][relative]
    spec=importlib.util.spec_from_file_location(f'{package.__name__}.{name}',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
expanded_path=ROOT/'analysis/work_smoke_expanded_signatures.json';expanded=set(read_json(expanded_path))
producer=sys.modules[f'{package.__name__}.sequence_freshness_data']
regenerated=producer.evaluation_worlds(expanded,meta['development_seed'],meta['episodes'],meta['revision_episodes'])
frozen=read_json(packet/'episodes.json');assert regenerated==frozen
result=dict(status='passed',scope='actual frozen producer data regeneration only; no model construction/inference',
    source_commit=meta['source_commit'],source_sha256=source_hashes,expanded_signatures=len(expanded),expanded_signatures_sha256=sha(expanded_path),
    episode_objects_identical=True,groups={name:len(episodes) for name,episodes in regenerated.items()},
    frozen_episode_file_sha256=sha(packet/'episodes.json'))
(ROOT/'analysis/work_smoke_regeneration_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
