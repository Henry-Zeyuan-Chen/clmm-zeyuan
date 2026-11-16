"""Synthetic integrity checks for packet reviewer, without model imports."""
import copy,gzip,importlib.util,json,sys,types
from pathlib import Path
import numpy as np
from audit_packet import audit_packet,context_signature,render_record
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;REPO=Path('C:/Users/henry/Documents/Codex/clmm-replay')
package=types.ModuleType('packet_fixture_source');package.__path__=[str(REPO/'clmm/pretrain')];sys.modules[package.__name__]=package
for name in ['fact_streams','kv_scheduler','sequence_replay','sequence_freshness_data']:
    spec=importlib.util.spec_from_file_location(f'{package.__name__}.{name}',REPO/f'clmm/pretrain/{name}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
source=sys.modules[f'{package.__name__}.sequence_freshness_data']
fixture=ROOT/'synthetic_packet_fixture';fixture.mkdir(exist_ok=True)
worlds=source.training_worlds(20,2,91302);arrays=source.training_arrays(worlds)
excluded={source.signature(ep['events']) for world in worlds for ep in world['episodes']}
groups=source.evaluation_worlds(excluded,n=4,revision_n=2)
def dump(path,value):path.write_text(json.dumps(value),encoding='utf8')
dump(fixture/'worlds.json',worlds);np.savez_compressed(fixture/'arrays.npz',**arrays)
dump(fixture/'evaluation.json',groups);dump(fixture/'exclusions.json',[])
paths=[fixture/'worlds.json',fixture/'arrays.npz',fixture/'evaluation.json',fixture/'exclusions.json']
passed=audit_packet(*paths)
checks=['small real-producer packet independently reconstructs all records/targets/plans']
def reject(name,arguments):
    try:audit_packet(*arguments)
    except (AssertionError,KeyError,ValueError):checks.append(name);return
    raise AssertionError('Corruption was accepted: '+name)
bad={key:value.copy() for key,value in arrays.items()};bad['periodic'][0,0,0]^=1
np.savez_compressed(fixture/'bad-input.npz',**bad);reject('altered training byte rejected',[paths[0],fixture/'bad-input.npz',*paths[2:]])
bad={key:value.copy() for key,value in arrays.items()};bad['targets'][0,0]^=1
np.savez_compressed(fixture/'bad-target.npz',**bad);reject('altered target rejected',[paths[0],fixture/'bad-target.npz',*paths[2:]])
bad=copy.deepcopy(groups);bad['acquire_train_seen'][0]['queries'][0]['target']='z'
dump(fixture/'bad-evaluation.json',bad);reject('query target inconsistent with real history rejected',[*paths[:2],fixture/'bad-evaluation.json',paths[3]])
bad=copy.deepcopy(groups);bad['acquire_dev_heldout'][0]['events'][0]['key']='999'
dump(fixture/'bad-pairing.json',bad);reject('unpaired heldout semantics rejected',[*paths[:2],fixture/'bad-pairing.json',paths[3]])
identifier=context_signature([render_record(r) for r in groups['acquire_train_seen'][0]['events']])
dump(fixture/'bad-exclusions.json',[identifier]);reject('consulted-history collision rejected',[*paths[:3],fixture/'bad-exclusions.json'])
result=dict(status='passed',checks=checks,fixture_result=passed,scope='synthetic evidence-auditor checks only; no efficacy output')
(ROOT/'packet_auditor_selfchecks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status='passed',checks=checks)))
