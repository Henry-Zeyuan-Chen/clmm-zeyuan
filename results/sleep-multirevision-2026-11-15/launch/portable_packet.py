"""Create small, byte-linked expected subset for a separate-device recheck."""
import csv,hashlib,json,shutil,tarfile
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-sleep-multirevision-2026-11-15'
OUT=ROOT/'portable-proof';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
selected=lambda r:int(r['episode'])<16 and ((r['cell']=='gate_k16_r3' and r['arm']=='online') or (r['cell']=='pressure_k16_p2_d128' and r['arm'] in ('online','continuous','periodic')))
records=[]
for model in json.loads((ROOT/'packet/models.json').read_text()):
 source=ROOT/'runs'/model['name']/'evaluation';dest=OUT/'expected'/model['name'];dest.mkdir(parents=True)
 with (source/'predictions.csv').open() as f:allrows=list(csv.DictReader(f))
 indices=[i for i,row in enumerate(allrows) if selected(row)];rows=[allrows[i] for i in indices]
 with (dest/'predictions.csv').open('w',newline='',encoding='utf8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 scores=np.load(source/'scores.npy',mmap_mode='r');np.save(dest/'scores.npy',scores[indices])
 plans=[r for r in json.loads((source/'plans.json').read_text()) if selected(r)];(dest/'plans.json').write_text(json.dumps(plans),encoding='utf8')
 records.append(dict(case=model['name'],queries=len(indices),plans=len(plans),source_files={n:sha(source/n) for n in ('predictions.csv','scores.npy','plans.json')},subset_files={n:sha(dest/n) for n in ('predictions.csv','scores.npy','plans.json')},row_indices=indices))
manifest=dict(status='prepared',selection='First16 histories: neutral gate online; phase2D128 online/continuous/periodic; no selection by outcome',cases=records)
(OUT/'selection.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
archive=WORK/'portable-proof.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in (ROOT/'packet').rglob('*'):
  if p.is_file() and p.suffix not in ('.pyc','.npz') and '__pycache__' not in p.parts:tar.add(p,arcname='root/packet/'+p.relative_to(ROOT/'packet').as_posix())
 for p in OUT.rglob('*'):
  if p.is_file():tar.add(p,arcname='root/portable-proof/'+p.relative_to(OUT).as_posix())
 tar.add(WORK/'protocol.bundle',arcname='protocol.bundle');tar.add(WORK/'recheck.py',arcname='recheck.py')
print(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size)))
