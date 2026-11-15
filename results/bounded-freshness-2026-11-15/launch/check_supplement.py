import hashlib,json,shutil,sys
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-bounded-freshness-2026-11-15'
sys.path.insert(0,str(WORK/'source-checkout'))
from scripts.freshness_experiment import collect_histories
from clmm.sleep.multirevision import make_datasets
report=json.loads((ROOT/'analysis/supplemental_exclusion_audit.json').read_text());extra=set()
for row in report['omitted_sources']:
 path=Path(row['path']);raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256']
 ids=collect_histories(json.loads(raw));assert ids==set(row['omitted_signatures']);extra|=ids
 shutil.copy2(path,ROOT/'provenance'/f'supplemental-{path.parent.name}-verification.json')
assert len(extra)==1
regenerated=[]
for packet,seed,n in [(ROOT/'packet',91223,128),(ROOT/'engineering/packet',91203,16)]:
 seen=set(json.loads((packet/'excluded_signatures.json').read_text()))
 data=json.loads((packet/'episodes.json').read_text());fresh=make_datasets(seed,n,seen|extra)
 assert fresh==data and not collect_histories(data).intersection(extra)
 lengths=sorted({len(ep['keys']) for rows in data.values() for ep in rows});assert 26 not in lengths
 regenerated.append(dict(seed=seed,histories_per_cell=n,literal_histories=sum(map(len,data.values())),full_dataset_including_labels_equal=True,lengths=lengths,overlap_with_supplement=0))
(ROOT/'validation').mkdir(exist_ok=True)
(ROOT/'validation/supplemental_crosscheck.json').write_text(json.dumps(dict(status='passed',scope='Actual producer regeneration with expanded exclusions, independent of reviewer generator; exact full dataset including query metadata',original_packets_modified=False,extra_signatures=sorted(extra),checks=regenerated),indent=2)+'\n',encoding='utf8')
print('Supplemental inputs hashed; both full datasets exactly regenerated without changes.')
