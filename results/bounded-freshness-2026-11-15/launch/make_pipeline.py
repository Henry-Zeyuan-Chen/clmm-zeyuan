from pathlib import Path
work=Path(__file__).resolve().parent
source=(work.parent/'clmm-sleep-multirevision/pipeline.py').read_text(encoding='utf8')
source=source.replace('clmm-sleep-multirevision-2026-11-15','clmm-bounded-freshness-2026-11-15').replace('sleep_multirevision_experiment.py','freshness_experiment.py')
source=source.replace('LIMIT=3300','LIMIT=4200').replace('91103','91203').replace('91123','91223').replace('<=2700','<=3600')
source=source.replace("'--max-seconds',1200","'--max-seconds',1500")
(work/'pipeline.py').write_text(source,encoding='utf8')
