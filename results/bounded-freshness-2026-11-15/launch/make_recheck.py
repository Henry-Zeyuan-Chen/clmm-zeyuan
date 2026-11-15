from pathlib import Path
work=Path(__file__).resolve().parent
source=(work.parent/'clmm-sleep-multirevision/recheck.py').read_text(encoding='utf8')
source=source.replace('from clmm.sleep import SleepSession','from clmm.sleep import SleepSession\nfrom scripts.freshness_experiment import session_factory,ARMS')
source=source.replace("('pressure_k16_p2_d128',['online','continuous','periodic'])","('pressure_k16_p2_d128',ARMS)")
source=source.replace("session=SleepSession(model,16,arm,[ep['seed'] for ep in batch])","session=session_factory(model,16,arm,[ep['seed'] for ep in batch])")
source=source.replace('online/continuous/periodic','all six pressure arms')
(work/'recheck.py').write_text(source,encoding='utf8')
