import json,sys
from datetime import datetime,timezone
from pathlib import Path
repo=Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
event=dict(time_utc=datetime.now(timezone.utc).isoformat(),event='sleep_multirevision_started',branch='sleep-k16-multirevision',source_commit='06c0a35b543c25437659765128b81718e5bfa1c5',checkpoint_source='c9f5c153f786f1b82ac383d01c1d18c6c595946b',models=12,new_optimizer_steps=0,main_seed=91123,histories=128,K=16,genuine_changes_per_revised_key=3,phases=[0,2],tails=[0,64,128],arms=['online','repeat','continuous','periodic'],paired_literal_samples=True,fast_state_bytes=65536,tests_passed=359,preflight_concurrent_seconds=33.5010014,padded_projection_seconds=2010.060084,main_limit_seconds=2700,pipeline_limit_seconds=3300,main_workers=2,torch_threads_per_worker=1,scope='Fixed checkpoint reference component replay timing',output_root='C:/Users/henry/Documents/Codex/2026-10-22/i/outputs/clmm-sleep-multirevision-2026-11-15',cluster_jobs_created=0)
with (repo/'docs/EXPERIMENT_LOG.jsonl').open('a',encoding='utf8') as handle:handle.write(json.dumps(event,separators=(',',':'))+'\n')
