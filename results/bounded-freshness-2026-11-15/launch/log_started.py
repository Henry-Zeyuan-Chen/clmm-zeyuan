import json,shutil
from datetime import datetime,timezone
from pathlib import Path
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]/'outputs/clmm-bounded-freshness-2026-11-15';REPO=Path(r'C:\Users\henry\Documents\Codex\clmm-replay')
(ROOT/'provenance').mkdir(exist_ok=True)
for name in ('preexisting-user.patch','preexisting_worktree.json'):shutil.copy2(WORK/name,ROOT/'provenance'/name)
event=dict(time_utc=datetime.now(timezone.utc).isoformat(),event='bounded_freshness_started',branch='sleep-bounded-freshness',source_commit='9ab3901860548d2d87a9afc61e8ce58f950a03ce',models=12,tests_passed=417,new_optimizer_steps=0,main_seed=91223,histories=128,engineering_seed=91203,arms=['online','repeat','continuous','periodic','continuous_bounded','periodic_bounded'],matched='frozen candidate draws/key order/writes/FIFO8+snapshot4/64KiB fast traces',bounded_application_content_can_differ=True,pipeline_limit_seconds=4200,main_runtime_threshold_seconds=3600,workers=2,torch_threads=1,output_root=ROOT.as_posix(),preexisting_user_edit_preserved='docs/SLEEP_MULTIREVISION.md',model_size_decision='Fix81921component parameters now; existing839248parametersequencepath requires held-out acquisition before enlarging',new_cluster_allocations=0)
with (REPO/'docs/EXPERIMENT_LOG.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(event,separators=(',',':'))+'\n')
