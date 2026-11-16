"""Record versions/hardware without credentials, inside the actual allocation."""
import importlib.metadata,json,os,platform,sys,time
print('PY_START',flush=True);started=time.monotonic()
import torch
count=torch.cuda.device_count()
data=dict(python=sys.version,platform=platform.platform(),torch=str(torch.__version__),cuda=torch.version.cuda,
          cuda_devices=count,device_names=[torch.cuda.get_device_name(i) for i in range(count)],
          slurm_job=os.environ.get('SLURM_JOB_ID'),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
          packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
          import_seconds=time.monotonic()-started)
print(json.dumps(data,indent=2),flush=True)
assert count==int(sys.argv[1])
