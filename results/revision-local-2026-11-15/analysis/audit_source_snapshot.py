"""Verify exact source file bytes against a declared recoverable Git snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--commit',required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()==a.commit
assert subprocess.check_output(['git','status','--porcelain'],cwd=a.repo,text=True)==''
files=subprocess.check_output(['git','ls-tree','-r','--name-only',a.commit],cwd=a.repo,text=True).splitlines()
checked={}
for name in files:
    if not name.endswith('.py') and name not in ('pyproject.toml','requirements-lock.txt'):
        continue
    blob=subprocess.check_output(['git','show',f'{a.commit}:{name}'],cwd=a.repo)
    actual=(a.repo/name).read_bytes()
    assert actual==blob,name
    checked[name]=hashlib.sha256(blob).hexdigest()
report=dict(status='passed',commit=a.commit,repo=str(a.repo),exact_git_blob_matches=len(checked),files=checked)
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='files'}))
