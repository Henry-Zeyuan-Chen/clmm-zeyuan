"""Bind final analysis code, tables and reviewed figures after completion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
target = root/'final_artifact_inventory.json'
files = {}
for path in sorted(root.rglob('*')):
    if path.is_file() and path != target and '__pycache__' not in path.parts:
        files[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size,
                                                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), files=files,
              source_commit='06c0a35b543c25437659765128b81718e5bfa1c5',
              model_source_commit='c9f5c153f786f1b82ac383d01c1d18c6c595946b',
              scope='Independent analysis code and saved-output arithmetic; full inference proofs owned by coordinator',
              all_twelve_cases_retained=True, numerical_figures=5, figure_formats=['png', 'pdf', 'svg'])
target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(files=len(files), inventory=str(target))))
