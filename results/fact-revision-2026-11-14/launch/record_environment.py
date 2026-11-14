import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
a.out.write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),
    packages={d.metadata['Name']:d.version for d in metadata.distributions()}),indent=2,sort_keys=True)+'\n',encoding='utf8')
