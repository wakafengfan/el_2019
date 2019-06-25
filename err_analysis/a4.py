import json
from pathlib import Path

from configuration.config import data_dir

for l in (Path(data_dir)/'train.json').open():
    l = json.loads(l)
