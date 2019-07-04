import json
import random
from pathlib import Path
import numpy as np

from configuration.config import data_dir

for fn in (Path(data_dir) / 'pretrain_data').iterdir():
    if 'metric' in fn.name:
        continue
    right_list = []
    wrong_list = []
    for l in Path(fn).open():
        l_ = json.loads(l)
        if l_['is_random_next']:
            wrong_list.append(l)
        else:
            right_list.append(l)

    right_size = len(right_list)
    wrong_size = len(wrong_list)
    print(f'right_size: {right_size}')
    print(f'wrong_size: {wrong_size}')

    assert wrong_size > right_size * 3
    wrong_list = random.sample(wrong_list, k=right_size * 3)
    total_list = right_list + wrong_list

    np.random.shuffle(total_list)
    v2_path = (Path(data_dir) / 'pretrain_data_v2' / fn.name.replace('.json', '_v2.json')).open('w')
    for d in total_list:
        v2_path.write(d + '\n')
    metrics_filename = (Path(data_dir) / 'pretrain_data_v2' / fn.name.replace('.json', '_metrics.json')).open('w')
    metrics_filename.write(json.dumps({'num_training_examples': len(total_list)}))
    print(f'num_training_examples: {len(total_list)}')
