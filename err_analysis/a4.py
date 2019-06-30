import json
from pathlib import Path
import numpy as np
from configuration.config import data_dir

err = json.load((Path(data_dir)/'err_log/err_log__[el_pt_object.py].json').open())['err']
nil_list, wrong_list, right_list = [],[],[]
for e in err:
    M = sorted(e['mention_data'], key=lambda x:x[0])
    P = sorted(e['predict'], key=lambda x: x[0])

    for m, p in zip(M, P):
        assert m[0] == p[0]

        if m[2] == 'NIL':
            nil_list.append(float(p[3]))
        elif m[2] != p[2]:
            wrong_list.append(float(p[3]))
        elif m[2] == p[2]:
            right_list.append(float(p[3]))
        else:
            pass


print(f'nil_list size: {len(nil_list)}, wrong_list size: {len(wrong_list)}, right_list size: {len(right_list)}')

bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

print('nil')
print(np.histogram(nil_list, bins=bins))

print('wrong')
print(np.histogram(wrong_list, bins=bins))

print('right')
print(np.histogram(right_list, bins=bins))





