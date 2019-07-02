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


"""
log loss
nil_list size: 575, wrong_list size: 1220, right_list size: 3420
nil
(array([133,  60,  44,  33,  27,  23,  36,  40,  34, 145]), array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]))
wrong
(array([117, 103,  72,  74,  97, 100,  92,  95, 147, 323]), array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]))
right
(array([  44,   33,   30,   30,   50,   67,   80,  126,  228, 2732]), array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]))


mse loss
nil_list size: 575, wrong_list size: 1276, right_list size: 3512
nil
(array([145,  55,  47,  45,  33,  39,  32,  29,  46, 104]), array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]))
wrong
(array([ 82,  91,  66,  79,  89,  97,  80,  97, 150, 445]), array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]))
right
(array([  32,   35,   21,   30,   52,   48,   58,   95,  169, 2972]), array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]))

mse loss
nil_list size: 575, wrong_list size: 1156, right_list size: 3297
nil
(array([158,  69,  48,  48,  38,  17,  29,  29,  32, 107]), array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]))
wrong
(array([ 96,  88,  73,  73,  58,  75,  82,  91, 147, 373]), array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]))
right
(array([  16,   34,   25,   36,   39,   43,   50,   85,  226, 2743]), array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]))

"""


