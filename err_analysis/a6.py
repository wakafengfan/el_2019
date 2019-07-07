import collections
import json
from collections import defaultdict
from pathlib import Path

from configuration.config import data_dir


no_recall = defaultdict(lambda : 0)
recall_no_precision = defaultdict(int)
for l in json.load((Path(data_dir)/'err_log__[el_pt_subject.py].json').open())['err']:
    T = [i[0] for i in l['mention_data']]
    P = [i[0] for i in l['predict']]

    for x in T:
        if x not in P:
            no_recall[x] += 1

    for x in P:
        if x not in T:
            recall_no_precision[x] += 1

a = collections.Counter(no_recall).most_common()
b = collections.Counter(recall_no_precision).most_common()

freq = json.load((Path(data_dir)/'el_freq_dic_1.json').open())
a1 = [x for x in a if x[0] in freq and freq[x[0]]['exp']<5 and freq[x[0]]['per']==0.5]

print('Done')