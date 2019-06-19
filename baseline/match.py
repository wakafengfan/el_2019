import json
import pickle
import re
from pathlib import Path

import ahocorasick
from tqdm import tqdm

from configuration.config import data_dir

# load kb_data, get all entity,build ac
# kb_ac = ahocorasick.Automaton()
#
# for l in (Path(data_dir)/'kb_data').open():
#     _ = json.loads(l)
#     subject_id = _['subject_id']
#     subject_alias = list(set([_['subject']] + _.get('alias', [])))
#     subject_alias = [a.lower() for a in subject_alias]
#     for a in subject_alias:
#         kb_ac.add_word(a, (a, subject_id))
#
# kb_ac.make_automaton()
# pickle.dump(kb_ac, (Path(data_dir)/'kb_ac.pkl').open('wb'))
kb_ac = pickle.load((Path(data_dir)/'kb_ac.pkl').open('rb'))


# 补余匹配出实体
def match2(text):
    r = []

    # 补余方式1
    for s in re.findall(r'《([^《》]*?)》', text):
        r.append((s, str(text.index(s)), str(text.index(s)+len(s))))
    r_ = [_[0] for _ in r]

    # 补余方式2
    words = []
    i = 0
    while i < len(text):
        j = i+1
        word = ''
        while j < len(text):
            w = text[i:j]
            if all(w not in rr for rr in r_) and kb_ac.exists(w) and len(w) > len(word):
                word = w
            j += 1
        if len(word)>2:
            words.append((i,word))
            i += len(word)
        else:
            i += 1

    for start_idx, w in words:
        _, sid = kb_ac.get(w)
        r.append((w, str(start_idx), str(start_idx+len(w))))

    return list(set(r))

def tst_recall():
    train_data = (Path(data_dir)/'train.json').open()
    exact,R,T = 0,0,0

    for l in tqdm(train_data):
        l = json.loads(l)
        t = l['text']
        m = set([(d['mention'],d['offset']) for d in l['mention_data'] if d['kb_id']!='NIL'])
        rs = set(match2(t))
        if m == rs:
            exact += 1

        else:
            print(f'{t}\n{m}\n{rs}\n')
        R += len(m & rs)
        T += len(m)

    print(f'total exact:{exact/90000}')
    print(f'recall rate: {R/T}')


if __name__ == '__main__':
    tst_recall()


