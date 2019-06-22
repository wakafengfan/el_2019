import json
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from configuration.config import data_dir
from configuration.match import match2


def freq():
    id2kb = {}
    for l in tqdm((Path(data_dir) / 'kb_data').open()):
        _ = json.loads(l)
        subject_id = _['subject_id']
        subject_alias = list(set([_['subject']] + _.get('alias', [])))
        subject_alias = [sa.lower() for sa in subject_alias]
        subject_desc = ''
        for i in _['data']:
            if '摘要' in i['predicate']:
                subject_desc = i['object']
                break
            else:
                subject_desc += f'{i["predicate"]}:{i["object"]}\n'

        subject_desc = subject_desc[:300].lower()
        if subject_desc:
            id2kb[subject_id] = {'subject_alias': subject_alias, 'subject_desc': subject_desc}

    kb2id = defaultdict(list)  # subject: [sid1, sid2,...]
    for i, j in tqdm(id2kb.items()):
        for k in j['subject_alias']:
            kb2id[k].append(i)

    train_data = (Path(data_dir) / 'train.json').open()
    kb_ac = pickle.load((Path(data_dir) / 'kb_ac.pkl').open('rb'))
    freq_dic = defaultdict(dict)
    cnt = 0
    for i, l in tqdm(enumerate(train_data)):
        # if i > 20000:
        #     break
        l = json.loads(l)
        t = l['text']
        exp_words = [(k,sidx) for k,sidx,_ in match2(t)]
        labeled_words = [(m['mention'],m['offset']) for m in l['mention_data'] if m['kb_id']!='NIL']
        if set(exp_words).issuperset(set(labeled_words)):
            cnt+= 1

        for w, start_idx in exp_words:
            if 'exp' not in freq_dic[w]:
                freq_dic[w]['exp'] = 1
            else:
                freq_dic[w]['exp'] += 1
            if (w,start_idx) in labeled_words:
                if 'labeled' not in freq_dic[w]:
                    freq_dic[w]['labeled'] = 1
                else:
                    freq_dic[w]['labeled'] += 1


    print(f'cnt: {cnt}')

    for w in freq_dic:
        if 'labeled' not in freq_dic[w]:
            freq_dic[w]['labeled'] = 0
        freq_dic[w]['per'] = freq_dic[w]['labeled']/freq_dic[w]['exp']
    a = [(w, freq_dic[w]['per']) for w in freq_dic]
    a = sorted(a, key=lambda x: x[1], reverse=True)

    p = (Path(data_dir)/'el_freq_dic.json').open('w')
    for w in freq_dic:
        doc = freq_dic[w]
        doc.update({'word': w})
        s = json.dumps(doc)
        p.write(s + '\n')

if __name__ == '__main__':
    freq()
