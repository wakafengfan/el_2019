import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import ahocorasick
import jieba
from tqdm import tqdm

from configuration.config import data_dir, expire_list

# load kb_data, get all entity,build ac
# kb_ac = ahocorasick.Automaton()
#
# for l in tqdm((Path(data_dir)/'kb_data').open()):
#     _ = json.loads(l)
#     subject_id = _['subject_id']
#     subject_alias = list(set([_['subject']] + _.get('alias', [])))
#     subject_alias = [a.lower() for a in subject_alias]
#     for a in subject_alias:
#         kb_ac.add_word(a, (a, subject_id))
#
# kb_ac.make_automaton()
# pickle.dump(kb_ac, (Path(data_dir)/'kb_ac.pkl').open('wb'))
kb_ac = pickle.load((Path(data_dir) / 'kb_ac.pkl').open('rb'))


def match_rules(w):
    # 长度大于1个字
    if len(w) < 2:
        return False
    # 数字或英文
    if re.match(r'^\d+$', w) or re.match(r'^[a-zA-Z\s]+$', w):
        return False
    # xx年 xx月
    if re.match(r'^\d+[年月]$', w):
        return False

    # 在排除list
    if w in expire_list:
        return False

    return True


# 补余匹配出实体
def match2(text):
    r = []

    # 补余方式1
    # for s in re.findall(r'《([^《》]*?)》', text):
    #     if kb_ac.exists(s):
    #         r.append((s, str(text.index(s)), str(text.index(s) + len(s))))
    # r_ = [_[0] for _ in r]

    # 补余方式2
    words = []
    i = 0
    while i < len(text):
        j = i + 1
        word = ''
        while j <= len(text):
            w = text[i:j]
            # if all(w not in rr for rr in r_) and kb_ac.exists(w) and len(w) > len(word):
            if kb_ac.exists(w) and len(w) > len(word) and w not in ['《']:
                word = w
            j += 1
        if match_rules(word):
            words.append((i, word))
            i += len(word)
        else:
            i += 1

    for start_idx, w in words:
        _, sid = kb_ac.get(w)
        r.append((w, str(start_idx), str(start_idx + len(w))))

    return list(set(r))


def tst_recall():
    train_data = (Path(data_dir) / 'train.json').open()
    exact, R,A, B = 0, 0, 0,0

    for l in tqdm(train_data):
        l = json.loads(l)
        t = l['text']
        m = set([(str(d['mention']), str(d['offset'])) for d in l['mention_data'] if d['kb_id'] != 'NIL'])
        rs = set((str(d[0]), str(d[1])) for d in match2(t))
        if m == rs:
            exact += 1

        else:
            print(f'{t}\n{m}\n{rs}\n')
        R += len(m & rs)
        A += len(rs)
        B += len(m)


    print(f'total exact:{exact / 90000}')
    print(f'precision rate: {R / A}')
    print(f'recall rate: {R / B}')


def match_score(q, r):
    # char
    r_c = [c for c in r]
    qr_c = [c for c in q if c in r_c]

    char_jaccard_score = len(qr_c) / len(q)
    char_jaccard_score_s = len(set(qr_c)) / len(set(q))

    # word
    r_w = jieba.lcut(r)
    q_w = jieba.lcut(q)
    qr_w = [w for w in q_w if w in r_w]
    word_jaccard_score = len(qr_w) / len(q_w)
    word_jaccard_score_s = len(set(qr_w)) / len(set(q_w))

    matched_word = [_[0] for _ in match2(q) if _[0] in r]
    matched_word_score = len(matched_word) / len(q)

    return char_jaccard_score + char_jaccard_score_s + word_jaccard_score + word_jaccard_score_s + matched_word_score


def tst_entity_des_match():
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
    R, T = 0, 0

    for i, l in tqdm(enumerate(train_data)):
        if i > 20000:
            break
        l = json.loads(l)
        t = l['text']
        for d in l['mention_data']:
            subject = d['mention']
            kb_id = d['kb_id']
            if kb_id == 'NIL':
                continue
            if subject not in kb2id:
                print('*' * 50)
                print(f'{kb_id} - {subject} - {id2kb[kb_id]["subject_alias"]}')

            r_score = [(sid, 1.0) for sid in kb2id[subject]]
            # r_score = [(sid, match_score(t,id2kb[sid]['subject_desc'])) for sid in kb2id[subject]]
            # r_score = sorted(r_score, key=lambda x: x[1], reverse=True)
            r_score_ = r_score[:20]

            if kb_id in [_[0] for _ in r_score_]:
                R += 1
            # else:
            #     print(f'True kb_id:{kb_id}, desc:{id2kb[kb_id]["subject_desc"]}')
            #     print(f'True kb_id:{r_score[0][0]}, desc:{id2kb[r_score[0][0]]["subject_desc"]}')

            T += 1

    print(f'recall: {R / T}')


if __name__ == '__main__':
    # tst_entity_des_match()
    tst_recall()
    # for a in ['《乱世情》电视剧全集-在线观看-西瓜影音-美国电视剧 ...',
    #           '弗朗茨·舒伯特-德语学习2003年03期',
    #           '叶展_齐鲁证券资管叶展 - 私募基金经理 ╟ 中投在线']:
    #     print(match2(a))
