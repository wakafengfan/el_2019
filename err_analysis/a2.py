import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from configuration.config import data_dir
from configuration.match import match2


def group():
    id2kb = {}
    for l in tqdm((Path(data_dir) / 'kb_data').open(), desc='kb_data'):
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
    freq_dic = defaultdict(dict)

    cnt = 0
    tmp_p = (Path(data_dir)/'el_group_word.json').open('w')
    tmp_dic = {}

    for i, l in tqdm(enumerate(train_data), desc='train_data 1'):
        # if i > 20000:
        #     break
        l = json.loads(l)
        t = l['text']
        exp_words = [(k,sidx,int(sidx)+len(k)) for k,sidx,_ in match2(t)]
        labeled_words = [(m['mention'],m['offset'],int(m['offset'])+len(m['mention'])) for m in l['mention_data'] if m['kb_id']!='NIL']
        if not set(exp_words).issuperset(set(labeled_words)):
            for lw, lw_s, lw_e in labeled_words:
                for ew, ew_s, ew_e in exp_words:
                    if lw_s == ew_s and lw_e !=ew_e and ew.startswith(lw):
                        if ew not in tmp_dic:
                            tmp_dic[ew] = defaultdict(list)
                        tmp_dic[ew]['s_same'].append(lw)

                    if lw_e == ew_e and lw_s != ew_s and ew.endswith(lw):
                        if ew not in tmp_dic:
                            tmp_dic[ew] = defaultdict(list)
                        tmp_dic[ew]['e_same'].append(lw)

                    if lw_e == ew_e and lw_s != ew_s and ew.endswith(lw):
                        if ew not in tmp_dic:
                            tmp_dic[ew] = defaultdict(list)
                        tmp_dic[ew]['e_same'].append(lw)
            cnt+= 1

    for i, l in tqdm(enumerate((Path(data_dir) / 'train.json').open()), desc='train_data 2'):
        l = json.loads(l)
        ews = [k for k,_,_ in match2(l['text'])]
        lws = [m['mention'] for m in l['mention_data'] if m['kb_id']!='NIL']
        for w in ews:
            if w in tmp_dic:
                tmp_dic[w]['group_exp'].append(1)
        for w in lws:
            if w in tmp_dic:
                tmp_dic[w]['group_labeled'].append(1)
    tmp_sum_dic = defaultdict(dict)
    for w in tqdm(tmp_dic, desc='tmp_sum_dic'):
        tmp_sum_dic[w]['group_exp_cnt'] = len(tmp_dic[w]['group_exp'])
        tmp_sum_dic[w]['group_labeled_cnt'] = len(tmp_dic[w]['group_labeled'])
        tmp_sum_dic[w]['s_same_cnt'] = len(tmp_dic[w]['s_same'])
        tmp_sum_dic[w]['e_same_cnt'] = len(tmp_dic[w]['e_same'])
        tmp_sum_dic[w]['group_labeled_per'] = tmp_sum_dic[w]['group_labeled_cnt'] / tmp_sum_dic[w]['group_exp_cnt']
        tmp_sum_dic[w]['s_same_per'] = tmp_sum_dic[w]['s_same_cnt'] / tmp_sum_dic[w]['group_exp_cnt']
        tmp_sum_dic[w]['e_same_per'] = tmp_sum_dic[w]['e_same_cnt'] / tmp_sum_dic[w]['group_exp_cnt']




    json.dump(tmp_sum_dic, tmp_p, ensure_ascii=False)

    #     for w, start_idx in exp_words:
    #         if 'exp' not in freq_dic[w]:
    #             freq_dic[w]['exp'] = 1
    #         else:
    #             freq_dic[w]['exp'] += 1
    #         if (w,start_idx) in labeled_words:
    #             if 'labeled' not in freq_dic[w]:
    #                 freq_dic[w]['labeled'] = 1
    #             else:
    #                 freq_dic[w]['labeled'] += 1
    #
    # # if match_rules(word)  21825
    # # if word != ''  16347
    # print(f'cnt: {cnt}')
    #
    # for w in freq_dic:
    #     if 'labeled' not in freq_dic[w]:
    #         freq_dic[w]['labeled'] = 0
    #     freq_dic[w]['per'] = freq_dic[w]['labeled']/freq_dic[w]['exp']
    # a = [(w, freq_dic[w]['per']) for w in freq_dic]
    # a = sorted(a, key=lambda x: x[1], reverse=True)
    #
    #
    # p = (Path(data_dir)/'el_freq_dic_1.json').open('w')
    # json.dump(freq_dic, p, ensure_ascii=False)
    # for w in freq_dic:
    #     doc = freq_dic[w]
    #     doc.update({'word': w})
    #     s = json.dumps(doc, ensure_ascii=False)
    #     p.write(s + '\n')


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
    freq_dic = defaultdict(dict)
    cnt = 0
    for i, l in tqdm(enumerate(train_data)):
        # if i > 20000:
        #     break
        l = json.loads(l)
        t = l['text']
        exp_words = [(k,sidx) for k,sidx,_ in match2(t)]
        labeled_words = [(m['mention'],m['offset']) for m in l['mention_data'] if m['kb_id']!='NIL']
        if not set(exp_words).issuperset(set(labeled_words)):
            cnt+= 1


if __name__ == '__main__':
    group()
