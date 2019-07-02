import collections
import json
import logging
from collections import defaultdict
from pathlib import Path
from random import choice

import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertAdam, BertConfig
from tqdm import tqdm

from baseline_3.model_zoo import SubjectModel
from configuration.config import data_dir, bert_vocab_path, bert_model_path, bert_data_path
from configuration.match import match2

min_count = 2
mode = 0
hidden_size = 768
epoch_num = 10
batch_size = 64

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

id2kb = {}
for l in (Path(data_dir) / 'kb_data').open():
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
            subject_desc += f'{i["predicate"]}:{i["object"]}' + ' '

    subject_desc = ' '.join(subject_alias)[:50] + ' ' + subject_desc[:100].lower()
    if subject_desc:
        id2kb[subject_id] = {'subject_alias': subject_alias, 'subject_desc': subject_desc}

kb2id = defaultdict(list)  # subject: [sid1, sid2,...]
for i, j in id2kb.items():
    for k in j['subject_alias']:
        kb2id[k].append(i)

train_data = []
for l in tqdm(json.load((Path(data_dir) / 'train_data_me.json').open())):
    train_data.append({
        'text': l['text'].lower(),
        'mention_data': [(x['mention'].lower(), int(x['offset']), x['kb_id'])
                         for x in l['mention_data'] if x['kb_id'] != 'NIL'],
        'text_words': list(map(lambda x: x.lower(), l['text_words']))
    })

if not (Path(data_dir) / 'random_order_train.json').exists():
    random_order = list(range(len(train_data)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        (Path(data_dir) / 'random_order_train.json').open('w'),
        indent=4
    )
else:
    random_order = json.load((Path(data_dir) / 'random_order_train.json').open())

dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]


def seq_padding(X):
    ML = max(map(len, X))
    return np.array([list(x) + [0] * (ML - len(x)) for x in X])


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


bert_vocab = load_vocab(bert_vocab_path)

# wv_model = gensim.models.KeyedVectors.load(str(Path(data_dir) / 'tencent_embed_for_el2019'))
# word2vec = wv_model.wv.syn0
# word_size = word2vec.shape[1]
# word2vec = np.concatenate([np.zeros((1, word_size)), np.zeros((1, word_size)), word2vec])  # [word_size+2,200]
# id2word = {i + 2: j for i, j in enumerate(wv_model.wv.index2word)}
# word2id = {j: i for i, j in id2word.items()}
#
#
# def seq2vec(token_ids):
#     V = []
#     for s in token_ids:
#         V.append([])
#         for w in s:
#             for _ in w:
#                 V[-1].append(word2id.get(w, 1))
#     V = seq_padding(V)
#     V = word2vec[V]
#     return V


class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        idxs = list(range(len(self.data)))
        np.random.shuffle(idxs)
        X1, S1, S2, Y, X1_MASK = [], [], [], [], []
        for i in idxs:
            d = self.data[i]
            text = d['text']

            x1 = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text]
            x1_mask = [1] * len(x1)
            s1, s2 = np.zeros(len(text)), np.zeros(len(text))
            mds = {}
            for md in d['mention_data']:
                if md[0] in kb2id:  # train subject存在于kb subject
                    j1 = md[1]
                    j2 = md[1] + len(md[0])
                    s1[j1] = 1
                    s2[j2 - 1] = 1
                    mds[(j1, j2)] = (md[0], md[2])

            if mds:
                j1, j2 = choice(list(mds.keys()))
                y = np.zeros(len(text))
                y[j1:j2] = 1

                X1.append(x1)
                S1.append(s1)
                S2.append(s2)
                Y.append(y)
                X1_MASK.append(x1_mask)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = torch.tensor(seq_padding(X1), dtype=torch.long)  # [b,s1]
                    S1 = torch.tensor(seq_padding(S1), dtype=torch.float32)  # [b,s1]
                    S2 = torch.tensor(seq_padding(S2), dtype=torch.float32)  # [b,s1]
                    Y = torch.tensor(seq_padding(Y), dtype=torch.float32)  # [b,s1]
                    X1_MASK = torch.tensor(seq_padding(X1_MASK), dtype=torch.long)
                    X1_SEG = torch.zeros(*X1.size(), dtype=torch.long)

                    yield [X1, S1, S2, Y, X1_MASK, X1_SEG]
                    X1, S1, S2, Y, X1_MASK = [], [], [], [], []


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()

pretrain = True
if pretrain:
    config = BertConfig(str(Path(data_dir) / 'subject_1/subject_model_config.json'))
    subject_model = SubjectModel(config)
    subject_model.load_state_dict(
        torch.load(Path(data_dir) / 'subject_1/subject_model.pt', map_location='cpu' if not torch.cuda.is_available() else None))

else:
    subject_model = SubjectModel.from_pretrained(pretrained_model_name_or_path=bert_model_path, cache_dir=bert_data_path)

subject_model.to(device)
if n_gpu > 1:
    torch.cuda.manual_seed_all(42)

    logger.info(f'let us use {n_gpu} gpu')
    subject_model = torch.nn.DataParallel(subject_model)

freq = json.load((Path(data_dir)/'el_freq_dic_1.json').open())
group = json.load((Path(data_dir)/ 'el_group_word.json').open())

def extract_items(text_in):
    _X1 = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text_in]
    _X1_MASK = [1] * len(_X1)
    _X1 = torch.tensor([_X1], dtype=torch.long, device=device)  # [1,s1]
    _X1_MASK = torch.tensor([_X1_MASK], dtype=torch.long, device=device)
    _X1_SEG = torch.zeros(*_X1.size(), dtype=torch.long, device=device)

    with torch.no_grad():
        _k1, _k2, _ = subject_model(device, _X1, _X1_SEG, _X1_MASK)  # _k1:[1,s]
        _k1 = _k1[0, :].detach().cpu().numpy()
        _k2 = _k2[0, :].detach().cpu().numpy()
        _k1, _k2 = np.where(_k1 > 0.3)[0], np.where(_k2 > 0.5)[0]

    _subjects = []
    if len(_k1) and len(_k2):
        for i in _k1:
            j = _k2[_k2 >= i]
            if len(j) > 0:
                j = j[0]
                _subject = text_in[i:j + 1]
                if _subject in kb2id:
                    if _subject in freq:
                        if freq[_subject]['per'] > 0:
                            _subjects.append((_subject, str(i), str(j + 1)))
                    else:
                        _subjects.append((_subject, str(i), str(j + 1)))

    # subject补余
    for _s in match2(text_in):
        if _s[0] in freq:
            if freq[_s[0]]['per'] > 0.8:
                _subjects.append(_s)

    _subjects = list(set(_subjects))
    _subjects_new = _subjects.copy()
    for _s,_s_s, _s_e in _subjects:
        for _i, _i_s,_i_e in _subjects:
            if _s_s == _i_s and _s_e != _i_e and _s in group:
                if group[_s]['group_labeled_per'] > 1.5*group[_s]['s_same_per'] and (_i,_i_s,_i_e) in _subjects_new:
                    _subjects_new.remove((_i,_i_s,_i_e))
                if group[_s]['s_same_per'] > 1.5*group[_s]['group_labeled_per'] and (_s,_s_s,_s_e) in _subjects_new:
                    _subjects_new.remove((_s,_s_s,_s_e))

            if _s_s != _i_s and _s_e == _i_e and _s in group:
                if group[_s]['group_labeled_per'] > 1.5*group[_s]['e_same_per'] and (_i,_i_s,_i_e) in _subjects_new:
                    _subjects_new.remove((_i,_i_s,_i_e))
                if group[_s]['e_same_per'] > 1.5*group[_s]['group_labeled_per'] and (_s,_s_s,_s_e) in _subjects_new:
                    _subjects_new.remove((_s,_s_s,_s_e))

    return list(set(_subjects_new))


subject_model.eval()

A, B, C = 1e-10, 1e-10, 1e-10
err_dict = defaultdict(list)
for eval_idx, d in tqdm(enumerate(dev_data)):
    M = [m for m in d['mention_data'] if m[0] in kb2id]

    R = set(map(lambda x: (str(x[0]), str(x[1])), set(extract_items(d['text']))))
    T = set(map(lambda x: (str(x[0]), str(x[1])), set(M)))
    A += len(R & T)
    B += len(R)
    C += len(T)

    if R != T:
        err_dict['err'].append({'text': d['text'],
                                'mention_data': list(T),
                                'predict': list(R)})
    if eval_idx % 100 == 0:
        logger.info(f'eval_idx:{eval_idx} - precision:{A/B:.5f} - recall:{A/C:.5f} - f1:{2 * A / (B + C):.5f}')

f1, precision, recall = 2 * A / (B + C), A / B, A / C
json.dump(err_dict, (Path(data_dir) / 'err_log_[el_pt_subject_eval.py].json').open('w'), ensure_ascii=False)
logger.info(f'precision:{precision:.4f}-recall:{recall:.4f}-f1:{f1:.4f}')
