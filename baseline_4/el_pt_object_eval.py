import collections
import json
import logging
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertConfig
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baseline_4.model_zoo import ObjectModel
from configuration.config import data_dir, bert_vocab_path

min_count = 2
mode = 0
hidden_size = 768
epoch_num = 10
batch_size = 32

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
    subject_desc_all = ''
    for i in _['data']:
        # if '摘要' in i['predicate']:
        #     subject_desc = i['object']
        #     break
        # else:
        # subject_desc += f'{i["predicate"]}:{i["object"]}' + ' '
        if i['predicate'] in ['摘要', '标签', '义项描述']:
            subject_desc += i['object'] + ' '
        subject_desc_all += f'{i["predicate"]}:{i["object"]}' + ' '
    if subject_desc == '':
        subject_desc = ' '.join(subject_alias)[:50] + ' ' + subject_desc_all[:200].lower()
    else:
        if len(subject_desc) > 200:
            subject_desc = ' '.join(subject_alias)[:50] + ' ' + subject_desc[:100].lower() + ' ' + subject_desc[-100:].lower()
        else:
            subject_desc = ' '.join(subject_alias)[:50] + ' ' + subject_desc[:200].lower()
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
        'mention_data': [(x['mention'].lower(), int(x['offset']), x['kb_id']) for x in l['mention_data']],
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
test_data, train_data = train_data[-10000:], train_data[:-10000]


def seq_padding(X):
    ML = max(map(len, X))
    return np.array([list(x) + [0] * (ML - len(x)) for x in X])

def seq_padding_bert(X_):
    X_ids, X_SEGs, X_MASKs = [],[],[]
    ML = max(map(lambda x: len(x[0]) + len(x[1]) + 3, X_))
    for _ in X_:
        X1_ids = [bert_vocab['[CLS]']] + _[0] + [bert_vocab['[SEP]']]
        X1_SEG = [0] * len(X1_ids)
        X2_ids = _[1]
        X2_SEG = [1] * len(X2_ids)

        X_id = X1_ids + X2_ids
        X_SEG = X1_SEG + X2_SEG

        L = len(X_id)
        X_MASK = [1] * len(X_id) + [0] * (ML-L)

        X_id += [0] * (ML-L)  # padding
        X_SEG += [0] * (ML-L)

        X_ids.append(X_id)
        X_SEGs.append(X_SEG)
        X_MASKs.append(X_MASK)

    return X_ids, X_SEGs, X_MASKs


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()

config = BertConfig(str(Path(data_dir) / 'object_model_config.json'))
object_model = ObjectModel(config)
object_model.load_state_dict(
    torch.load(Path(data_dir) / 'object_model.pt', map_location='cpu' if not torch.cuda.is_available() else None))

object_model.to(device)
if n_gpu > 1:
    torch.cuda.manual_seed_all(42)

    logger.info(f'let us use {n_gpu} gpu')
    object_model = torch.nn.DataParallel(object_model)


def extract_items(d):
    _subjects = []
    text = d['text']
    _x1 = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text]
    mention_data = d['mention_data']

    if mention_data:
        for m in mention_data:
            if m[0] in kb2id:
                _subjects.append((m[0], m[1], m[1]+len(m[0])))

    if _subjects:
        R = []
        _X, _S = [], []
        _IDXS = {}
        for _X1 in _subjects:
            _IDXS[_X1] = kb2id.get(_X1[0], [])
            for idx, i in enumerate(_IDXS[_X1]):
                # if idx > 15:  # 只取15个匹配
                #     break
                _x2 = id2kb[i]['subject_desc']
                _x2 = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in _x2]

                _X.append((_x1, _x2))
                _S.append(_X1)
        if _X:
            _O = []

            _X_ids, _X_SEGs, _X_MASKs = seq_padding_bert(_X)
            _X_ids = torch.tensor(_X_ids, dtype=torch.long)
            _X_SEGs = torch.tensor(_X_SEGs, dtype=torch.long)
            _X_MASKs = torch.tensor(_X_MASKs, dtype=torch.long)

            eval_dataloader = DataLoader(
                TensorDataset(_X_ids, _X_SEGs, _X_MASKs), batch_size=64)

            for batch_idx, batch in enumerate(eval_dataloader):
                batch = tuple(t.to(device) for t in batch)
                _x_ids, _x_seg, _x_mask = batch
                with torch.no_grad():
                    _o = object_model(_x_ids, _x_seg, _x_mask)  # _o:[b,1]
                    _o = _o.detach().cpu().numpy()
                    _O.extend(_o)

            for k, v in groupby(zip(_S, _O), key=lambda x: x[0]):
                v = np.array([j[1] for j in v])
                # if np.max(v) < 0.1:
                #     R.append((k[0], k[1], 'NIL', np.max(v)))
                # else:
                kbid = _IDXS[k][np.argmax(v)]
                R.append((k[0], k[1], kbid, np.max(v)))
        return list(set(R))
    else:
        return []


object_model.eval()
A, B, C = 1e-10, 1e-10, 1e-10
err_dict = defaultdict(list)


# for eval_idx, d in tqdm(enumerate((Path(data_dir)/'eval_subject.json').open())):
#     d = json.loads(d)
for eval_idx, d in enumerate(test_data):
    M = [m for m in d['mention_data'] if m[0] in kb2id]
    p = set(map(lambda x: (str(x[0]), str(x[1]), str(x[2]), f'{x[3]:.5f}'), extract_items(d)))

    R = set(map(lambda x: (str(x[0]), str(x[1]), str(x[2])), p))
    T = set(map(lambda x: (str(x[0]), str(x[1]), str(x[2])), set(M)))
    A += len(R & T)
    B += len(R)
    C += len(T)

    if R != T:
        err_dict['err'].append({'text': d['text'],
                                'mention_data': list(T),
                                'predict': list(p)})
    if eval_idx % 100 == 0:
        logger.info(f'eval_idx:{eval_idx} - precision:{A/B:.5f} - recall:{A/C:.5f} - f1:{2 * A / (B + C):.5f}')

json.dump(err_dict, (Path(data_dir) / 'err_log__[el_pt_object_eval.py].json').open('w'), ensure_ascii=False, indent=4)

f1, precision, recall = 2 * A / (B + C), A / B, A / C
logger.info(f'precision:{precision:.4f}-recall:{recall:.4f}-f1:{f1:.4f}')
