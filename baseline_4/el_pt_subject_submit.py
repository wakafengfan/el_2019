import collections
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertConfig
from tqdm import tqdm

from baseline_4.model_zoo import SubjectModel
from configuration.config import data_dir, bert_vocab_path
from configuration.match import match2

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

config = BertConfig(str(Path(data_dir) / 'subject_model_config.json'))
subject_model = SubjectModel(config)
subject_model.load_state_dict(
        torch.load(Path(data_dir) / 'subject_model.pt', map_location='cpu' if not torch.cuda.is_available() else None))


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
            if freq[_s[0]]['per'] > 0.8 or (freq[_s[0]]['exp']<5 and freq[_s[0]]['per']==0.5):
                _subjects.append(_s)

    _subjects = list(set(_subjects))
    _subjects_new = _subjects.copy()
    for _s,_s_s, _s_e in _subjects:
        for _i, _i_s,_i_e in _subjects:
            if _s_s == _i_s and _s_e != _i_e and _s in group:
                _subjects_new.remove((_i,_i_s,_i_e))

            if _s_s != _i_s and _s_e == _i_e and _s in group:
                _subjects_new.remove((_i,_i_s,_i_e))

    return list(set(_subjects_new))


subject_model.eval()
output_path = Path('submission_subject.json').open('w')
for l in tqdm((Path(data_dir) / 'develop.json').open()):
    doc = json.loads(l)
    text = doc['text']
    R = extract_items(text)
    doc.update({
        'mention_data': [(r[0], int(r[1])) for r in R]
    })
    output_path.write(json.dumps(doc, ensure_ascii=False) + '\n')

