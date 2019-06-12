import collections
import json
import logging
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from random import choice

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertAdam
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig
from tqdm import tqdm

from baseline.model_zoo import SubjectModel, ObjectModel
from configuration.config import data_dir, bert_vocab_path, bert_model_path, bert_data_path


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
            subject_desc += f'{i["predicate"]}:{i["object"]}\n'

    subject_desc = subject_desc[:300].lower()
    if subject_desc:
        id2kb[subject_id] = {'subject_alias': subject_alias, 'subject_desc': subject_desc}

kb2id = defaultdict(list)  # subject: [sid1, sid2,...]
for i, j in id2kb.items():
    for k in j['subject_alias']:
        kb2id[k].append(i)

dev_data = []
for l in (Path(data_dir) / 'develop.json').open():
    _ = json.loads(l)
    dev_data.append({
        'text': _['text'].lower(),
        'text_id': _['text_id']
        # 'mention_data': [(x['mention'], int(x['offset']), x['kb_id'])
        #                  for x in _['mention_data'] if x['kb_id'] != 'NIL']
    })

id2char, char2id = json.load((Path(data_dir) / 'all_chars_me.json').open())

# load model
config = BertConfig(str(Path(data_dir)/'subject_model_config.json'))
subject_model = SubjectModel(config)
subject_model.load_state_dict(torch.load(Path(data_dir)/'subject_model.pt'))

object_model = ObjectModel()
object_model.load_state_dict(torch.load(Path(data_dir)/'object_model.pt'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

subject_model.to(device)
object_model.to(device)

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

def seq_padding(X):
    ML = max(map(len, X))
    return np.array([list(x) + [0] * (ML - len(x)) for x in X])

def extract_items(text_in):
    _s = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text_in]
    _input_mask = [1] * len(_s)
    _s = torch.tensor([_s], dtype=torch.long, device=device)  # [1,s1]
    _input_mask = torch.tensor([_input_mask], dtype=torch.long, device=device)
    _segment_ids = torch.zeros(*_s.size(), dtype=torch.long, device=device)

    with torch.no_grad():
        _k1, _k2, _x1_hs, _x1_h = subject_model('x1', _s, _segment_ids, _input_mask)  # _k1:[1,s]
        _k1 = _k1[0, :].detach().cpu().numpy()
        _k2 = _k2[0, :].detach().cpu().numpy()
        _k1, _k2 = np.where(_k1 > 0.5)[0], np.where(_k2 > 0.5)[0]

    _subjects = []
    if len(_k1) and len(_k2):
        for i in _k1:
            j = _k2[_k2 >= i]
            if len(j) > 0:
                j = j[0]
                _subject = text_in[i:j + 1]
                _subjects.append((_subject, i, j + 1))
    if _subjects:
        R = []
        _X2, _X2_MASK, _Y = [], [], []
        _S, _IDXS = [], {}
        for _s in _subjects:
            _y = np.zeros(len(text_in))
            _y[_s[1]:_s[2]] = 1
            _IDXS[_s] = kb2id.get(_s[0], [])
            for i in _IDXS[_s]:
                _x2 = id2kb[i]['subject_desc']
                _x2 = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in _x2]
                _x2_mask = [1] * len(_x2)
                _X2.append(_x2)
                _X2_MASK.append(_x2_mask)
                _Y.append(_y)
                _S.append(_s)
        if _X2:
            _X2 = torch.tensor(seq_padding(_X2), dtype=torch.long, device=device)  # [b,s2]
            _X2_MASK = torch.tensor(seq_padding(_X2_MASK), dtype=torch.long, device=device)
            _X2_SEG = torch.zeros(*_X2.size(), dtype=torch.long, device=device)
            _Y = torch.tensor(seq_padding(_Y), dtype=torch.float32, device=device)
            _X1_HS = _x1_hs.expand(_X2.size(0), -1, -1)  # [b,s1]
            _X1_H = _x1_h.expand(_X2.size(0), -1)  # [b,s1]
            _input_mask = _input_mask.expand(_X2.size(0), -1)  # [b,s1]

            with torch.no_grad():
                _x2, _x2_h = subject_model('x2', None,None,None,_X2, _X2_SEG, _X2_MASK)
                _o, _, _ = object_model(_X1_HS, _X1_H, _input_mask, _Y, _x2, _x2_h, _X2_MASK)  # _o:[b,1]
                _o = _o.detach().cpu().numpy()
                for k, v in groupby(zip(_S, _o), key=lambda x: x[0]):
                    v = np.array([j[1] for j in v])
                    kbid = _IDXS[k][np.argmax(v)]
                    R.append((k[0], k[1], kbid))
        return R
    else:
        return []

output_path = (Path(data_dir)/'submision.json').open('w')
for l in (Path(data_dir)/'develop.json').open():
    doc = json.loads(l)
    text = doc['text']
    R = extract_items(text)
    doc.update({
        'mention_data': [{'kb_id':r[2],'mention':r[0],'offset':r[1]} for r in R]
    })
    output_path.write(json.dumps(doc) + '\n')



