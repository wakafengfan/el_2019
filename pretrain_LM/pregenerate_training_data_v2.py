import collections
import json
import random
from functools import partial
from pathlib import Path

from tqdm import trange, tqdm

from configuration.config import data_dir, bert_vocab_path
from multiprocessing import Pool


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
            subject_desc = ' '.join(subject_alias)[:50] + ' ' + subject_desc[:100].lower() + ' ' + subject_desc[
                                                                                                   -100:].lower()
        else:
            subject_desc = ' '.join(subject_alias)[:50] + ' ' + subject_desc[:200].lower()
    if subject_desc:
        id2kb[subject_id] = {'subject_alias': subject_alias, 'subject_desc': subject_desc}

kb2id = collections.defaultdict(list)  # subject: [sid1, sid2,...]
for i, j in id2kb.items():
    for k in j['subject_alias']:
        kb2id[k].append(i)

epochs_to_generate = 4
max_seq_len_1 = 50
max_seq_len_2 = 200
short_seq_prob = 0.1
masked_lm_prob = 0.15
max_predictions_per_seq = 20


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


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", 'label'])
def create_masked_lm_predictions(T):
    cand_indices = []
    for (i, token) in enumerate(T):
        if token in ['[CLS]', '[SEP]']:
            continue
        cand_indices.append(i)

    random.shuffle(cand_indices)
    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(T) * masked_lm_prob))))
    masked_lms, covered_indexes = [], set()
    for index in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        if len(masked_lms) + 1 > num_to_mask:
            continue
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_token = None

        # 80% of the time , replace with [MASK]
        if random.random() < 0.8:
            masked_token = '[MASK]'

        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = T[index]
            # 10% of the time, replace with random word
            else:
                masked_token = random.choice(list(bert_vocab.keys()))

        masked_lms.append(MaskedLmInstance(index=index, label=T[index]))
        T[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x:x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return T, mask_indices, masked_token_labels

train_data = json.load((Path(data_dir)/'train_data_me.json').open())

def next_sentence(d):
    tmp_instance = []
    seq_a = d['text']
    tokens_a = [c for c in seq_a[:max_seq_len_1]]

    for m in d['mention_data']:
        if m['mention'] not in kb2id:
            continue

        # 非随机next sentence
        if m['kb_id'] != 'NIL':
            seq_b = id2kb[m['kb_id']]['subject_desc']
            tokens_b = [c for c in seq_b[:max_seq_len_2]]
            tokens = ["[CLS]"] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
            segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]
            tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokens)
            instance = {
                'tokens': tokens,
                'segment_ids': segment_ids,
                'is_random_next': False,
                'masked_lm_positions': masked_lm_positions,
                'masked_lm_labels': masked_lm_labels
            }
            tmp_instance.append(instance)


        # 随机next sentence
        seq_b_non_1_kids = kb2id[m['mention']].copy()
        if m['kb_id'] in seq_b_non_1_kids:
            seq_b_non_1_kids.remove(m['kb_id'])
            seq_b_non_1_kids = random.sample(seq_b_non_1_kids, k=min(len(seq_b_non_1_kids), 2))
        else:
            print(f"***{m['mention']}***")

        seq_b_non_2_kids = set(id2kb.keys()) - set(kb2id[m['mention']])
        seq_b_non_2_kids = random.sample(seq_b_non_2_kids, k=min(len(seq_b_non_2_kids), 2))

        for kid in seq_b_non_1_kids + seq_b_non_2_kids:
            seq_b = id2kb[kid]['subject_desc']
            tokens_b = [c for c in seq_b[:max_seq_len_2]]
            tokens = ["[CLS]"] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
            segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]
            tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokens)
            instance = {
                'tokens': tokens,
                'segment_ids': segment_ids,
                'is_random_next': True,
                'masked_lm_positions': masked_lm_positions,
                'masked_lm_labels': masked_lm_labels
            }
            tmp_instance.append(instance)
    return tmp_instance



threads = 8
chunk_size = 64
for epoch in trange(epochs_to_generate):
    epoch_filename = (Path(data_dir)/ f'epoch_{epoch}.json').open('w')

    instances = []
    with Pool(threads) as p:
        func = partial(next_sentence)
        tmp_list = list(tqdm(p.imap(func, train_data, chunksize=chunk_size), desc=f'Epoch:{epoch}'))

    for tmp in tmp_list:
        instances.extend(tmp)

    for instance in instances:
        epoch_filename.write(json.dumps(instance, ensure_ascii=False) + '\n')

    metrics_filename = (Path(data_dir)/f'epoch_{epoch}_metrics.json').open('w')
    metrics_filename.write(json.dumps({'num_training_instance': len(instances)}))











