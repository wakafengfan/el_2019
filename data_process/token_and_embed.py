
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import jieba
from tqdm import tqdm

from configuration.config import data_dir, tencent_w2v_path
word_set = set()

if not (Path(data_dir)/'train_data_me.json').exists():

    train_data = []
    for doc in tqdm((Path(data_dir)/'train.json').open()):
        doc = json.loads(doc)
        text = doc['text']
        text_word = [i.lower() for i in jieba.lcut(text)]
        doc.update({'text_words':text_word})
        train_data.append(doc)

        word_set.update(text_word)

    train_upt_path = (Path(data_dir)/'train_data_me.json').open('w')
    json.dump(train_data, train_upt_path, indent=4, ensure_ascii=False)
else:
    for doc in tqdm(json.load((Path(data_dir)/'train_data_me.json').open())):
        text_word = doc['text_words']

        word_set.update(text_word)

if not (Path(data_dir)/'dev_data_me.json').exists():
    dev_data = []
    for doc in tqdm((Path(data_dir)/'develop.json').open()):
        doc = json.loads(doc)
        text = doc['text']
        text_word = [i.lower() for i in jieba.lcut(text)]
        doc.update({'text_words':text_word})
        dev_data.append(doc)

        word_set.update(text_word)

    dev_upt_path = (Path(data_dir)/'dev_data_me.json').open('w')
    json.dump(dev_data, dev_upt_path, indent=4, ensure_ascii=False)
else:
    for doc in tqdm(json.load((Path(data_dir)/'dev_data_me.json').open())):
        text_word = doc['text_words']

        word_set.update(text_word)


for l in tqdm((Path(data_dir) / 'kb_data').open()):
    _ = json.loads(l)
    for i in _['data']:
        if '摘要' in i['predicate']:
            subject_desc = i['object']
            text_word = jieba.lcut(subject_desc)
            word_set.update(text_word)


print(f'total word: {len(word_set)}')  # 110999 598900(加上摘要信息)

# get tencent embedding from .gz
tmpdir = tempfile.mkdtemp()
with tarfile.open(Path(tencent_w2v_path)/'Tencent_AILab_ChineseEmbedding.tar.gz', 'r:gz') as archive:
    archive.extractall(tmpdir)

serialization_dir = tmpdir
for fn in Path(serialization_dir).iterdir():
    print(fn.name)
    if 'Tencent_AILab_ChineseEmbedding' in fn.name:
        break
raw_tencent_embeds = fn.open(errors='ignore')
upt_tencent_embeds = (Path(data_dir)/'Tencent_AILab_ChineseEmbedding_for_el.txt').open('w')
if tmpdir:
    shutil.rmtree(tmpdir)
first_line = raw_tencent_embeds.readline()
print(first_line)


# filter tencent embedding to glove format
word_cnt = 0
for line in tqdm(raw_tencent_embeds):
    l = line.strip().split()
    if len(l) != 201 or l[0] not in word_set:
        continue
    word_cnt += 1
    upt_tencent_embeds.write(line)

print(word_cnt)  #100910 433675

