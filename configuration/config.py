import os
from pathlib import Path

ROOT_PATH = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

# data
data_dir = os.path.join(ROOT_PATH, "data")
model_dir = os.path.join(ROOT_PATH, "model")

bert_data_path = Path.home()/'.pytorch_pretrained_bert'
bert_vocab_path = bert_data_path / 'bert-base-chinese-vocab.txt'
bert_model_path = bert_data_path / 'bert-base-chinese.tar.gz'

tencent_w2v_path = Path.home()/'.word2vec'


# expire list
expire_list = ['开拍','出来','整理','下载','出演','首次','在线','在线播放',
               '激情','在线','阅读','代表','著作','专业','所谓','一行','张先',
               '为什么','上线','开局','改变','感觉','跪求','什么', '...','——',
               '哪里','关于','分钟','关于','吓人',]