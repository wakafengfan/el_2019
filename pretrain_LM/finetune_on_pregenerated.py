import collections
import json
import logging
import random
from collections import namedtuple
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from pytorch_pretrained_bert.modeling import BertForPreTraining
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm

from configuration.config import data_dir, bert_data_path, bert_vocab_path, bert_model_path

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logging.info(f'data_dir: {data_dir}')

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

def convert_example_to_features(example, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in tokens]
    masked_label_ids = [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in masked_lm_labels]
    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, num_data_epochs, reduce_memory=False):
        self.vocab = bert_vocab
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}_v2.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        # seq_len = metrics['max_seq_len']
        seq_len = 253
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir / 'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir / 'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir / 'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir / 'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir / 'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
        logging.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)))

reduce_memory = True
train_batch_size = 32
warmup_proportion = 0.1
learning_rate = 3e-5
epochs = 4

samples_per_epoch = []
for i in range(epochs):
    epoch_file = Path(data_dir) /'pretrain_data_v2'/ f"epoch_{i}_v2.json"
    metrics_file = Path(data_dir)/'pretrain_data_v2' / f"epoch_{i}_metrics.json"
    if epoch_file.is_file() and metrics_file.is_file():
        metrics = json.loads(metrics_file.read_text())
        samples_per_epoch.append(metrics['num_training_examples'])
    else:
        if i == 0:
            exit("No training data was found!")
        print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({epochs}).")
        print("This script will loop over the available data, but training diversity may be negatively impacted.")
        num_data_epochs = i
        break
else:
    num_data_epochs = epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logging.info("device: {} n_gpu: {}".format(device, n_gpu))


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if n_gpu > 0:
    torch.cuda.manual_seed_all(42)

total_train_examples = 0
for i in range(epochs):
    # The modulo takes into account the fact that we may loop over limited epochs of data
    total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

num_train_optimization_steps = int(total_train_examples / train_batch_size)

# Prepare model
model = BertForPreTraining.from_pretrained(pretrained_model_name_or_path=bert_model_path, cache_dir=bert_data_path)
model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)

logging.info("***** Running training *****")
logging.info(f"  Num examples = {total_train_examples}")
logging.info("  Batch size = %d", train_batch_size)
logging.info("  Num steps = %d", num_train_optimization_steps)
model.train()
for epoch in range(epochs):
    epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=Path(data_dir)/'pretrain_data_v2',
                                        num_data_epochs=num_data_epochs, reduce_memory=reduce_memory)
    train_sampler = RandomSampler(epoch_dataset)
    train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=train_batch_size)
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
            loss = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            pbar.update(1)
            mean_loss = tr_loss / nb_tr_steps
            pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
            optimizer.step()
            optimizer.zero_grad()

# Save a trained model
logging.info("** ** * Saving fine-tuned model ** ** * ")
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self


torch.save(model_to_save.state_dict(), data_dir+'/finetune_pretrain.pt')
model_to_save.config.to_json_file(data_dir + '/finetune_pretrain_config.json')


