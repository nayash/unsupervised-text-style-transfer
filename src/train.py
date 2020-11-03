from models import *
import os
import pickle
from pathlib import Path
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.mwe import MWETokenizer
import nltk
from nltk.corpus import stopwords
import sys
import numpy as np
import time
import json
import argparse
from tqdm.auto import tqdm
from logger import Logger
from utils import *
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, random_split
import torch.nn.init as init

'''
supported arguments
---------------------

input file with source sentences: -s or --insrc
input file with target sentences: -t or --intgt
help : -h or --help
path to file with raw (text as paragraphs) source data: --rawsrc
path to file with raw (text as paragraphs) target data: --rawtgt => if
rawsrc or rawtgt is provided then data is processed and stored at
input path. 'insrc' or 'intgt' params are ignored if provided and the
processed (and saved) data is used instead.
model config: -c or --config  -> path to a file with model configuration
in JSON format (python dict like). Check out default config dic for
all supported keys.

'''

seed = 999
np.random.seed(seed)

ROOT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent
INPUT_PATH = ROOT_PATH / 'inputs'
OUTPUT_PATH = ROOT_PATH / 'outputs'
GLOVE_PATH = INPUT_PATH / 'glove.6B.200d.txt'
YELP_PATH = INPUT_PATH / 'yelp-reviews-preprocessed'
src_sents = 'sentiment.0.train.txt'
tgt_sents = 'sentiment.1.train.txt'

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-s', '--insrc', default=YELP_PATH / src_sents,
                        help='path for source data file with each source \
                        sentence in a new line. Default is yelp negative')

arg_parser.add_argument('-t', '--intgt', default=YELP_PATH / tgt_sents,
                        help='path for target data file with each target \
                        sentence in a new line. Default is yelp dataset')
arg_parser.add_argument('-c', '--config', default=INPUT_PATH / 'config.json',
                        help='configuration/hyperparameters in json format')
arg_parser.add_argument('-e', '--expid', default='temp',
                        help='identifier for track your experiment. This is\
                        used to create log files folder. All files specific \
                        to this experiment will be stored in a specific \
                        folder. If passed run_id already exists, exception \
                        is be thrown. Use special "temp" for test runs.')
arg_parser.add_argument('-f', '--force', default=False, action='store_true',
                        help='if True then the data clean up processing \
                        is done again instead of of using the saved data \
                        checkpoints')

args = arg_parser.parse_args()
source_file_path = os.path.abspath(args.insrc)
target_file_path = os.path.abspath(args.intgt)
config_path = os.path.abspath(args.config)
run_id = args.expid
force_preproc = args.force

run_path = OUTPUT_PATH / 'runs' / run_id
log_path = run_path / 'logs'
data_cp_path = OUTPUT_PATH / 'data_cp.pk'
tensors_path = OUTPUT_PATH / 'data_tensors_cp.pt'

src2tgt = 'C2NQ'
tgt2src = 'NQ2C'
src2src = 'C2C'
tgt2tgt = 'NQ2NQ'

SOS_SRC = 'SOSSRC'
SOS_TGT = 'SOSTGT'


with open(source_file_path, encoding='utf-8',
          errors='ignore') as file:
    src_sents = file.readlines()

with open(target_file_path, encoding='utf-8',
          errors='ignore') as file:
    tgt_sents = file.readlines()

if not os.path.exists(run_path):
    os.makedirs(run_path)
else:
    if run_id != 'temp':
        raise Exception('run_id already exists')

logger = Logger(str(log_path), run_id, std_out=True)
with open(config_path, 'r') as file:
    _ = file.read()
    config_dict = json.loads(_)
logger.append_log('config: ', config_dict)

if force_preproc or not data_cp_path.exists():
    logger.append_log('cleaning up sentences...')
    src_sents = [clean_text_yelp(sent) for sent in tqdm(src_sents)]
    tgt_sents = [clean_text_yelp(sent) for sent in tqdm(tgt_sents)]

    logger.append_log('building vocabulary...')
    words = word_tokenize(' '.join(src_sents))
    words.extend(word_tokenize(' '.join(tgt_sents)))
    count_dict = Counter(words)
    words = [w for w in count_dict.keys() if count_dict[w] > 1]
    words = list(set(words))
    logger.append_log('number of words', len(words))

    logger.append_log('fetching word embeddings from Glove ...')
    extra_tokens = [SOS_SRC, SOS_TGT, 'EOS', 'PAD', 'UNK']
    words.extend(extra_tokens)
    word2idx, idx2word, word_emb, diff = vocab_from_pretrained_emb(
        GLOVE_PATH, words)

    data_cp = {'tgt': tgt_sents, 'src': src_sents, 'words': words}
    max_len = max([len(word_tokenize(sent)) for sent in src_sents + tgt_sents])
    data_cp['max_sent_len'] = max_len
    data_cp['word2idx'] = word2idx
    data_cp['idx2word'] = idx2word
    data_cp['word_emb'] = word_emb
    pickle.dump(data_cp, open(str(data_cp_path), 'wb'))
    logger.append_log('saved cleaned up sentences at {}, file size {}\
            KB'.format(str(data_cp_path), data_cp_path.stat().st_size // 1024))

data_cp = pickle.load(open(str(data_cp_path), 'rb'))
tgt_sents = data_cp['tgt']
src_sents = data_cp['src']
words = data_cp['words']
max_len = data_cp['max_sent_len']
max_len += 3  # extra tokens for SOS, EOS, OpType
word2idx = data_cp['word2idx']
idx2word = data_cp['idx2word']
word_emb = data_cp['word_emb']
print(len(src_sents), len(tgt_sents), len(words), max_len)

logger.append_log('number of samples in source and target are\
 {}, {}'.format(len(src_sents), len(tgt_sents)))

word_emb_tensor = torch.tensor(word_emb)
for k, v in word2idx.items():
    assert k == idx2word[v]


def sent_to_tensor(sentence, word2idx=word2idx, max_len=max_len,
                   prefix=None, shuffle=False, type='neg', dropout=False):
    temp = []
    sos = word2idx[SOS_SRC] if type == 'neg' else word2idx[SOS_TGT]
    temp.append(sos)
    if prefix:
        for _ in prefix.split():
            temp.append(word2idx[_])
    words = word_tokenize(sentence.strip())

    if dropout:
        drop_idx = np.random.randint(len(words))
        # don't drop NER mask token
        if not words[drop_idx].isupper() and \
                not words[drop_idx] == '.' and not words[drop_idx] == '?':
            words = words[:drop_idx] + words[drop_idx + 1:]

    if shuffle:
        words = permute_items(words, k=4)

    temp.extend([word2idx.get(w, word2idx['UNK']) for w in words])
    temp.append(word2idx['EOS'])
    temp.extend([word2idx['PAD']] * (max_len - len(temp)))
    return torch.tensor(temp)  # , device=device


def tensor_to_sentence(sent_tensor):
    sent_tensor = sent_tensor.squeeze().cpu().numpy()
    sent = []
    for idx in sent_tensor:
        sent.append(idx2word[idx])
    return ' '.join(sent)


def get_noisy_tensor(tensor, drop_prob=0.1, k=3, word2idx=word2idx):
    try:
        eos_index = (tensor == word2idx['EOS']).nonzero(as_tuple=False)[0][0]
    except:
        eos_index = tensor.size(0) - 1
    proper_tensor = tensor[1:eos_index]
    final = torch.zeros(tensor.size())
    final[:] = word2idx['PAD']
    final[0] = tensor[0]

    drop_mask = torch.FloatTensor(proper_tensor.size(0)).uniform_(0, 1)
    res = proper_tensor[drop_mask > drop_prob]

    res = permute_tensor(res)
    final[1:res.size(0) + 1] = res
    final[res.size(0) + 1] = word2idx['EOS']
    return final.long()


def get_noisy_tensor_grad(tensor, drop_prob=0.1, k=3, word2idx=word2idx):
    try:
        try:
            eos_index = (tensor == word2idx['EOS']).nonzero(
                as_tuple=False)[0][0]
        except:
            eos_index = tensor.size(0) - 1

        tensor[1:eos_index] = tensor[1:eos_index][torch.randperm(eos_index - 1)]
        # drop_mask = torch.FloatTensor(tensor[1:eos_index].size(0)).uniform_(0, 1)
    except Exception as e:
        print('Exception', e, word2idx['EOS'], eos_index)
    return tensor


def row_apply(tensor, func, is_stack=True):
    res = []
    for row in tensor:
        res.append(func(row))
    return torch.stack(res) if is_stack else res


def weights_init(m):
    try:
        if isinstance(m, torch.nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                    # print('filled', m)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
                    # print('filled', m)
        else:
            if not isinstance(m, nn.Embedding):
                init.xavier_normal_(m.weight.data)
                # print('filled', m)
    except:
        # print('failed', m)
        pass


def isNan(tensor):
    return torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item()


if force_preproc or not tensors_path.exists():
    logger.append_log('converting sentences to tensors...')
    x_src = []
    y_src = []
    x_tgt = []
    y_tgt = []

    natq_label = 1
    cloze_label = 0

    permute_prob = config_dict['permute_prob']
    drop_noise = config_dict['drop_noise']

    for i, cloze in enumerate(src_sents):
        tensor = sent_to_tensor(cloze.strip(), word2idx, max_len, type='neg')
        x_src.append(get_noisy_tensor(tensor))
        y_src.append(tensor.clone())

    print('len x_src', len(x_src))

    for i, q in enumerate(tgt_sents):
        tensor = sent_to_tensor(q.strip(), word2idx, max_len, type='pos')
        x_tgt.append(get_noisy_tensor(tensor))
        y_tgt.append(tensor.clone())

    print('len x_tgt', len(x_tgt))
    x_src = torch.stack(x_src)
    y_src = torch.stack(y_src)
    x_tgt = torch.stack(x_tgt)
    y_tgt = torch.stack(y_tgt)

    delta = x_tgt.shape[0] - x_src.shape[0]
    if delta > 0:
        indexes = np.random.choice(x_src.size(0), delta)
        x_src = torch.cat((x_src, x_src[indexes]))
        y_src = torch.cat((y_src, y_src[indexes]))
    else:
        indexes = np.random.choice(x_tgt.size(0), np.abs(delta))
        x_tgt = torch.cat((x_tgt, x_tgt[indexes]))
        y_tgt = torch.cat((y_tgt, y_tgt[indexes]))

    assert len(x_src) == len(x_tgt)
    # x_src.half(), y_src.half(), x_tgt.half(), y_tgt.half()

    data_tensors_cp = {'x_src': x_src, 'y_src': y_src,
                       'x_tgt': x_tgt, 'y_tgt': y_tgt}
    torch.save(data_tensors_cp, str(tensors_path))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    logger.append_log('warning!!! CPU device is being used')
data_tensors_cp = torch.load(tensors_path)

x_src = data_tensors_cp['x_src']
y_src = data_tensors_cp['y_src']
x_tgt = data_tensors_cp['x_tgt']
y_tgt = data_tensors_cp['y_tgt']

x_src = x_src.to(device)
y_src = y_src.to(device)
x_tgt = x_tgt.to(device)
y_tgt = y_tgt.to(device)
logger.append_log('loaded tensors...', x_src.shape, y_src.shape,
                  x_tgt.shape, y_tgt.shape, x_src.is_cuda)

# create data loaders
total_samples = len(x_src)
train_size = int(0.95 * total_samples)
test_size = total_samples - train_size
ds_cloze = TensorDataset(x_src, y_src)
ds_natq = TensorDataset(x_tgt, y_tgt)
ds_cloze_train, ds_cloze_test = random_split(ds_cloze, [train_size, test_size],
                                             generator=torch.Generator().manual_seed(seed))
ds_natq_train, ds_natq_test = random_split(ds_natq, [train_size, test_size],
                                           generator=torch.Generator().manual_seed(seed))

dl_cloze_train = DataLoader(ds_cloze_train, batch_size=config_dict['batch_size'], drop_last=True, num_workers=0,
                            pin_memory=False)
dl_natq_train = DataLoader(ds_natq_train, batch_size=config_dict['batch_size'], drop_last=True, num_workers=0,
                           pin_memory=False)

dl_cloze_test = DataLoader(ds_cloze_test, batch_size=1)
dl_natq_test = DataLoader(ds_natq_test, batch_size=1)

logger.append_log('train/test size', len(dl_cloze_train), len(dl_cloze_test))

input_vocab = len(word2idx)

generator = GeneratorModel(input_vocab, config_dict['hidden_dim'],
                           config_dict['batch_size'], word_emb_tensor, layers=config_dict['layers'],
                           bidirectional=config_dict['bidir'], lstm_do=config_dict['lstm_do'])

# lat_clf = LatentClassifier((2 if config_dict['bidir'] else 1) * config_dict['hidden_dim'], 
#                            1, config_dict['clf_hidden_nodes'])

clf_in_shape = max_len * (2 if config_dict['bidir'] else 1) * config_dict['hidden_dim']
lat_clf = LatentClassifier(clf_in_shape, 1, int(clf_in_shape / 1.5))
# lat_clf = LatentClassifier((2 if config_dict['bidir'] else 1)*config_dict['hidden_dim'], 1, 
#                            2*(2 if config_dict['bidir'] else 1)*config_dict['hidden_dim'])
assert not isNan(generator.encoder.emb.weight)
# generator.half(), lat_clf.half()
generator.to(device), lat_clf.to(device)
generator.apply(weights_init), lat_clf.apply(weights_init)

logger.flush()
