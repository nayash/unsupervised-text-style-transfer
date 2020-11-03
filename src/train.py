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
INPUT_PATH = ROOT_PATH/'inputs'
OUTPUT_PATH = ROOT_PATH/'outputs'
GLOVE_PATH = INPUT_PATH/'glove.6B.200d.txt'
YELP_PATH = INPUT_PATH/'yelp-reviews-preprocessed'
src_sents = 'sentiment.0.train.txt'
tgt_sents = 'sentiment.1.train.txt'

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-s', '--insrc', default=YELP_PATH/src_sents,
                        help='path for source data file with each source \
                        sentence in a new line. Default is yelp negative')

arg_parser.add_argument('-t', '--intgt', default=YELP_PATH/tgt_sents,
                        help='path for target data file with each target \
                        sentence in a new line. Default is yelp dataset')
arg_parser.add_argument('-c', '--config', default=INPUT_PATH/'config.json',
                        help='configuration/hyperparameters in json format')
arg_parser.add_argument('-e', '--expid', default='temp',
                        help='identifier for track your experiment. This is\
                        used to create log files folder. All files specific \
                        to this experiment will be stored in a specific \
                        folder. If passed run_id already exists, exception \
                        is be thrown. Use special "temp" for test runs.')
arg_parser.add_argument('-f', '--force', default=False,
                        help='if True then the data clean up processing \
                        is done again instead of of using the saved data \
                        checkpoints')


args = arg_parser.parse_args()
source_file_path = os.path.abspath(args.insrc)
target_file_path = os.path.abspath(args.intgt)
config_path = os.path.abspath(args.config)
run_id = args.expid
force_preproc = args.force

run_path = OUTPUT_PATH/'runs'/run_id
log_path = run_path/'logs'
data_cp_path = OUTPUT_PATH/'data_cp.pk'

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
print(config_dict)

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

    data_cp = {'tgt': tgt_sents, 'src': src_sents, 'words': words}
    max_len = max([len(word_tokenize(sent)) for sent in src_sents+tgt_sents])
    data_cp['max_sent_len'] = max_len
    pickle.dump(data_cp, open(str(data_cp_path), 'wb'))
    logger.append_log('saved cleaned up sentences at {}, file size {}\
     KB'.format(str(data_cp_path), data_cp_path.stat().st_size//1024))
else:
    data_cp = pickle.load(open(str(data_cp_path), 'rb'))
    tgt_sents = data_cp['tgt']
    src_sents = data_cp['src']
    words = data_cp['words']
    max_len = data_cp['max_len']
    max_len += 3  # extra tokens for SOS, EOS, OpType
    print(len(src_sents), len(tgt_sents), len(my_words), max_len)

logger.append_log('number of samples in yelp negative and positive are\
 {}, {}'.format(len(src_sents), len(tgt_sents)))


src2tgt = 'C2NQ'
tgt2src = 'NQ2C'
src2src = 'C2C'
tgt2tgt = 'NQ2NQ'

SOS_SRC = 'SOSSRC'
SOS_TGT = 'SOSTGT'

# 'SOS', src2tgt, tgt2src, src2src, tgt2tgt
extra_tokens = [SOS_SRC, SOS_TGT, 'EOS', 'PAD', 'UNK']
my_words.extend(extra_tokens)
word2idx, idx2word, word_emb, diff = vocab_from_pretrained_emb(
    GLOVE_PATH, my_words)
word_emb_tensor = torch.tensor(word_emb)


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
            words = words[:drop_idx]+words[drop_idx+1:]

    if shuffle:
        words = permute_items(words, k=4)

    temp.extend([word2idx.get(w, word2idx['UNK']) for w in words])
    temp.append(word2idx['EOS'])
    temp.extend([word2idx['PAD']] * (max_len-len(temp)))
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
        eos_index = tensor.size(0)-1
    proper_tensor = tensor[1:eos_index]
    final = torch.zeros(tensor.size())
    final[:] = word2idx['PAD']
    final[0] = tensor[0]

    drop_mask = torch.FloatTensor(proper_tensor.size(0)).uniform_(0, 1)
    res = proper_tensor[drop_mask > drop_prob]

    res = permute_tensor(res)
    final[1:res.size(0)+1] = res
    final[res.size(0)+1] = word2idx['EOS']
    return final.long()


def get_noisy_tensor_grad(tensor, drop_prob=0.1, k=3, word2idx=word2idx):
    try:
        try:
            eos_index = (tensor == word2idx['EOS']).nonzero(
                as_tuple=False)[0][0]
        except:
            eos_index = tensor.size(0)-1

        tensor[1:eos_index] = tensor[1:eos_index][torch.randperm(eos_index-1)]
        # drop_mask = torch.FloatTensor(tensor[1:eos_index].size(0)).uniform_(0, 1)
    except Exception as e:
        print('Exception', e, word2idx['EOS'], eos_index)
    return tensor


def row_apply(tensor, func, is_stack=True):
    res = []
    for row in tensor:
        res.append(func(row))
    return torch.stack(res) if is_stack else res


logger.flush()
