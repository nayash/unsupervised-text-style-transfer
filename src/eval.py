from models import *
import os
import pickle
from pathlib import Path
from nltk.tokenize import sent_tokenize, word_tokenize
import argparse
from tqdm.auto import tqdm
from logger import Logger
from utils import *
import torch
from constants import *
from constants import *

ROOT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent
INPUT_PATH = ROOT_PATH / 'inputs'
OUTPUT_PATH = ROOT_PATH / 'outputs'
GLOVE_PATH = INPUT_PATH / 'glove.6B.200d.txt'
YELP_PATH = INPUT_PATH / 'yelp-reviews-preprocessed'
src_sents = 'sentiment.0.all.txt'
tgt_sents = 'sentiment.1.all.txt'

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--expid', help='experiment id for the model to evaluate')
arg_parser.add_argument('-f', help='file with sentences to evaluate the model'
                                   'with')
arg_parser.add_argument('--cleanfunc', default='clean_text_yelp',
                        help='text cleaning function. use the same function '
                             'used for treating training data. for e.g. see '
                             '"clean_text" func in utils.py. Default = clean_text_yelp')
arg_parser.add_argument('--cpfile', default='data_cp8.pt',
                        help='checkpoint file path which holds training vocabulary, '
                             'embeddings etc. default=data_cp8.pt')
arg_parser.add_argument('--evaltype', default='forward',
                        help='translate from source to target if value = "forward",'
                             'else reverse.')

args = arg_parser.parse_args()
eval_file_path = os.path.abspath(args.f)
run_id = args.expid
clean_text_func = locals()[args.cleanfunc]
run_path = OUTPUT_PATH / 'runs' / run_id
data_cp_path = OUTPUT_PATH / args.cpfile
resume_history = run_path / 'state.pt'
mode = src2tgt if args.evaltype == 'forward' else tgt2src
# tensors_path = OUTPUT_PATH / ('data_tensors_cp'+str(max_len)+'.pt')

data_cp = pickle.load(open(str(data_cp_path), 'rb'))
# tgt_sents = data_cp['tgt']
# src_sents = data_cp['src']
# words = data_cp['words']
max_len = data_cp['max_sent_len']
max_len += 3  # extra tokens for SOSOpType, EOS, PAD
print('max_len with extra_tokens=', max_len)
word2idx = data_cp['word2idx']
idx2word = data_cp['idx2word']
word_emb = data_cp['word_emb']
config_dict = data_cp['config_dict']


def sent_to_tensor(sentence, word2idx=word2idx, max_len=max_len, type=mode,
                   prefix=None):
    temp = []
    sos = word2idx[SOS_SRC] if type == 'forward' else word2idx[SOS_TGT]
    temp.append(sos)
    if prefix:
        for _ in prefix.split():
            temp.append(word2idx[_])
    words = word_tokenize(sentence.strip())

    temp.extend([word2idx.get(w, word2idx['UNK']) for w in words])
    temp.append(word2idx['EOS'])
    temp.extend([word2idx['PAD']] * (max_len - len(temp)))
    return torch.tensor(temp)  # , device=device


with open(eval_file_path, encoding='utf-8', errors='ignore') as file:
    lines_ = file.readlines()
    lines = [clean_text_func(line) for line in lines_]
    tensors = [sent_to_tensor(sent) for sent in lines]
    tensors = torch.stack(tensors)

word_emb_tensor = torch.tensor(word_emb)
word_emb_tensor = word_emb_tensor/torch.norm(word_emb_tensor, dim=1).unsqueeze(-1)

generator = GeneratorModel(len(word2idx), config_dict['hidden_dim'],
                           config_dict['batch_size'], word_emb_tensor, 'cuda',
                           layers=config_dict['layers'],
                           bidirectional=bool(config_dict['bidir']),
                           lstm_do=config_dict['lstm_do'],
                           use_attn=config_dict['use_attention'],
                           emb_do=config_dict['emb_do'])


state = torch.load(resume_history, map_location='cpu')
generator.load_state_dict(state['modelG'])
device = next(generator.parameters()).device
generator.to(device)
tensors.to(device)


def tensor_to_sentence(sent_tensor, idx2word=idx2word):
    sent_tensor = sent_tensor.squeeze().cpu().numpy()
    sent = []
    for idx in sent_tensor:
        sent.append(idx2word[idx])
    return ' '.join(sent)


def eval_model_tensor(generator, x, mode, word2idx):
    generator.set_mode(mode, word2idx)
    input = x if len(x.size()) > 1 \
        else x.view(1, -1)
    gen_out, gen_raw, enc_out = generator(input)
    # loss = 0
    # for k in range(gen_raw.size(0)):
    #     loss += loss_ce(gen_raw[k], y[k])
    sents = [tensor_to_sentence(row)for row in gen_out]
    return sents


result = eval_model_tensor(generator, tensors, mode, word2idx)
for i, line in enumerate(lines_):
    print(line, '-->', result[i])
