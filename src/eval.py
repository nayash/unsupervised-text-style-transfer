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
from torchtext.data.metrics import bleu_score
import json

'''
python eval.py --expid torchAttn_2lstm_smallData -f ../inputs/test_sentences.txt --cpfile data_cp-1.pk
python eval.py --expid 1dirDec_300emb_largData_max8 -f ../inputs/test_sentences.txt --cpfile data_cp8.pk
python eval.py --expid fr_en-learnableEmbDim-pt1DO -f ../inputs/test_sentences_fr.txt --cpfile data_cp15.pk
python eval.py --expid st-yelp_freezeEmb -f ../inputs/test_sentences.txt --cpfile ../outputs/runs/st-yelp_freezeEmb/data_cp10.pk --model best_modelG_2.pt
python eval.py --expid st-yelp_freezeEmb-attn -f ../inputs/test_sentences.txt --cpfile ../outputs/runs/st-yelp_freezeEmb/data_cp10.pk --model best_modelG_1.pt --config '../inputs/config.json'
python eval.py --expid st-yelp_frzEmb-hidDimDiff -f ../inputs/test_sentences.txt --model best_modelG_14.pt --config '../inputs/config.json' --cpfile '../outputs/runs/st-yelp_frzEmb-hidDimDiff/data_cp10.pk'

/home/asutosh/Documents/ml_projects/unsupervised-text-style-transfer/outputs/runs/1dirDec_300emb_largData_max8
'''

ROOT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).parent
INPUT_PATH = ROOT_PATH / 'inputs'
OUTPUT_PATH = ROOT_PATH / 'outputs'
GLOVE_PATH = INPUT_PATH / 'word_embs' / 'cc.en.300.vec'
YELP_PATH = INPUT_PATH / 'yelp-reviews-preprocessed'
src_sents = 'sentiment.0.all.txt'
tgt_sents = 'sentiment.1.all.txt'

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--expid',
                        help='experiment id for the model to evaluate')
arg_parser.add_argument('-f',
                        help='text file with sentences to evaluate the model'
                             ' with')
arg_parser.add_argument('--cleanfunc', default='clean_text_yelp',
                        help='text cleaning function. use the same function '
                             'used for treating training data. for e.g. see '
                             '"clean_text_yelp" func in utils.py. Default = clean_text_yelp')
arg_parser.add_argument('--cpfile',
                        help='checkpoint file path which holds training vocabulary, '
                             'embeddings etc. default=data_cp8.pk')
arg_parser.add_argument('--evaltype', default='forward',
                        help='translate from source to target if value = "forward",'
                             'else reverse.')
arg_parser.add_argument('--device', default='cuda')
arg_parser.add_argument('--model',
                        help='file name of saved best model to evaluate')
arg_parser.add_argument('--config',
                        help='configuration/hyperparameters in json format.'
                             ' Default is the config saved with "cpfile"')

args = arg_parser.parse_args()
eval_file_path = os.path.abspath(args.f)
run_id = args.expid
device = 'cuda:0' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
clean_text_func = locals()[args.cleanfunc]
run_path = OUTPUT_PATH / 'runs' / run_id
data_cp_path = Path(os.path.abspath(args.cpfile))  # run_path / args.cpfile
resume_history = run_path / 'state.pt'
best_model_path = run_path / args.model
mode = src2tgt if args.evaltype == 'forward' else tgt2src
# tensors_path = OUTPUT_PATH / ('data_tensors_cp'+str(max_len)+'.pt')
print('best_model_path', best_model_path)
data_cp = pickle.load(open(str(data_cp_path), 'rb'))

if not args.config:
    config_dict = data_cp['config_dict']
else:
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as file:
        _ = file.read()
        config_dict = json.loads(_)

max_len = data_cp['max_sent_len']
max_len += 3  # extra tokens for SOSOpType, EOS, PAD
word2idx_src = data_cp['word2idx_src']
idx2word_src = data_cp['idx2word_src']
word_emb_src = data_cp['word_emb_src']
word2idx_tgt = data_cp['word2idx_tgt']
idx2word_tgt = data_cp['idx2word_tgt']
word_emb_tgt = data_cp['word_emb_tgt']

word2idx = word2idx_src if mode == src2tgt else word2idx_tgt
idx2word = idx2word_tgt if mode == src2tgt else idx2word_src


def sent_to_tensor(sentence, word2idx=word2idx, max_len=max_len, type=mode,
                   prefix=None):
    temp = []
    sos = word2idx[SOS_SRC] if type == src2tgt else word2idx[SOS_TGT]
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
    max_len = max([len(line.split()) for line in lines]) + 3
    tensors = [sent_to_tensor(sent, max_len=max_len) for sent in lines]
    tensors = torch.stack(tensors)

word_emb_src = torch.tensor(word_emb_src)
word_emb_src = word_emb_src / torch.norm(word_emb_src, dim=1).unsqueeze(-1)

word_emb_tgt = torch.tensor(word_emb_tgt)
word_emb_tgt = word_emb_tgt / torch.norm(word_emb_tgt, dim=1).unsqueeze(-1)

generator = GeneratorModel(len(word2idx_src), len(word2idx_tgt),
                           config_dict['hidden_dim'],
                           config_dict['embedding_dim'],
                           config_dict['batch_size'], word_emb_src,
                           word_emb_tgt, device,
                           layers_gen=config_dict['layers_gen'],
                           layers_dec=config_dict['layers_dec'],
                           bidir_gen=config_dict['bidir_gen'],
                           bidir_dec=config_dict['bidir_dec'],
                           lstm_do=config_dict['lstm_do'],
                           use_attn=config_dict['use_attention'],
                           emb_do=config_dict['emb_do'])  # no word_dropout needed

# state = torch.load(resume_history, map_location='cpu')
try:
    generator.load_state_dict(torch.load(best_model_path))
except Exception as e:
    print(e)
    best_model_dir = best_model_path.parts[-2]
    data_cp_dir = data_cp_path.parts[-2]
    if best_model_dir != data_cp_dir:
        print(
            'ERROR : cpfile (the checkpoint file which contains model configurations)'
            ' belongs to expid "{}" and best model checkpoint belongs to "{}". '
            'If error was due to weight mismatch, please verify if the model '
            'weights/configuration is same for both expids.'.
            format(data_cp_dir, best_model_dir))
    sys.exit(0)

generator.to(device)
tensors.to(device)
generator.eval()


def tensor_to_sentence(sent_tensor, idx2word=idx2word):
    sent_tensor = sent_tensor.squeeze().cpu().numpy()
    sent = []
    for idx in sent_tensor:
        sent.append(idx2word[idx])
    return ' '.join(sent)


def eval_model_tensor(generator, x, mode, batch_size=100):
    x = x.to(device)
    generator.set_mode(mode, word2idx_src, word2idx_tgt)
    input = x if len(x.size()) > 1 \
        else x.view(1, -1)
    sents = []
    batches = (len(x) - 1) // batch_size + 1
    for i in range(batches):
        gen_out, gen_raw, enc_out = generator(
            input[i * batch_size: (i + 1) * batch_size])
        sents.extend([tensor_to_sentence(row) for row in gen_out])
    return sents


result = eval_model_tensor(generator, tensors, mode)
candidate_corpus = []
ref_corpus = []
for i, line in enumerate(lines):
    print('\n', line.strip(), '-->', ' '.join([w for w in result[i].split(' ')
                                               if w not in [SOS_TGT, SOS_SRC,
                                                            'EOS', 'PAD']]))
    candidate_corpus.append(word_tokenize(result[i]))
    ref_corpus.append(word_tokenize(line))
    # print('bleu', bleu_score(candidate_corpus, ref_corpus))
