from models import GeneratorModel
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
import math
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch.autograd.profiler as profiler

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
                        help='if passed then the data clean up processing \
                        is done again instead of of using the saved data \
                        checkpoints')
arg_parser.add_argument('-r', '--resume', default=False, action='store_true',
                        help='if passed then training is resumed from saved states.'
                             'please note that with this option you must pass existing "expid" argument')
arg_parser.add_argument('--test', default=False, action='store_true',
                        help='short test run')

args = arg_parser.parse_args()
source_file_path = os.path.abspath(args.insrc)
target_file_path = os.path.abspath(args.intgt)
config_path = os.path.abspath(args.config)
run_id = args.expid
force_preproc = args.force
is_resume = args.resume

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
tgt_label = 1
src_label = 0


with open(source_file_path, encoding='utf-8',
          errors='ignore') as file:
    src_sents = file.readlines()

with open(target_file_path, encoding='utf-8',
          errors='ignore') as file:
    tgt_sents = file.readlines()

if not run_path.exists():
    os.makedirs(run_path)
else:
    if run_id != 'temp' and not is_resume:
        raise Exception('expid already exists. please pass unique expid or pass "-r" argument')

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
        logger.append_log('Exception', e, word2idx['EOS'], eos_index)
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
                    # logger.append_log('filled', m)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
                    # logger.append_log('filled', m)
        else:
            if not isinstance(m, nn.Embedding):
                init.xavier_normal_(m.weight.data)
                # logger.append_log('filled', m)
    except:
        # logger.append_log('failed', m)
        pass


def isNan(tensor):
    return torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item()


def standard_scaler(tensor):
    with torch.no_grad():
        means = tensor.mean(dim=0)
        stds = tensor.std(dim=0) + 1e-8
        # logger.append_log('scaler',isNan(tensor),isNan(means),isNan(stds), isNan((tensor-means)/stds))
        return (tensor - means) / stds


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def weight_clip(model, c):
    for p in model.parameters():
        p.data.clamp_(-c, c)


def log_samples_text(samples, epoch):
    text = '-------------epoch=' + str(epoch) + '-------------\n'
    for sample in samples[::int(len(samples) / 5)]:
        for i in range(len(sample[0])):
            cd1 = 'CD src2tgt =>' + sample[0][i] + '-->' + sample[1][i] + '-->' + sample[2][i]
            cd2 = 'CD tgt2src =>' + sample[3][i] + '-->' + sample[4][i] + '-->' + sample[5][i]
            text += cd1 + '\n' + cd2 + '\n\n'
        text += '\n*********************\n'
    return text


if force_preproc or not tensors_path.exists():
    logger.append_log('converting sentences to tensors...')
    x_src = []
    y_src = []
    x_tgt = []
    y_tgt = []

    permute_prob = config_dict['permute_prob']
    drop_noise = config_dict['drop_noise']

    for i, src_sent in enumerate(src_sents):
        tensor = sent_to_tensor(src_sent.strip(), word2idx, max_len, type='neg')
        x_src.append(get_noisy_tensor(tensor))
        y_src.append(tensor.clone())

    for i, q in enumerate(tgt_sents):
        tensor = sent_to_tensor(q.strip(), word2idx, max_len, type='pos')
        x_tgt.append(get_noisy_tensor(tensor))
        y_tgt.append(tensor.clone())

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
ds_src = TensorDataset(x_src, y_src)
ds_tgt = TensorDataset(x_tgt, y_tgt)
ds_src_train, ds_src_test = random_split(ds_src, [train_size, test_size],
                                         generator=torch.Generator().manual_seed(seed))
ds_tgt_train, ds_tgt_test = random_split(ds_tgt, [train_size, test_size],
                                         generator=torch.Generator().manual_seed(seed))

dl_src_train = DataLoader(ds_src_train, batch_size=config_dict['batch_size'], drop_last=True, num_workers=0,
                          pin_memory=False)
dl_tgt_train = DataLoader(ds_tgt_train, batch_size=config_dict['batch_size'], drop_last=True, num_workers=0,
                          pin_memory=False)

dl_src_test = DataLoader(ds_src_test, batch_size=1)
dl_tgt_test = DataLoader(ds_tgt_test, batch_size=1)

logger.append_log('train/test size', len(dl_src_train), len(dl_src_test))

# construct models, optimizers, losses
input_vocab = len(word2idx)
generator = GeneratorModel(input_vocab, config_dict['hidden_dim'],
                           config_dict['batch_size'], word_emb_tensor, layers=config_dict['layers'],
                           bidirectional=config_dict['bidir'], lstm_do=config_dict['lstm_do'])
clf_in_shape = max_len * (2 if config_dict['bidir'] else 1) * config_dict['hidden_dim']
lat_clf = LatentClassifier(clf_in_shape, 1, int(clf_in_shape / 1.5))
assert not isNan(generator.encoder.emb.weight)
# generator.half(), lat_clf.half()
generator.to(device), lat_clf.to(device)
generator.apply(weights_init), lat_clf.apply(weights_init)

lr_reduce_factor = 0.1
lr_reduce_patience = 2

lr_reduce_factorD = 0.1
lr_reduce_patienceD = 2

base_lr = config_dict['base_lr']
max_lr = config_dict['max_lr']
step_size = config_dict['step_size']
mode = config_dict['mode']

loss_disc = nn.BCEWithLogitsLoss()  # BCE doesn't use sigmoid internally
loss_ce = nn.NLLLoss()  # nn.CrossEntropyLoss()
# optimG = torch.optim.Adam(generator.parameters(), lr=config_dict['gen_lr'], betas=(0.5, 0.999))
optimG = torch.optim.RMSprop(generator.parameters(), lr=config_dict['gen_lr'], alpha=0.99, eps=1e-08,
                             weight_decay=0, momentum=0.9, centered=False)
# optimD = torch.optim.Adam(lat_clf.parameters(), lr=config_dict['clf_lr'], betas=(0.5, 0.999))
optimD = torch.optim.RMSprop(lat_clf.parameters(), lr=config_dict['clf_lr'], alpha=0.99, eps=1e-08,
                             weight_decay=0, momentum=0.9, centered=False)

if config_dict['gen_lr_sched'] == 'cyclic':
    lr_sched_G = torch.optim.lr_scheduler.CyclicLR(optimG, base_lr, max_lr, step_size_up=step_size,
                                                   step_size_down=None, mode=mode, gamma=1.0,
                                                   scale_mode='cycle', last_epoch=-1)
else:
    lr_sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimG, mode='min',
                                                            factor=lr_reduce_factor, patience=lr_reduce_patience,
                                                            verbose=True, threshold=0.00001, threshold_mode='rel',
                                                            cooldown=0, min_lr=1e-8, eps=1e-08)

if config_dict['disc_lr_sched'] == 'cyclic':
    lr_sched_D = torch.optim.lr_scheduler.CyclicLR(optimD, 1e-6, 1e-5, step_size_up=2000,
                                                   step_size_down=None, mode='triangular', gamma=1.0,
                                                   scale_mode='cycle', last_epoch=-1)
else:
    lr_sched_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimD, mode='min',
                                                            factor=lr_reduce_factorD, patience=lr_reduce_patienceD,
                                                            verbose=True, threshold=0.00001, threshold_mode='rel',
                                                            cooldown=0, min_lr=1e-8, eps=1e-08)


# training code

epochs = config_dict['epoch']
train_lossesG = []
train_lossesD = []
start_time = time.time()
prev_best_loss = np.inf
early_stop_patience = 30  # lr_reduce_patience * 3
early_stop_counter = 0
scaler = StandardScaler()
skip_disc = False
disc_noise_prob = 0.2

lambda_auto = config_dict['wgt_loss_auto']
lambda_cd = config_dict['wgt_loss_cd']
lambda_adv = config_dict['wgt_loss_adv'] if not skip_disc else 0

test_mode = args.test
resume_history = run_path / 'state.pt'
resume_epoch = 0
if is_resume:
    state = torch.load(resume_history, map_location='cpu')

    resume_epoch = state['epoch']
    generator.load_state_dict(state['modelG'])
    lat_clf.load_state_dict(state['modelD'])
    optimG.load_state_dict(state['optim_stateG'])
    optimD.load_state_dict(state['optim_stateD'])
    lr_sched_G.load_state_dict(state['lr_sched_g'])
    lr_sched_D.load_state_dict(state['lr_sched_d'])
    prev_best_loss = state['last_val_loss']

    generator.to(device)
    lat_clf.to(device)
    lr_reduce_patience = 10
    early_stop_patience = 50
    logger.append_log('lr_reduce_patience and early_stop_patience changed', lr_reduce_patience, early_stop_patience)
    logger.append_log('loaded previous saved model')

writer = SummaryWriter(run_path, max_queue=1, flush_secs=1)
# writer.add_text('run_changes', run_changes, lambda_adv)
logger.append_log('training start time', time.ctime())

src_label_vec = torch.tensor([src_label] * config_dict['batch_size'], requires_grad=False).float().to(device)
tgt_label_vec = torch.tensor([tgt_label] * config_dict['batch_size'], requires_grad=False).float().to(device)

generator.train()
lat_clf.train()

# scaler = GradScaler()

zero_tensor = torch.tensor(0)

for epoch in range(resume_epoch, epochs):
    epoch_loss_G = []
    epoch_loss_D = []
    epoch_start_time = time.time()
    samples = []
    for iter_no, (data_src, data_tgt) in enumerate(zip(dl_src_train, dl_tgt_train)):
        with profiler.profile(record_shapes=True, profile_memory=True, use_cuda=True, enabled=False) as prof:
            iter_start_time = time.time()

            in_src, in_tgt = data_src[0], data_tgt[0]  # noisy, torch.Size([50, 21]) torch.Size([50])
            org_src, org_tgt = data_src[1], data_tgt[1]  # non-noisy original sentence tensors

            optimD.zero_grad()
            optimG.zero_grad()

            # with autocast():

            # 1. auto-encode
            with profiler.record_function("auto-encode"):
                ## 1.1 src to src
                generator.set_mode(src2src)
                _, gen_raw, _ = generator(in_src)
                loss_auto_src = 0
                for k in range(gen_raw.size(0)):
                    loss_auto_src += loss_ce(gen_raw[k], org_src[k])

                ## 1.2 tgt to tgt
                generator.set_mode(tgt2tgt)
                _, gen_raw1, _ = generator(in_tgt)
                loss_auto_tgt = 0
                for k in range(gen_raw1.size(0)):
                    loss_auto_tgt += loss_ce(gen_raw1[k], org_tgt[k])

                loss_auto = (loss_auto_src + loss_auto_tgt)

            # 2. cross domain
            with profiler.record_function("cross-domain"):
                ## 2.1 src to tgt
                generator.set_mode(src2tgt)
                gen_out_src, _, enc_out_src = generator(org_src)

                generator.set_mode(tgt2src)  # need to add noise to gen_our_src and preserve gradient too!!!
                gen_bt_src2tgt, gen_out_bt_raw, _ = generator(row_apply(gen_out_src, get_noisy_tensor_grad))

                loss_cd_s2t = 0
                for k in range(gen_out_bt_raw.size(0)):
                    loss_cd_s2t += loss_ce(gen_out_bt_raw[k], org_src[k])

                ## 2.2 tgt to src
                generator.set_mode(tgt2src)
                gen_out_tgt, _, enc_out_tgt = generator(org_tgt)

                generator.set_mode(src2tgt)
                gen_bt_tgt2src, gen_out_bt_raw1, _ = generator(row_apply(gen_out_tgt, get_noisy_tensor_grad))

                loss_cd_t2s = 0
                for k in range(gen_out_bt_raw1.size(0)):
                    loss_cd_t2s += loss_ce(gen_out_bt_raw1[k], org_tgt[k])

                loss_cd = (loss_cd_s2t + loss_cd_t2s)

            # 3. adversarial training
            with profiler.record_function("adverserial"):
                # if encoder source was tgt, encoder should be trained to produce latent vector close to
                # source and vice versa so pair enc_out_src with tgt_label and vice versa

                if not skip_disc:
                    enc_out_src = enc_out_src.reshape(enc_out_src.size(0), -1)
                    enc_out_src = standard_scaler(enc_out_src)
                    disc_out = lat_clf(enc_out_src)  # (batch_size, )
                    lossSrc = loss_disc(disc_out, tgt_label_vec)

                    enc_out_tgt = enc_out_tgt.reshape(enc_out_tgt.size(0), -1)
                    enc_out_tgt = standard_scaler(enc_out_tgt)
                    disc_out1 = lat_clf(enc_out_tgt)
                    lossTgt = loss_disc(disc_out1, src_label_vec)

                    loss_adv = (lossSrc + lossTgt)
                else:
                    loss_adv = zero_tensor
                    lossSrc = zero_tensor
                    lossTgt = zero_tensor

            lossG = lambda_auto * loss_auto + lambda_cd * loss_cd + lambda_adv * loss_adv
            # forward section end
            # end of autocast for G

            with profiler.record_function("back & step"):
                # scaler.scale(lossG).backward()
                lossG.backward()  # delay 661.956924996 secs on CPU, 1.2 secs on GPU

                # unscale optimizer before grad clipping
                # scaler.unscale_(optimG)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), config_dict['gen_grad_clip'])

                # scaler.step(optimG)
                optimG.step()

            # weight_clip(generator, 5)

            epoch_loss_G.append(lossG.item())
            samples.append((row_apply(org_src[:5], tensor_to_sentence, False),
                            row_apply(gen_out_src[:5], tensor_to_sentence, False),
                            row_apply(gen_bt_src2tgt[:5], tensor_to_sentence, False),
                            row_apply(org_tgt[:5], tensor_to_sentence, False),
                            row_apply(gen_out_tgt[:5], tensor_to_sentence, False),
                            row_apply(gen_bt_tgt2src[:5], tensor_to_sentence, False)))

            # 4. train discriminator to identify encoder outputs belonging to input type src and tgt
            with profiler.record_function("disc-train"):
                # using a combined and shuffled encoder outputs from src2tgt and tgt2src with targets as source_label
                # & tgt_label resp.

                if not skip_disc:
                    # combine and shuffle both encoder outputs
                    # with autocast():
                    generator.set_mode(src2tgt)
                    _, _, enc_out_src1 = generator(in_src)

                    generator.set_mode(tgt2src)
                    _, _, enc_out_tgt1 = generator(in_tgt)

                    # no mix train now.
                    # enc_out_mix = torch.cat((enc_out_src1, enc_out_tgt1), dim=0)
                    # disc_tgt = torch.cat((src_label_vec, tgt_label_vec))
                    # shuffle_idx = torch.randperm(enc_out_mix.size(0))
                    # enc_out_mix = enc_out_mix[shuffle_idx]
                    # disc_tgt = disc_tgt[shuffle_idx]
                    # clf_out = lat_clf(enc_out_mix.detach().reshape(enc_out_mix.size(0), -1))  # detach is not causing 0 grad
                    # lossD = loss_bce(clf_out, disc_tgt.float())

                    enc_out_src1 = enc_out_src1.detach().reshape(enc_out_src1.size(0), -1)
                    enc_out_src1 = standard_scaler(enc_out_src1)

                    enc_out_tgt1 = enc_out_tgt1.detach().reshape(enc_out_tgt1.size(0), -1)
                    if torch.isnan(enc_out_tgt1.max()).item() or torch.isinf(enc_out_tgt1.max()).item():
                        logger.append_log('check1', 'nan found')
                    enc_out_tgt1 = standard_scaler(enc_out_tgt1)  # has nan
                    if torch.isnan(enc_out_tgt1.max()).item() or torch.isinf(enc_out_tgt1.max()).item():
                        logger.append_log('check2', 'nan found')

                    # for d_ in range(1):
                    optimD.zero_grad()
                    # with autocast():
                    if np.random.uniform(0, 1) < disc_noise_prob:
                        enc_out_src1 += torch.randn(enc_out_src1.size(), device=device).uniform_(0, 1)
                        enc_out_tgt1 += torch.randn(enc_out_tgt1.size(), device=device).uniform_(0, 1)

                        clf_out1 = lat_clf(enc_out_src1)
                        clf_out2 = lat_clf(enc_out_tgt1)
                    else:
                        clf_out1 = lat_clf(enc_out_src1)
                        clf_out2 = lat_clf(enc_out_tgt1)

                    if np.random.uniform(0, 1) < disc_noise_prob:
                        # add noise to label and input
                        noisy_src_labels = src_label_vec.clone()
                        noisy_tgt_labels = tgt_label_vec.clone()
                        noisy_src_labels[
                            np.random.randint(0, src_label_vec.size(0), int(0.3 * src_label_vec.size(0)))] = tgt_label
                        noisy_tgt_labels[
                            np.random.randint(0, tgt_label_vec.size(0), int(0.3 * tgt_label_vec.size(0)))] = src_label

                        lossD_src = loss_disc(clf_out1, noisy_src_labels)
                        lossD_tgt = loss_disc(clf_out2, noisy_tgt_labels)
                    else:
                        lossD_src = loss_disc(clf_out1, src_label_vec)
                        lossD_tgt = loss_disc(clf_out2, tgt_label_vec)
                    lossD = (lossD_src + lossD_tgt)

                    # scaler.scale(lossD).backward()
                    lossD.backward()
                    # scaler.unscale_(optimD)
                    torch.nn.utils.clip_grad_norm_(lat_clf.parameters(), config_dict['disc_grad_clip'])
                    # scaler.step(optimD)
                    optimD.step()
                    # weight_clip(lat_clf, 1)
                else:
                    lossD = zero_tensor

            epoch_loss_D.append(lossD.item())

            if config_dict['gen_lr_sched'] == 'cyclic':
                lr_sched_G.step()

            if config_dict['disc_lr_sched'] == 'cyclic' and not skip_disc:
                lr_sched_D.step()

            # scaler.update()

            step = epoch * len(dl_src_train) + iter_no
            writer.add_scalar('loss/loss_auto_src', loss_auto_src.item(), step)
            writer.add_scalar('loss/loss_auto_tgt', loss_auto_tgt.item(), step)
            writer.add_scalar('loss/loss_cd_s2t', loss_cd_s2t.item(), step)
            writer.add_scalar('loss/loss_cd_t2s', loss_cd_t2s.item(), step)
            writer.add_scalar('loss/loss_adv_src', lossSrc.item(), step)
            writer.add_scalar('loss/loss_adv_tgt', lossTgt.item(), step)
            writer.add_scalar('loss/loss_disc', lossD.item(), step)
            writer.add_scalar('grad/generator', get_grad_norm(generator), step)
            if not skip_disc:
                writer.add_scalar('grad/discrim', get_grad_norm(lat_clf), step)

            if iter_no % 50 == 0:
                logger.append_log('epoch:{}/{} | iter:{}/{} | train_lossG={}, train_lossD={}, duration={} secs'.
                      format(epoch, epochs, iter_no, len(dl_src_train), np.mean(epoch_loss_G),
                             np.mean(epoch_loss_D), (time.time() - iter_start_time)))
                if test_mode:
                    break

            # end of iter loop
        # end of profiler

    train_lossesD.append(np.mean(epoch_loss_D))
    train_lossesG.append(np.mean(epoch_loss_G))

    val_loss = train_lossesG[-1]  # (train_lossesD[-1]+train_lossesG[-1])/2  # eval_model(clf_model, test_dl)[0]

    if config_dict['gen_lr_sched'] != 'cyclic':
        lr_sched_G.step(train_lossesG[-1])

    if config_dict['disc_lr_sched'] != 'cyclic' and not skip_disc:
        lr_sched_D.step(train_lossesD[-1])

    if val_loss < prev_best_loss:
        prev_best_loss = val_loss
        state = {'epoch': epoch + 1,
                 'modelG': generator.state_dict(),
                 'modelD': lat_clf.state_dict(),
                 'optim_stateD': optimD.state_dict(),
                 'optim_stateG': optimG.state_dict(),
                 'lr_sched_g': lr_sched_G.state_dict(),
                 'lr_sched_d': lr_sched_D.state_dict(),
                 'last_train_loss': train_lossesG[-1],
                 'last_val_loss': val_loss
                 }

        torch.save(state, run_path + '/state.pt')
        if is_resume:
            # lr_reduce_patience = 2
            # early_stop_patience = lr_reduce_patience * 3
            # logger.append_log('lr and early_stop patience reverted', lr_reduce_patience, early_stop_patience)
            pass

        logger.append_log('new best loss {}. State saved.'.format(val_loss))
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        logger.append_log('early stop counter increament to', early_stop_counter, '/', early_stop_patience)

    epoch_summary = 'epoch:{} | train_lossG={}, train_lossD={}, val_loss={}, duration={} secs'.\
        format(epoch, train_lossesG[-1], train_lossesD[-1], val_loss, (time.time() - epoch_start_time))
    logger.append_log(epoch_summary)
    writer.add_text('epoch_summary', epoch_summary, global_step=epoch)
    if prof:
        logger.append_log(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        break

    # logger.append_log some samples
    if epoch % 2 == 0:
        try:
            logger.append_log('samples:\n', log_samples_text(samples, epoch), '\n')
            with open(run_path + '/samples.txt', 'a') as file:
                file.write(log_samples_text(samples, epoch))
        except:
            logger.append_log('log failed')

#     if early_stop_counter > early_stop_patience:
#         logger.append_log('stopping early at {} epoch and loss {}'.format(epoch, val_loss))
#         break

logger.append_log('training finished in {} mins'.format((time.time() - start_time) / 60))

logger.flush()
