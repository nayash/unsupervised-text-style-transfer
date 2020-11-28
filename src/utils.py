#
# Copyright (c) 2020. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import sys
sys.path.append('../src')
import re
import calendar
import unicodedata
from collections import Counter

import numpy as np
import torch
import time
from nltk.tokenize import sent_tokenize, word_tokenize
from constants import *
import multiprocessing as mp


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[\"\'\`\~\#\$\%\&\^\*\“\”\’\‘]', '', text)
    text = re.sub(r'[-—_,]', ' ', text)
    text = re.sub(r'((\?+)|(\!+)|(;+)|(\.+))', '.', text)
    text = re.sub(r'[()]', '', text)
    # text = re.sub(r'\d{1,2}:\d{2} (p\.\s?m\.|a\.\s?m\.)?', '<time>', text)
    text = re.sub(r"\n+\s*\n+", ' <p> ', text)
    text = re.sub(r"\s+", ' ', text)
    text = re.sub(r'to\smorrow', 'tomorrow', text)

    month_names = '|'.join([calendar.month_name[i].lower()
                            for i in range(1, 13)])
    day_regex = r'\d{1,2}(st|nd|rd|th)'
    date_regex = day_regex+'\s('+month_names+')'

    text = re.sub(date_regex, '<date>', text)
    text = re.sub(r'\d+', '<number>', text)

    return text


def clean_text_yelp(text):
    # text = ''.join(text)
    text = text.lower()
    text = ' '.join(word_tokenize(text))
    text = text.replace('_num_', 'NUMBER')
    text = text.replace('can\'t', 'can not')
    text = text.replace(" n't", ' not')
    text = text.replace(" 's", ' is')
    text = text.replace(" 've", ' have')
    text = text.replace(" ve ", ' have')
    text = text.replace(" 'd", ' would')
    text = text.replace(" 'm", ' am')
    text = text.replace(" m ", ' am ')
    text = text.replace("&", 'and')
    text = text.replace('\\n', ' ').replace("\\","")
    text = re.sub(r'[\"\'\`\~\#\$\%\&\+\^\*\“\”\’\‘\:]', ' ', text)
    text = re.sub(r'[-—_,]', ' ', text)
    text = re.sub(r'((\?+)|(\!+)|(;+)|(\.+))', '.', text)
    text = re.sub(r'[()]', '', text)
    text = re.sub(r'(\.\s?\.)', '.', text)
    # text = re.sub(r'\d{1,2}:\d{2} (p\.\s?m\.|a\.\s?m\.)?', '<time>', text)
    # text = re.sub(r"\n+\s*\n+", ' <p> ', text)
    text = re.sub(r"\s+", ' ', text)
    text = re.sub(r'to\smorrow', 'tomorrow', text)

    month_names = '|'.join([calendar.month_name[i].lower()
                            for i in range(1, 13)])
    day_regex = r'\d{1,2}(st|nd|rd|th)'
    date_regex = day_regex+'\s('+month_names+')'

    text = re.sub(date_regex, 'DATE', text)
    text = re.sub(r'\d+', 'NUMBER', text)
    return text


def get_subclauses(sent):
    clauses = []
    for word in sent:
        if word.dep_ in ('xcomp', 'ccomp'):
            subtree_span = sent[word.left_edge.i: word.right_edge.i + 1]
            clauses.append(subtree_span.text)
    if len(clauses) == 0:
        clauses.append(sent.text)
    return clauses


def get_vocab_dicts(words):
    word2idx = {'SOS': len(words)+0, 'EOS': len(words)+1,
                'PAD': len(words)+2, 'UNK': len(words)+3}
    idx2word = {v: k for k, v in word2idx.items()}

    for w in words:
        word2idx[w] = len(word2idx)
        idx2word[len(idx2word)] = w

    return word2idx, idx2word


def vocab_from_pretrained_emb(emb_path, words, start=0, end=0, batch_num=0,
                              batch_size=0, emb_dim=-1):
    word2idx = {}
    idx2word = {}
    word_emb = []
    offset = (batch_size*batch_num)
    if not emb_path == 'random':
        with open(emb_path) as file:
            for i, line in enumerate(file):
                split = line.split()
                word = split[0]
                if word not in words:
                    continue
                emb = split[1:]
                idx = len(word_emb)+offset
                word2idx[word] = idx
                idx2word[idx] = word
                word_emb.append([float(i) for i in emb])
    else:
        assert emb_dim > 0, "if embedding vectors file is not provided, " \
                            "embedding dimension has to be passed explicitly."
        for word in words:
            idx = len(word_emb) + offset
            word2idx[word] = idx
            idx2word[idx] = word
            word_emb.append(np.random.randn(emb_dim))

    return word2idx, idx2word, word_emb


def vocab_from_pretrained_emb_parallel(emb_path, words, pool, workers=3,
                                       emb_dim=-1):
    word2idx = {}
    idx2word = {}
    word_emb = []
    results = []

    offset = len(words) // workers
    for i in range(workers):
        s = i*offset
        e = s+offset-1
        results.append(pool.apply_async(vocab_from_pretrained_emb,
                                   args=[emb_path, words[s:e+1], s, e, i,
                                         offset, emb_dim]))
    [res.wait() for res in results]
    prev = None
    # print('words', len(words))
    for i, res in enumerate(results):
        _ = res.get()
        if prev:
            prev_len = len(word2idx)
            diff = i*offset - prev_len
            _[1].clear()
            temp = _[0].copy()
            _[0].clear()
            for k, v in temp.items():  # word2idx
                _[0][k] = v-diff
                _[1][v-diff] = k  # idx2word

        word2idx.update(_[0])
        idx2word.update(_[1])
        word_emb.extend(_[2])
        prev = _[0]

    # now add words from corpus which are missing in Glove embeddings
    diff = list(set(words).difference(word2idx.keys()))
    for word in diff:
        word2idx[word] = len(word_emb)
        if len(word_emb) in idx2word:
            raise Exception("word index already exists", len(word_emb),
                            idx2word[len(word_emb)], word)
        idx2word[len(word_emb)] = word
        word_emb.append(np.random.uniform(0, 1, len(word_emb[-1])))
    return word2idx, idx2word, word_emb


def roll_prepend(tensor, prefix_num, dim=1):
    b = torch.roll(tensor, 1, dim)
    b[:, 0] = prefix_num
    return b


def permute_items(collection, k=3):
    q = np.random.uniform(0, k+1, len(collection)) + np.arange(len(collection))
    return list(np.array(collection)[np.argsort(q)])


# def permute_tensor(tensor, k=3):
#     c = torch.FloatTensor(tensor.size(0)).uniform_(
#         0, k+1)+torch.arange(tensor.size(0))
#     return tensor[torch.argsort(c)]


def permute_tensor(tensor, k=3):
    c = torch.FloatTensor(tensor.size(1)).uniform_(
        0, k+1)+torch.arange(tensor.size(1))
    return tensor[:, torch.argsort(c)]


def get_readable_ctime():
    return time.strftime("%d-%m-%Y %H_%M_%S")


# small functions to enable parallelization
def is_sent_shorter(sent, max_len):
    return len(word_tokenize(sent)) <= max_len


def len_word_tokenize(sent):
    return len(word_tokenize(sent))


def clean_text_wrapper(text, clean_text_func, max_len):
    # print(text, 'func=',clean_text_func, 'max_len',max_len)
    sent = clean_text_func(text)
    is_short = is_sent_shorter(sent, max_len) if max_len > 0 else True
    return sent if is_short else None


def iter_apply(func, coll, kwargs):
    res = []
    for item in coll:
        res.append(func(item, **kwargs))
    return res


def parallelize(func, collection, pool, parts, **kwargs):
    batch_size = len(collection)//parts
    results = []
    output = []

    for i in range(parts):
        s = i*batch_size
        e = s+batch_size-1
        results.append(pool.apply_async(iter_apply, (func, collection[s:e+1], kwargs)))

    for res in results:
        output.extend(res.get())

    return output


def tensor_to_sentence(sent_tensor, idx2word):
    sent_tensor = sent_tensor.squeeze().cpu().numpy()
    sent = []
    for idx in sent_tensor:
        sent.append(idx2word[idx])
    return ' '.join(sent)


def sent_to_tensor(sentence, **kwargs):
    word2idx = kwargs['word2idx'] if 'word2idx' in kwargs else word2idx
    max_len = kwargs['max_len'] if 'max_len' in kwargs else max_len
    prefix = kwargs['prefix'] if 'prefix' in kwargs else None
    shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else False
    type = kwargs['type'] if 'type' in kwargs else 'src'
    dropout = kwargs['dropout'] if 'dropout' in kwargs else False
    word_dropout = kwargs['word_dropout'] if 'word_dropout' in kwargs else 0
    shuffle_prob = kwargs['shuffle_prob'] if 'shuffle_prob' in kwargs else 0

    temp = []
    sos = word2idx[SOS_SRC] if type == 'src' else word2idx[SOS_TGT]
    temp.append(sos)
    if prefix:
        for _ in prefix.split():
            temp.append(word2idx[_])
    words = word_tokenize(sentence.strip())

    if dropout and np.random.uniform(0, 1) < word_dropout:
        drop_idx = np.random.randint(len(words))
        # don't drop NER mask token
        if not words[drop_idx].isupper() and \
                not words[drop_idx] == '.' and not words[drop_idx] == '?':
            words = words[:drop_idx] + words[drop_idx + 1:]

    if shuffle and np.random.uniform(0,1) < shuffle_prob:
        words = permute_items(words, k=4)

    temp.extend([word2idx.get(w, word2idx['UNK']) for w in words])
    temp.append(word2idx['EOS'])
    temp.extend([word2idx['PAD']] * (max_len - len(temp)))
    return torch.tensor(temp)


def vocab_from_sents(sents, pool, extra_tokens, GLOVE_PATH=None, emb_dim=-1):
    words = pool.map(word_tokenize, sents)
    words = [w for l in words for w in l]
    count_dict = Counter(words)
    words = [w for w in count_dict.keys() if count_dict[w] > 2]
    words = list(set(words))
    words.extend(extra_tokens)
    word2idx, idx2word, word_emb = vocab_from_pretrained_emb_parallel(
        GLOVE_PATH, words, pool, mp.cpu_count(), emb_dim=emb_dim)

    return word2idx, idx2word, word_emb
