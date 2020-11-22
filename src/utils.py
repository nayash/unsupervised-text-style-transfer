#
# Copyright (c) 2020. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import re
import calendar
import unicodedata
import numpy as np
import torch
import time
from nltk.tokenize import sent_tokenize, word_tokenize


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


def vocab_from_pretrained_emb(emb_path, words, start=0, end=0):
    word2idx = {}
    idx2word = {}
    word_emb = []
    count = start
    with open(emb_path) as file:
        for i, line in enumerate(file):
            split = line.split()
            word = split[0]
            if word not in words:
                continue
            emb = split[1:]
            word2idx[word] = count+start
            idx2word[count+start] = word
            word_emb.append([float(i) for i in emb])
            count += 1

    # print('start-end', start, end)
    # print('thread lens', len(word2idx), len(idx2word), len(word_emb),
    #       list(word2idx.values())[0], list(word2idx.values())[-1], list(idx2word.keys())[0],
    #       list(idx2word.keys())[-1])
    return word2idx, idx2word, word_emb


def vocab_from_pretrained_emb_parallel(emb_path, words, pool, extra_tokens=[],
                                       workers=3):
    word2idx = {}
    idx2word = {}
    word_emb = []
    results = []

    offset = len(words) // workers
    for i in range(workers):
        s = i*offset
        e = s+offset-1
        results.append(pool.apply_async(vocab_from_pretrained_emb,
                                   args=[emb_path, words[s:e+1], s, e]))
    [res.wait() for res in results]
    for res in results:
        _ = res.get()
        # print('post-proc', list(_[0].values())[0], list(_[0].values())[-1], list(_[1].keys())[0],
        #   list(_[1].keys())[-1], len(_[2]))
        word2idx.update(_[0])
        idx2word.update(_[1])
        word_emb.extend(_[2])

    # now add words from corpus which are missing in Glove embeddings
    diff = set(words).difference(word2idx.keys())
    # print('words not found in Glove', len(diff), len(word2idx), len(idx2word), len(word_emb))
    # print(list(diff)[:100])
    idx = max(max(word2idx.values()), max(idx2word.keys()))+1
    for word in extra_tokens:
        word2idx[word] = idx
        if idx in idx2word:
            raise Exception("word index already exists", idx,
                            idx2word[idx], word)
        idx2word[idx] = word
        word_emb.append(np.random.uniform(0, 1, len(word_emb[-1])))
        idx += 1
    # print('after adding extra words', len(word2idx), len(idx2word), len(word_emb))
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


def clean_text_wrapper(clean_text_func, text, max_len):
    sent = clean_text_func(text)
    is_short = is_sent_shorter(sent, max_len)
    return sent if is_short else None
