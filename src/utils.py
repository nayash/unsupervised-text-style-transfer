import re
import calendar
import unicodedata
import numpy as np
import torch
import time


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
    text = text.replace('_num_', 'NUMBER')
    text = text.replace("n't", 'not')
    # text = text.replace(" 's", 's')
    text = text.replace(" 've", ' have')
    text = text.replace(" 'd", ' would')
    text = text.replace(" 'm", ' am')
    text = text.replace("&", 'and')
    text = re.sub(r'[\"\'\`\~\#\$\%\&\+\^\*\“\”\’\‘\:]', ' ', text)
    text = re.sub(r'[-—_,]', ' ', text)
    text = re.sub(r'((\?+)|(\!+)|(;+)|(\.+))', '.', text)
    text = re.sub(r'[()]', '', text)
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


def vocab_from_pretrained_emb(emb_path, words):
    # TODO skip glove embeddings for out of vocab words
    word2idx = {}
    idx2word = {}
    word_emb = []
    with open(emb_path) as file:
        for i, line in enumerate(file):
            split = line.split()
            word = split[0]
            if word not in words:
                continue
            emb = split[1:]
            word2idx[word] = len(word2idx)
            idx2word[len(idx2word)] = word
            word_emb.append([float(i) for i in emb])

    # now add words from corpus which are missing in Glove embeddings
    diff = set(words).difference(word2idx.keys())
    # print('extra words', len(diff))

    for word in words:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
            idx2word[len(idx2word)] = word
            word_emb.append(np.random.uniform(0, 1, len(word_emb[-1])))

    return word2idx, idx2word, word_emb, list(diff)


def roll_prepend(tensor, prefix_num, dim=1):
    b = torch.roll(tensor, 1, dim)
    b[:, 0] = prefix_num
    return b


def permute_items(collection, k=3):
    q = np.random.uniform(0, k+1, len(collection)) + np.arange(len(collection))
    return list(np.array(collection)[np.argsort(q)])


def permute_tensor(tensor, k=3):
    c = torch.FloatTensor(tensor.size(0)).uniform_(
        0, k+1)+torch.arange(tensor.size(0))
    return tensor[torch.argsort(c)]


def get_readable_ctime():
    return time.strftime("%d-%m-%Y %H_%M_%S")
