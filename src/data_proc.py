#
# Copyright (c) 2020 - present. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import time
import multiprocessing as mp
from functools import partial
from utils.utils import *
import torch
import numpy as np


class DataProc:
    def __init__(self, **kwargs):
        super(DataProc, self)
        self.config_dict = kwargs
        self.logger = self.config_dict['logger']
        self.pool = mp.Pool(mp.cpu_count())

        # these values are calculated below
        self.word2idx_src, self.idx2word_src, self.word_emb_src, \
        self.oov_words_size_src, self.pre_trained_emb_size_src, \
        self.word2idx_tgt, self.idx2word_tgt, self.word_emb_tgt, \
        self.oov_words_size_tgt, self.pre_trained_emb_size_tgt = [None]*10
        self.max_len = -1
        self.src_sents = []
        self.tgt_sents = []
        self.x_src = []
        self.x_tgt = []

        self.process_data()

    def read_sentences_multiproc(self):
        src_file_path = self.config_dict['src_path']
        tgt_file_path = self.config_dict['tgt_path']
        clean_text_func_src = locals()[self.config_dict['cleanfuncsrc']]
        clean_text_func_tgt = locals()[self.config_dict['cleanfunctgt']]
        max_len = self.config_dict['max_sentence_len']
        # since text file could be very large read it in chunks
        buffer = int(self.config_dict['batch_preproc_buffer'])
        self.log('processing source sentences from {}, with buffer size {}'.
                 format(src_file_path, buffer))
        src_sents = []
        tgt_sents = []

        # read source file sentences
        with open(src_file_path, encoding='utf-8', errors='ignore') as file:
            stime = time.time()
            temp = file.readlines(buffer)
            while temp:
                src_sents.extend(self.pool.map(partial(clean_text_wrapper,
                                                  clean_text_func=clean_text_func_src,
                                                  max_len=max_len), temp,
                                          chunksize=20000))
                src_sents = list(filter(None, src_sents))
                if len(src_sents) > self.config_dict['max_samples']:
                    self.log('reached ', len(src_sents),
                             ' src samples. skipping rest')
                    break
                temp = file.readlines(buffer)

        self.log('processed src sentences in', (time.time() - stime), 'secs')

        self.log('processing target sentences from {}, with buffer size {}'.
                format(tgt_file_path, buffer))

        # read tgt file sentences
        with open(tgt_file_path, encoding='utf-8', errors='ignore') as file:
            stime = time.time()
            temp = file.readlines(buffer)
            while temp:
                tgt_sents.extend(self.pool.map(partial(clean_text_wrapper,
                                                  clean_text_func=clean_text_func_tgt,
                                                  max_len=max_len), temp,
                                               chunksize=20000))
                tgt_sents = list(filter(None, tgt_sents))
                if len(tgt_sents) > self.config_dict['max_samples']:
                    self.log('reached ', len(tgt_sents),
                             ' tgt samples. skipping rest')
                    break
                temp = file.readlines(buffer)

        self.log('processed tgt sentences in', (time.time() - stime), 'secs')

        self.log('number of src/tgt sentences collected', len(src_sents),
              len(tgt_sents))
        random.shuffle(src_sents)
        random.shuffle(tgt_sents)
        self.log(src_sents[:5], tgt_sents[-5:])

        if len(src_sents) < len(tgt_sents):
            tgt_sents = tgt_sents[:len(src_sents)]
        else:
            src_sents = src_sents[:len(tgt_sents)]

        # if user has provided any max sentence length config then use it, else
        # if it's -1 use the maximum sentence length in the dataset
        _ = self.pool.map(len_word_tokenize, src_sents + tgt_sents) \
            if max_len == -1 else max_len
        try:
            max_len = max(_)
        except:
            max_len = _

        self.src_sents = src_sents
        self.tgt_sents = tgt_sents

    def build_vocab_emb(self):
        src_sents = self.src_sents
        tgt_sents = self.tgt_sents

        self.log('building vocabulary for source language ...')
        extra_tokens_src = [SOS_SRC, 'EOS', 'PAD', 'UNK']
        extra_tokens_tgt = [SOS_TGT, 'EOS', 'PAD', 'UNK']

        self.word2idx_src, self.idx2word_src, self.word_emb_src, \
        self.oov_words_size_src, self.pre_trained_emb_size_src = vocab_from_sents(
            src_sents, self.pool, extra_tokens_src,
            self.config_dict['src_word_emb_path'],
            emb_dim=self.config_dict['embedding_dim'],
            skip_oov=self.config_dict['skip_oov'],
            min_freq=self.config_dict['min_word_freq'])

        self.log('building vocabulary for target language ...')
        self.word2idx_tgt, self.idx2word_tgt, self.word_emb_tgt, \
        self.oov_words_size_tgt, self.pre_trained_emb_size_tgt = vocab_from_sents(
            tgt_sents, self.pool, extra_tokens_tgt,
            self.config_dict['tgt_word_emb_path,'],
            emb_dim=self.config_dict['embedding_dim'],
            skip_oov=self.config_dict['skip_oov'],
            min_freq=self.config_dict['min_word_freq'])

    def sent_to_tensor(self):
        self.log('converting sentences to tensors...')
        x_src = []
        x_tgt = []

        _ = time.time()
        [x_src.append(sent_to_tensor(sent, max_len=self.max_len, type='src',
                                     word2idx=self.word2idx_src)) for sent in
         self.src_sents]

        self.log('x_src size {}. Build time {} secs'.
                 format(len(x_src), (time.time() - _)))

        _ = time.time()
        [x_tgt.append(sent_to_tensor(sent, max_len=self.max_len, type='tgt',
                                     word2idx=self.word2idx_tgt)) for sent in
         self.tgt_sents]

        self.log('x_tgt size {}. Build time {} secs'.
                 format((x_tgt), (time.time() - _)))
        x_src = torch.stack(x_src)
        x_tgt = torch.stack(x_tgt)
        delta = x_tgt.shape[0] - x_src.shape[0]
        if delta > 0:
            indexes = np.random.choice(x_src.size(0), delta)
            x_src = torch.cat((x_src, x_src[indexes]))
        else:
            indexes = np.random.choice(x_tgt.size(0), np.abs(delta))
            x_tgt = torch.cat((x_tgt, x_tgt[indexes]))

        assert len(x_src) == len(x_tgt)
        self.x_src = x_src
        self.x_tgt = x_tgt

    def process_data(self):
        stime = time.time()
        self.read_sentences_multiproc()
        self.build_vocab_emb()
        self.sent_to_tensor()

        self.log('processing finished in {} seconds'.format(time.time()-stime))

    def log(self, *text):
        if self.logger:
            self.logger.append_log(text)
        else:
            print(text)
