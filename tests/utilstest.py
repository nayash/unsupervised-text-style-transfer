#
# Copyright (c) 2020. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import unittest
import pickle
import sys
sys.path.append('../src')
from src.utils import *
import multiprocessing as mp
import numpy as np
from src.constants import *


class UtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        with open('/home/asutosh/Documents/ml_projects/unsupervised-text-style-transfer/outputs/data_cp8.pk', 'rb') as f:
            checkpoint = pickle.load(f)
        self.words = checkpoint['words']
        self.emb_path = '/home/asutosh/Documents/ml_projects/unsupervised-text-style-transfer/inputs/glove.6B.200d.txt'
        self.pool = mp.Pool(mp.cpu_count())
        self.extra_tokens = [SOS_SRC, SOS_TGT, 'EOS', 'PAD', 'UNK']

    def test_vocab_from_pretrained_emb_parallel(self):
        print('check', SOS_SRC in self.words, SOS_TGT in self.words)
        word2idx, idx2word, emb = vocab_from_pretrained_emb_parallel(self.emb_path, self.words,
                                           self.pool, self.extra_tokens, mp.cpu_count())
        # word2idx, idx2word, emb = vocab_from_pretrained_emb(
        #     self.emb_path, self.words, extra_tokens=self.extra_tokens)
        print('lengths', len(word2idx), len(idx2word), len(emb))

        for k, v in word2idx.items():
            assert idx2word[v] == k, 'word2idx and idx2word not symmetrical' +\
            k+','+str(v)+','+idx2word[v]
        with open(self.emb_path) as file:
            for i, (k, v) in enumerate(word2idx.items()):
                for i, line in enumerate(file):
                    split = line.split()
                    word = split[0]
                    if word == k:
                        emb_vec = split[1:]
                        assert np.array_equal(np.array(emb[v]).astype(float), np.array(emb_vec).astype(float)), 'vectors don\'t match'+str(emb[v][:10])+','+str(split[1:][:10])+','+word
                # if i > 100000:
                #     break

        _ = list(word2idx.values())
        print(min(_), max(_), len(_))
        temp = np.arange(max(word2idx.values()) + 1)
        print(min(temp), max(temp), len(temp))
        assert np.array_equal(np.array(_), temp)
        _ = list(idx2word.keys())
        print(min(_), max(_), _[:50])
        assert np.array_equal(np.array(_), np.arange(max(idx2word.keys()) + 1))
        assert SOS_SRC in word2idx
        self.pool.close()
