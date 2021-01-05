#
# Copyright (c) 2020 - present. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import sys
sys.path.append('../src')
import unittest
import pickle
from src.utils import *
import multiprocessing as mp
import numpy as np
from src.constants import *
from functools import partial

# python -m unittest tests.utilstest

class UtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        with open('/home/asutosh/Documents/ml_projects/unsupervised-text-style-transfer/outputs/runs/de-en-noSkipOov/data_cp10.pk', 'rb') as f:
            checkpoint = pickle.load(f)
        self.words_src = list(checkpoint['word2idx_src'].keys())
        self.words_tgt = list(checkpoint['word2idx_tgt'].keys())
        self.emb_path_src = '/home/asutosh/Documents/ml_projects/unsupervised-text-style-transfer/inputs/word_embs/wiki.fr.align.vec'
        self.emb_path_tgt = '/home/asutosh/Documents/ml_projects/unsupervised-text-style-transfer/inputs/word_embs/wiki.en.align.vec'
        self.pool = mp.Pool(mp.cpu_count())
        self.extra_tokens_src = [SOS_SRC, 'EOS', 'PAD', 'UNK']
        self.extra_tokens_tgt = [SOS_TGT, 'EOS', 'PAD', 'UNK']
        self.sents_path = '/home/asutosh/Documents/ml_projects/unsupervised-text-style-transfer/inputs/news'

    def test_vocab_from_pretrained_emb_parallel_src(self):
        print('check', SOS_SRC in self.words_src, SOS_TGT in self.words_src)
        word2idx, idx2word, emb = vocab_from_pretrained_emb_parallel(self.emb_path_src, self.words_src,
                                           self.pool, self.extra_tokens_src, mp.cpu_count(), emb_dim=-1, skip_oov=False)
        # word2idx, idx2word, emb = vocab_from_pretrained_emb(
        #     self.emb_path, self.words, extra_tokens=self.extra_tokens)
        print('lengths', len(word2idx), len(idx2word), len(emb))

        for k, v in word2idx.items():
            assert idx2word[v] == k, 'word2idx and idx2word not symmetrical' +\
            k+','+str(v)+','+idx2word[v]
        with open(self.emb_path_src) as file:
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

    def test_vocab_from_pretrained_emb_parallel_tgt(self):
        print('check', SOS_SRC in self.words_tgt, SOS_TGT in self.words_tgt)
        word2idx, idx2word, emb = vocab_from_pretrained_emb_parallel(
            self.emb_path_tgt, self.words_tgt,
            self.pool, self.extra_tokens_tgt, mp.cpu_count(), emb_dim=-1, skip_oov=False)
        # word2idx, idx2word, emb = vocab_from_pretrained_emb(
        #     self.emb_path, self.words, extra_tokens=self.extra_tokens)
        print('lengths', len(word2idx), len(idx2word), len(emb))

        for k, v in word2idx.items():
            assert idx2word[v] == k, 'word2idx and idx2word not symmetrical' + \
                                     k + ',' + str(v) + ',' + idx2word[v]
        with open(self.emb_path_tgt) as file:
            for i, (k, v) in enumerate(word2idx.items()):
                for i, line in enumerate(file):
                    split = line.split()
                    word = split[0]
                    if word == k:
                        emb_vec = split[1:]
                        assert np.array_equal(np.array(emb[v]).astype(float),
                                              np.array(emb_vec).astype(
                                                  float)), 'vectors don\'t match' + str(
                            emb[v][:10]) + ',' + str(
                            split[1:][:10]) + ',' + word
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
        assert SOS_TGT in word2idx
        self.pool.close()

# {
#   "batch_size": 30,
#   "batch_preproc_buffer": 1e8,
#   "skip_oov": false,
#   "min_word_freq": 3,
#   "hidden_dim": 400,
#   "bidir_gen": true,
#   "bidir_dec": false,
#   "layers_gen": 2,
#   "layers_dec": 2,
#   "epoch": 15,
#   "lstm_do": 0.2,
#   "gen_grad_clip": 10,
#   "disc_grad_clip": 3,
#   "gen_wt_clip": 10,
#   "disc_wt_clip": 0,
#   "gen_lr_sched": "non-cyclic",
#   "gen_lr": 1e-4,
#   "base_lr": 1e-6,
#   "max_lr": 5e-5,
#   "gen_weight_decay": 0,
#   "gen_lr_reduce_factor": 0.1,
#   "gen_lr_reduce_patience": 0,
#   "gen_lr_reduce_factor_iter": 0.1,
#   "gen_lr_reduce_patience_iter": 15,
#   "early_stop_patience": 100,
#   "use_early_stop": false,
#   "step_size": 1000,
#   "mode": "triangular",
#   "disc_lr_sched": "cyclic",
#   "clf_lr": 1e-5,
#   "disc_base_lr": 1e-4,
#   "disc_max_lr": 1e-3,
#   "disc_weight_decay": 0.0,
#   "disc_lr_reduce_factor": 0.01,
#   "disc_lr_reduce_patience": 200,
#   "wgt_loss_auto": 1,
#   "wgt_loss_cd": 1,
#   "wgt_loss_adv": 1,
#   "noisy_input": true,
#   "noisy_cd_input": true,
#   "word_dropout": 0.3,
#   "use_attention": false,
#   "emb_do": 0.3,
#   "adv_training": true,
#   "max_sentence_len": 15,
#   "max_samples": 2e6,
#   "val_split": 0.020,
#   "sample_generation_interval_iters": 500,
#   "print_samples": true
# }