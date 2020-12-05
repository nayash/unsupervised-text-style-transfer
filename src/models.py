#
# Copyright (c) 2020. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import torch
from torch import nn
import torch.functional as F
from torch.nn import init
from constants import *


class Encoder(nn.Module):
    def __init__(self, input_size_src, input_size_tgt, hidden_size,
                 word_emb_src, word_emb_tgt, device, batch_first=True,
                 layers=1, bidirectional=False, dropout=0.0, word_do=0.0):
        super(Encoder, self).__init__()

        self.device = device
        self.batch_first = batch_first
        self.layers = layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.emb_src = nn.Embedding(input_size_src, hidden_size)
        self.emb_tgt = nn.Embedding(input_size_tgt, hidden_size)
        self.emb_src.load_state_dict({'weight': word_emb_src})
        self.emb_tgt.load_state_dict({'weight': word_emb_tgt})
        self.word_do = nn.Dropout2d(word_do)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=batch_first,
                            num_layers=layers, bidirectional=bidirectional,
                            dropout=dropout)  # input_size = len of sequence
        # self.apply(self.weights_init)

    def forward(self, input, hidden):
        assert self.mode
        if self.mode == src2tgt or self.mode == src2src:
            emb = self.emb_src(input)  # (batch, input_size, hidden_size)
        else:
            emb = self.emb_tgt(input)
        emb = self.word_do(emb)  # randomly sets word embs to 0
        if not self.batch_first:
            emb = emb.permute(1, 0, 2)

        # o,h1,h2 => [batch_size, max_len, emb_dim], [layers*num_dir, batch, emb_dim]
        o, h = self.lstm(emb, hidden)
        return o, h

    def init_hidden(self, batch_size):
        return (torch.zeros(self.layers * (1 if not self.bidirectional else 2),
                            batch_size, self.hidden_size, device=self.device,
                            requires_grad=True),
                torch.zeros(self.layers * (1 if not self.bidirectional else 2),
                            batch_size, self.hidden_size, device=self.device,
                            requires_grad=True))

    def weights_init(self, m):
        if isinstance(m, torch.nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal(param)
        else:
            init.xavier_uniform_(m.weight.data)

    def set_mode(self, mode):
        self.mode = mode


class Decoder(nn.Module):
    def __init__(self, output_size_src, output_size_tgt, hidden_size,
                 word_emb_src, word_emb_tgt, device, batch_first=True, layers=1,
                 bidirectional=False, dropout=0.0, use_attn=False, emb_do=0.0,
                 enc_layers=1, enc_bidir=False):
        super(Decoder, self).__init__()

        self.device = device
        self.use_attn = use_attn
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dir_dim = 1+int(bidirectional)
        self.layers = layers
        self.hidden_size = hidden_size
        self.enc_layers = enc_layers
        self.enc_bidir = enc_bidir
        self.is_enc_dec_diff = enc_bidir != bidirectional or layers != enc_layers
        self.out_size_src = output_size_src
        self.out_size_tgt = output_size_tgt
        self.emb_src = nn.Embedding(output_size_src, hidden_size)
        self.emb_src.load_state_dict({'weight': word_emb_src})
        self.emb_tgt = nn.Embedding(output_size_tgt, hidden_size)
        self.emb_tgt.load_state_dict({'weight': word_emb_tgt})
        self.emb_do = nn.Dropout(emb_do)
        self.lstm = nn.LSTM(self.lstm_in_size(), hidden_size,
                            batch_first=batch_first,
                            num_layers=layers, bidirectional=bidirectional,
                            dropout=dropout)  # input_size = len of sequence
        attn_in_size = self.attn_in_size()
        if use_attn:
            self.attention = Attention((1+int(bidirectional))*hidden_size)

        self.linear = nn.Linear((1+int(bidirectional)) * hidden_size,
                                max(output_size_src, output_size_tgt))
        # self.lrelu = nn.LeakyReLU(0.2)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, enc_out, position):
        # input => [batch_size, 1], hidden[0]/[1] => [layers*dir, bs, emb_dim]
        # enc_out => [bs, max_len, emb_dim*dir]
        # print('original', hidden[0].shape, hidden[1].shape, enc_out.shape, self.is_enc_dec_diff)
        if position == 0:
            if self.is_enc_dec_diff:
                h0 = hidden[0].view(self.enc_layers, 1 + int(self.enc_bidir),
                                    input.size(0), self.hidden_size)
                h1 = hidden[1].view(self.enc_layers, 1 + int(self.enc_bidir),
                                    input.size(0), self.hidden_size)
                # print('1_', h0.shape, h1.shape)
                hidden = (h0[-self.layers:, :self.dir_dim].
                          view(-1, input.size(0), self.hidden_size).contiguous()
                    , h1[-self.layers:, :self.dir_dim].
                          view(-1, input.size(0), self.hidden_size).contiguous())
                self.enc_out = enc_out.view(input.size(0), enc_out.size(1), 1+int(self.enc_bidir),
                             self.hidden_size)[:, :, 0, :]  # keep only forward hidden
            else:
                self.enc_out = enc_out

        # print('after', hidden[0].shape, hidden[1].shape, self.enc_out.shape)

        if self.mode == src2tgt or self.mode == tgt2tgt:
            emb = self.emb_tgt(input)  # (batch, seq_len, emb_dim)
            out_dim = self.out_size_tgt
        else:
            emb = self.emb_src(input)
            out_dim = self.out_size_src
        emb = self.emb_do(emb)
        if not self.batch_first:
            emb = emb.permute(1, 0, 2)

        if self.use_attn:
            # Bahdanau's attention
            h = hidden[0].view(self.layers, self.dir_dim,
                               input.size(0), self.hidden_size)[-1]  # [dir, bs, dim]
            h = h.permute(1, 0, 2).reshape(input.size(0), 1, -1)  # [bs, 1, dir*emb_dim]
            lstm_in, attn_wts = self.attention(h, self.enc_out)  # [bs, 1, dir*emb]
            lstm_in = torch.cat((lstm_in, emb), -1)  # [bs, 1, dir*emb+emb]
        else:
            lstm_in = emb

        o, h = self.lstm(lstm_in, hidden)
        x = self.log_softmax(self.linear(o))
        return x[:, :, :out_dim], h

    def init_hidden(self, batch_size):
        return (torch.zeros(self.layers * (1 if not self.bidirectional else 2),
                            batch_size, self.hidden_size, device=self.device,
                            requires_grad=True),
                torch.zeros(self.layers * (1 if not self.bidirectional else 2),
                            batch_size, self.hidden_size, device=self.device,
                            requires_grad=True))

    def weights_init(self, m):
        if isinstance(m, torch.nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal(param)
        else:
            init.xavier_uniform_(m.weight.data)

    def attn_in_size(self):
        res = self.hidden_size
        if self.use_attn:
            res = res * 2 * (2 if self.bidirectional else 1)
        return res

    def lstm_in_size(self):
        res = self.hidden_size
        if self.use_attn:
            res = res * (int(self.bidirectional) + 1) + self.hidden_size
        return res

    def set_mode(self, mode):
        self.mode = mode


class GeneratorModel(nn.Module):
    def __init__(self, input_vocab_src, input_vocab_tgt, hidden_size,
                 batch_size, word_emb_src, word_emb_tgt, device,
                 batch_first=True, layers_gen=1, layers_dec=1, bidir_gen=False,
                 bidir_dec=False, lstm_do=0.0, use_attn=False, emb_do=0,
                 word_do=0.0):
        super(GeneratorModel, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.layers_gen = layers_gen
        self.layers_dec = layers_dec
        self.bidir_gen = bidir_gen
        self.bidir_dec = bidir_dec
        self.hidden_size = hidden_size
        self.encoder = Encoder(input_vocab_src, input_vocab_tgt, hidden_size,
                               word_emb_src, word_emb_tgt, device,
                               batch_first=batch_first,
                               bidirectional=bidir_gen, layers=layers_gen,
                               dropout=lstm_do, word_do=word_do)
        self.decoder = Decoder(input_vocab_src, input_vocab_tgt,  hidden_size,
                               word_emb_src, word_emb_tgt, device,
                               layers=layers_dec, bidirectional=bidir_dec,
                               batch_first=batch_first, dropout=lstm_do,
                               use_attn=use_attn, emb_do=emb_do,
                               enc_layers=layers_gen, enc_bidir=bidir_gen)
        self.enc_sos = None
        self.dec_sos = None
        # self.linear_topk = nn.Linear()

    def forward(self, x):
        enc_out, h = self.encoder(x, self.encoder.init_hidden(
            x.size(self.get_batch_index(self.batch_first))))
        decoder_out = []
        decoder_raw = []

        decoder_input = torch.tensor([[self.dec_sos]] * x.size(
            self.get_batch_index(self.batch_first)), device=self.device)

        decoder_hidden = h

        for i in range(enc_out.size(1)):
            out, decoder_hidden = self.decoder(decoder_input, decoder_hidden,
                                               enc_out, i)  # out=> [1, 1, vocab]
            pred, idx = out.topk(1)
            decoder_input = idx.view(idx.size(0), 1)
            decoder_out.append(idx.squeeze())
            # print('dec_out_loop', out.shape)  # [*, 120, 1]
            decoder_raw.append(out)

        if len(decoder_out[0].size()) > 0:
            stack_dim = 1
        else:
            stack_dim = 0
        # print('stack_dim', stack_dim)  # stack_dim = 1
        out1 = torch.stack(decoder_out, dim=stack_dim)
        out2 = torch.stack(decoder_raw, dim=stack_dim)  # [120, 22, 1]
        # print('gen_out_raw', out2.shape)
        return out1, out2.squeeze(), enc_out

    def get_batch_index(self, batch_first):
        return 0 if batch_first else 1

    def set_mode(self, mode, word2idx_src, word2idx_tgt):
        self.mode = mode
        self.encoder.set_mode(mode)
        self.decoder.set_mode(mode)
        if mode == src2src:
            self.enc_sos = word2idx_src[SOS_SRC]
            self.dec_sos = word2idx_src[SOS_SRC]
        elif mode == tgt2tgt:
            self.enc_sos = word2idx_tgt[SOS_TGT]
            self.dec_sos = word2idx_tgt[SOS_TGT]
        elif mode == src2tgt:
            self.enc_sos = word2idx_src[SOS_SRC]
            self.dec_sos = word2idx_tgt[SOS_TGT]
        elif mode == tgt2src:
            self.enc_sos = word2idx_tgt[SOS_TGT]
            self.dec_sos = word2idx_src[SOS_SRC]


class LatentClassifier(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_nodes):
        super(LatentClassifier, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_nodes = hidden_nodes

        self.linear1 = nn.Linear(input_shape, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, int(hidden_nodes / 2))
        self.linear3 = nn.Linear(int(hidden_nodes / 2), int(hidden_nodes / 8))
        # self.linear4 = nn.Linear(int(hidden_nodes/4), int(hidden_nodes/8))
        self.linear5 = nn.Linear(int(hidden_nodes / 8), int(hidden_nodes / 32))
        # self.linear6 = nn.Linear(int(hidden_nodes/16), int(hidden_nodes/32))
        self.linear7 = nn.Linear(int(hidden_nodes / 32), output_shape)
        self.bn3 = nn.BatchNorm1d(num_features=int(hidden_nodes / 8))
        self.bn5 = nn.BatchNorm1d(num_features=int(hidden_nodes / 32))
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = self.lrelu(self.linear1(input))
        x = self.lrelu(self.linear2(x))
        x = self.lrelu(self.bn3(self.linear3(x)))
        # x = self.lrelu(self.linear4(x))
        x = self.lrelu(self.bn5(self.linear5(x)))
        # x = self.lrelu(self.linear6(x))
        # x = self.sigmoid(self.linear7(x))
        x = self.linear7(x)
        return x.squeeze()

    def weights_init(self, m):
        init.xavier_uniform_(m.weight.data)


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()output
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len,
                                                 query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len,
                                                   query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len,
                                                dimensions)
        output = self.tanh(output)

        return output, attention_weights



# old
# dec_in torch.Size([100, 1]) torch.Size([4, 100, 200]) torch.Size([4, 100, 200]) torch.Size([100, 11, 400])
# dec_emb torch.Size([100, 1, 200])
# dec_lstm_in torch.Size([100, 1, 200])
# dec_lstm_out torch.Size([100, 1, 400]) torch.Size([4, 100, 200]) torch.Size([4, 100, 200])
# dec_out torch.Size([100, 1, 41485])
#
# generic
# dec_in torch.Size([120, 1]) torch.Size([2, 120, 200]) torch.Size([2, 120, 200]) torch.Size([120, 22, 400])
# dec_emb torch.Size([120, 1, 200])
# dec_lstm_in torch.Size([120, 1, 200])
# dec_lstm_out torch.Size([120, 1, 400]) torch.Size([2, 120, 200]) torch.Size([2, 120, 200])
# dec_out torch.Size([120, 1, 8276])