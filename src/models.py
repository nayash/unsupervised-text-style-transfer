from torch import nn
import torch.functional as F
from torch.nn import init

from train import src2src, word2idx, SOS_SRC, tgt2tgt, SOS_TGT, src2tgt, tgt2src


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, word_emb_tensor,batch_first=True,
                 layers=1, bidirectional=False, dropout=0.0):
        super(Encoder, self).__init__()

        self.batch_first = batch_first
        self.layers = layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_size = input_size  # size of input vocab
        self.emb = nn.Embedding(input_size, hidden_size)
        self.emb.load_state_dict({'weight': word_emb_tensor})
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=batch_first,
                            num_layers=layers, bidirectional=bidirectional,
                            dropout=dropout)  # input_size = len of sequence

        # self.apply(self.weights_init)

    def forward(self, input, hidden):
        emb = self.emb(input)  # (batch, input_size, hidden_size)
        if not self.batch_first:
            emb = emb.permute(1, 0, 2)

        # (o,h) => (torch.Size([128, 20, 200]),torch.Size([2, 128, 100]),
        # torch.Size([2, 128, 100]))
        o, h = self.lstm(emb, hidden)
        return o, h

    def initHidden(self, batch_size):
        return (torch.zeros(self.layers*(1 if not self.bidirectional else 2),
                            batch_size, self.hidden_size, device=device,
                            requires_grad=True),
                torch.zeros(self.layers*(1 if not self.bidirectional else 2),
                            batch_size, self.hidden_size, device=device,
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


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, word_emb_tensor, batch_first=True,
                 layers=1, bidirectional=False, dropout=0.0):
        super(Decoder, self).__init__()

        self.batch_first = batch_first
        self.output_size = output_size  # output vocab
        self.bidirectional = bidirectional
        self.layers = layers
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(output_size, hidden_size)
        self.emb.load_state_dict({'weight': word_emb_tensor})
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=batch_first,
                            num_layers=layers, bidirectional=bidirectional,
                            dropout=dropout)  # input_size = len of sequence
        self.linear = nn.Linear(
            (1 if not self.bidirectional else 2)*hidden_size, output_size)
        self.lrelu = nn.LeakyReLU(0.2)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        emb = self.emb(input)  # (batch, seq_len, hidden_size)
        if not self.batch_first:
            emb = emb.permute(1, 0, 2)
        o, h = self.lstm(emb, hidden)  # output/hidden => (1, 1, 256 or 512)
        x = self.log_softmax(self.linear(o))
        return x, h

    def initHidden(self, batch_size):
        return (torch.zeros(self.layers*(1 if not self.bidirectional else 2),
                            batch_size, self.hidden_size, device=device,
                            requires_grad=True),
                torch.zeros(self.layers*(1 if not self.bidirectional else 2),
                            batch_size, self.hidden_size, device=device,
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


class GeneratorModel(nn.Module):
    def __init__(self, input_vocab, hidden_size, batch_size, word_emb_tensor,
                 batch_first=True, layers=1, bidirectional=False, lstm_do=0.0):
        super(GeneratorModel, self).__init__()
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.encoder = Encoder(input_vocab, hidden_size, word_emb_tensor, batch_first=batch_first,
                               bidirectional=bidirectional, layers=layers,
                               dropout=lstm_do)
        self.decoder = Decoder(input_vocab, hidden_size, word_emb_tensor, layers=layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first, dropout=lstm_do)
        self.enc_sos = None
        self.dec_sos = None
        # self.linear_topk = nn.Linear()

    def forward(self, x):
        enc_out, h = self.encoder(x, self.encoder.initHidden(
            x.size(self.get_batch_index(self.batch_first))))
        decoder_out = []
        decoder_raw = []

        decoder_input = torch.tensor([[self.dec_sos]]*x.size(
            self.get_batch_index(self.batch_first)), device=device)

        decoder_hidden = h

        for i in range(max_len):
            out, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden)  # out=> [1, 1, vocab]
            pred, idx = out.topk(1)
            decoder_input = idx.view(idx.size(0), 1)
            decoder_out.append(idx.squeeze())
            decoder_raw.append(out)

        if len(decoder_out[0].size()) > 0:
            stack_dim = 1
        else:
            stack_dim = 0
        out1 = torch.stack(decoder_out, dim=stack_dim)
        out2 = torch.stack(decoder_raw, dim=stack_dim)
        return out1, out2.squeeze(), enc_out

    def get_batch_index(self, batch_first):
        return 0 if batch_first else 1

    def set_mode(self, mode):
        if mode == src2src:
            self.enc_sos = word2idx[SOS_SRC]
            self.dec_sos = word2idx[SOS_SRC]
        elif mode == tgt2tgt:
            self.enc_sos = word2idx[SOS_TGT]
            self.dec_sos = word2idx[SOS_TGT]
        elif mode == src2tgt:
            self.enc_sos = word2idx[SOS_SRC]
            self.dec_sos = word2idx[SOS_TGT]
        elif mode == tgt2src:
            self.enc_sos = word2idx[SOS_TGT]
            self.dec_sos = word2idx[SOS_SRC]


# class LatentClassifier(nn.Module):
#     def __init__(self, input_shape, output_shape, hidden_nodes):
#         super(LatentClassifier, self).__init__()
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.hidden_nodes = hidden_nodes

#         self.linear1 = nn.Linear(input_shape, hidden_nodes)
#         self.linear2 = nn.Linear(hidden_nodes, int(hidden_nodes/4))
#         self.linear3 = nn.Linear(int(hidden_nodes/4), output_shape)
#         self.lrelu = nn.LeakyReLU(0.2)
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, input):
#         x = self.linear1(input)  # [bs, 21, hidden_nodes]
#         # print('clf1', x.shape)
#         x = self.lrelu(self.linear2(x))  # [bs, 21, hidden_nodes/4]
#         # print('clf2', x.shape)
#         x = self.sigmoid(self.linear3(x))  # [bs, 21, 1]
#         # print('clf3', x.shape)
#         x = x.squeeze()
#         x = x.prod(dim=1)
#         # print('clf4', x.shape)
#         x = self.sigmoid(x)
#         # print('clf5', x.shape)
#         return x.squeeze()  # [batch_size]


class LatentClassifier(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_nodes):
        super(LatentClassifier, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_nodes = hidden_nodes

        self.linear1 = nn.Linear(input_shape, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, int(hidden_nodes/2))
        self.linear3 = nn.Linear(int(hidden_nodes/2), int(hidden_nodes/8))
        # self.linear4 = nn.Linear(int(hidden_nodes/4), int(hidden_nodes/8))
        self.linear5 = nn.Linear(int(hidden_nodes/8), int(hidden_nodes/32))
        # self.linear6 = nn.Linear(int(hidden_nodes/16), int(hidden_nodes/32))
        self.linear7 = nn.Linear(int(hidden_nodes/32), output_shape)
        self.bn3 = nn.BatchNorm1d(num_features=int(hidden_nodes/8))
        self.bn5 = nn.BatchNorm1d(num_features=int(hidden_nodes/32))
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
