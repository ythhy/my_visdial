import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from misc.attention import TanhAttention as Attention
from misc.attention import TanhAttention4 as Attention_h
from misc.attention import TanhAttention_C as Attention_c
from misc.model import RNNEncoder,LayerNorm


class _netE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout, img_feat_size):
        super(_netE, self).__init__()
        self.img_feat_size = img_feat_size
        self.d = dropout
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nhid = nhid
        self.ninp = ninp
        self.img_embed = weight_norm(nn.Linear(img_feat_size, nhid))
        # self.img_embed = nn.Linear(img_feat_size, nhid)

        self.rnn_layers = 2
        # self.ques_rnn = getattr(nn, rnn_type)(ninp, int(nhid / 2), nlayers, dropout=dropout, bidirectional=True)
        self.ques_rnn = RNNEncoder(word_size=ninp,
                                   hidden_size=nhid,
                                   bidirectional=False,
                                   drop_prob=dropout if self.rnn_layers > 1 else 0,
                                   n_layers=self.rnn_layers,
                                   rnn_type='lstm')
        # self.his_rnn = getattr(nn, rnn_type)(ninp, int(nhid / 2), nlayers, dropout=dropout, bidirectional=True)
        self.his_rnn = RNNEncoder(word_size=ninp,
                                  hidden_size=int(nhid / 2),
                                  bidirectional=False,
                                  drop_prob=dropout if self.rnn_layers > 1 else 0,
                                  n_layers=self.rnn_layers,
                                  rnn_type='lstm')
        self.attq = Attention(nhid, nhid, dropout=0.3)
        self.atti = Attention(nhid, nhid, dropout=0.3)
        self.atth = Attention_h(nhid, nhid, dropout=0.3)
        self.atti_s = Attention(nhid, nhid, dropout=0.3)
        self.atth_s = Attention_h(nhid, nhid, dropout=0.3)

        self.attq_c = Attention_c(1024, 1024)
        self.atti_c = Attention_c(1024, 1024)
        self.atth_c = Attention_c(1024, 1024)

        self.fc1 = nn.Linear(self.nhid * 3, self.ninp)
        self.fch1 = nn.Linear(1024, 1024)
        self.fch2 = nn.Linear(1024, 1024)
        self.fcv1 = nn.Linear(1024, 1024)
        self.fcv2 = nn.Linear(1024, 1024)
        self.fch = nn.Linear(1024, 2)
        self.fcv = nn.Linear(1024, 2)


    def forward(self, ques_emb, his_emb, img_raw, ques_input, his_input, memory, rnd):

        img_feat = F.tanh(self.img_embed(img_raw)).view(-1, 36, self.nhid)
        batch_size = img_feat.size(0)

        # ques_feat
        quesL = (ques_input != 0).sum(1)
        ques_f, ques_hidden, ques_emb = self.ques_rnn(ques_emb, quesL)
        # ques_f (bs, len, 1024) ques_hidden(bs, 1024)

        # his_feat
        hisL = (his_input != 0).sum(1)
        _, his_f, his_emb = self.his_rnn(his_emb, hisL)
        # his_f:(bs*rnd, 1024)
        his_f = his_f.view(-1, rnd, self.nhid)

        encoder_feat, [his_g0, img_g0], weight_q = self.recursive(ques_f, his_f, img_feat, memory[0],
                                                        memory[1])

        return encoder_feat, [his_g0, img_g0], weight_q

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, int(self.nhid / 2)).zero_()),
                    Variable(weight.new(self.nlayers, bsz, int(self.nhid / 2)).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, int(self.nhid / 2)).zero_())

    def init_bi_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * 2, bsz, int(self.nhid / 2)).zero_()),
                    Variable(weight.new(self.nlayers * 2, bsz, int(self.nhid / 2)).zero_()))
        else:
            return Variable(weight.new(self.nlayers * 2, bsz, int(self.nhid / 2)).zero_())

    def init_hidden_grucell(self, bsz, dim):
        weight = next(self.parameters()).data
        return Variable(weight.new(bsz, dim).zero_())

    def recursive(self, ques_feat, his_feat, img_feat, weight_h, weight_i):
        batch_size = ques_feat.size(0)

        _, his_s = self.atth_s(his_feat, None, None)
        _, img_s = self.atti_s(img_feat, None, None)

        if his_feat.size(1) == 1:
            his_m = Variable(torch.zeros(batch_size, 1, 1024)).cuda()
            img_m = Variable(torch.zeros(batch_size, 1, 1024)).cuda()
        else:
            his_m = torch.bmm(weight_h, his_feat[:, 0:-1, :])
            img_m = torch.bmm(weight_i, img_feat)


        hm = F.tanh(self.fch1(his_s)+self.fch2(his_m))
        hm = self.fch(F.dropout(hm, p=0.1, training=self.training))
        hm = F.softmax(hm, dim=2)
        h_cat = torch.cat((his_s,his_m), dim=1)
        his_g0 = torch.bmm(hm, h_cat)

        im = F.tanh(self.fcv1(img_s) + self.fcv2(img_m))
        im = self.fcv(F.dropout(im, p=0.1, training=self.training))
        im = F.softmax(im, dim=2)
        i_cat = torch.cat((img_s, img_m), dim=1)
        img_g0 = torch.bmm(im, i_cat)



        ##round1
        _, ques_g1 = self.attq(ques_feat, img_g0, his_g0)

        _, his_g1 = self.atth(his_feat, ques_g1, img_g0)

        _, img_g1 = self.atti(img_feat, ques_g1, his_g1)

        q = self.attq_c(ques_g1, img_g1, his_g1)
        v = self.atti_c(img_g1, ques_g1, his_g1)
        h = self.atth_c(his_g1, ques_g1, img_g1)

        ques_g1 = q
        his_g1 = h
        img_g1 = v

        ##round2

        _, ques_g2 = self.attq(ques_feat, img_g1, his_g1)

        weight_h2, his_g2 = self.atth(his_feat, ques_g2, img_g1)

        weight_i2, img_g2 = self.atti(img_feat, ques_g2, his_g2)

        ques_g2 = 0.5 * (ques_g1 + ques_g2)
        img_g2 = 0.5 * (img_g1 + img_g2)
        his_g2 = 0.5 * (his_g1 + his_g2)

        q = self.attq_c(ques_g2, img_g2, his_g2)
        v = self.atti_c(img_g2, ques_g2, his_g2)
        h = self.atth_c(his_g2, ques_g2, img_g2)

        ques_g2 = q
        his_g2 = h
        img_g2 = v

        ##round3

        weight_q, ques_g = self.attq(ques_feat, img_g2, his_g2)

        weight_h3, his_g = self.atth(his_feat, ques_g, img_g2)

        weight_i3, img_g = self.atti(img_feat, ques_g, his_g)

        ques_g = 0.5 * (ques_g2 + ques_g)
        img_g = 0.5 * (img_g2 + img_g)
        his_g = 0.5 * (his_g2 + his_g)

        q = self.attq_c(ques_g, img_g, his_g)
        v = self.atti_c(img_g, ques_g, his_g)
        h = self.atth_c(his_g, ques_g, img_g)

        concat_feat = torch.cat((q.view(-1, self.nhid), v.view(-1, self.nhid), \
                                 h.view(-1, self.nhid)), 1)

        encoder_feat = F.tanh(self.fc1(F.dropout(concat_feat, self.d, training=self.training)))

        weight_h = 0.5 * (weight_h2 + weight_h3)
        weight_i = 0.5 * (weight_i2 + weight_i3)

        weight_h.detach_()
        weight_i.detach_()

        return encoder_feat, [weight_h, weight_i], weight_q


class recursivefunc(nn.Module):
    def __init__(self, ninp, nhid, dropout=0.3):
        super(recursivefunc, self).__init__()
        self.attq = Attention(nhid, nhid, dropout=0.3)
        self.atti = Attention(nhid, nhid, dropout=0.3)
        self.atth = Attention_h(nhid, nhid, dropout=0.3)

        self.fc1 = nn.Linear(nhid * 3, ninp)

    def forward(self, ques_feat, his_feat, img_feat, img_g0, his_g0):
        ##round1
        _, ques_g1 = self.attq(ques_feat, img_g0, his_g0)

        _, his_g1 = self.atth(his_feat, ques_g1, img_g0)

        _, img_g1 = self.atti(img_feat, ques_g1, his_g1)

        his_g1 = 0.5 * (his_g1 + his_g0)
        img_g1 = 0.5 * (img_g1 + img_g0)

        ##round2

        _, ques_g2 = self.attq(ques_feat, img_g1, his_g1)

        _, his_g2 = self.atth(his_feat, ques_g2, img_g1)

        _, img_g2 = self.atti(img_feat, ques_g2, his_g2)

        ques_g2 = 0.5 * (ques_g1 + ques_g2)
        img_g2 = 0.5 * (img_g1 + img_g2)
        his_g2 = 0.5 * (his_g1 + his_g2)

        ##round3

        _, ques_g = self.attq(ques_feat, img_g2, his_g2)

        _, his_g = self.atth(his_feat, ques_g, img_g2)

        _, img_g = self.atti(img_feat, ques_g, his_g)

        ques_g = 0.5 * (ques_g2 + ques_g)
        img_g = 0.5 * (img_g2 + img_g)
        his_g = 0.5 * (his_g2 + his_g)

        concat_feat = torch.cat((ques_g.view(-1, self.nhid), his_g.view(-1, self.nhid), \
                                 img_g.view(-1, self.nhid)), 1)

        encoder_feat = F.tanh(self.fc1(F.dropout(concat_feat, self.d, training=self.training)))

        his_g.detach_()
        img_g.detach_()

        return encoder_feat, [his_g, img_g]