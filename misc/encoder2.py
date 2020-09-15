import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from misc.attention import Attentiontanh as Attention_h
from misc.attention import TanhAttention as Attention_i
from misc.attention import TanhAttention_C as Attention_c
from misc.model import RNNEncoder,LayerNorm
from misc.nlocal import NonLocalBlockFC


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


        self.ques_rnn = RNNEncoder(word_size=ninp,
                                          hidden_size=nhid,
                                          bidirectional=False,
                                          drop_prob=dropout if self.nlayers>1 else 0,
                                          n_layers=self.nlayers,
                                          rnn_type='lstm')
        self.his_rnn = RNNEncoder(word_size=ninp,
                                   hidden_size=nhid,
                                   bidirectional=False,
                                   drop_prob=dropout if self.nlayers > 1 else 0,
                                   n_layers=self.nlayers,
                                   rnn_type='lstm')
        self.attq = NonLocalBlockFC(nhid)
        self.atti = Attention_i(nhid, nhid, dropout=0.3)
        self.atth = Attention_h(nhid, nhid, dropout=0.3)
        self.LN = LayerNorm(nhid)

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


    def forward(self, ques_emb, his_emb, img_raw, ques_input, his_input, rnd):

        img_f = F.tanh(self.img_embed(img_raw)).view(-1, 36, self.nhid)

        # ques_feat
        quesL = (ques_input!=0).sum(1)
        ques_f, ques_hidden, ques_emb = self.ques_rnn(ques_emb, quesL)

        # his_feat
        hisL = (his_input!=0).sum(1)
        _, his_f, his_emb = self.his_rnn(his_emb, hisL)
        his_f = his_f.view(-1, rnd, self.nhid)


        encoder_feat = self.recursive(ques_f, his_f, img_f)

        return encoder_feat

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

    def recursive(self, ques_feat, his_feat, img_feat):
        batch_size = ques_feat.size(0)
        ques_g = self.attq(ques_feat.permute(0,2,1)).permute(0,2,1).mean(1)

        ##round1

        _, his_g1 = self.atth(his_feat, ques_g)

        _, img_g1 = self.atti(img_feat, ques_g, his_g1)


        v = self.atti_c(img_g1, ques_g, his_g1)
        h = self.atth_c(his_g1, ques_g, img_g1)


        his_g1 = h
        img_g1 = v

        ##round2


        weight_h2, his_g2 = self.atth(his_feat, ques_g)

        weight_i2, img_g2 = self.atti(img_feat, ques_g, his_g2)



        v = self.atti_c(img_g2, ques_g, his_g2)
        h = self.atth_c(his_g2, ques_g, img_g2)


        his_g2 = h
        img_g2 = v

        ##round3



        weight_h3, his_g = self.atth(his_feat, ques_g, img_g2)

        weight_i3, img_g = self.atti(img_feat, ques_g, his_g)


        img_g = 0.5 * (img_g2 + img_g)
        his_g = 0.5 * (his_g2 + his_g)


        v = self.atti_c(img_g, ques_g, his_g)
        h = self.atth_c(his_g, ques_g, img_g)

        concat_feat = torch.cat((ques_g.view(-1, self.nhid), v.view(-1, self.nhid), \
                                 h.view(-1, self.nhid)), 1)

        encoder_feat = F.tanh(self.fc1(F.dropout(concat_feat, self.d, training=self.training)))

        return encoder_feat
