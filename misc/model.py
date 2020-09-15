import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import math
import numpy as np
import torch.nn.functional as F
from misc.share_Linear import share_Linear
from torch.distributions import Categorical
from misc.utils import repackage_hidden, sample_batch_neg

class RNNEncoder(nn.Module):
    def __init__(self, word_size, hidden_size, bidirectional=True,
                 drop_prob=0, n_layers=1, rnn_type='lstm'):
        super(RNNEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=drop_prob)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, embedded, input_lengths):
        """
        Inputs:
        - input_labels: long (batch, seq_len)
        Outputs:
        - output  : float (batch, max_len, hidden_size * num_dirs)
        - hidden  : float (batch, num_layers * num_dirs * hidden_size)
        - embedded: float (batch, max_len, word_vec_size)
        """
        # make ixs
        batch_size = input_lengths.size(0)
        mask0 = input_lengths.eq(0)
        input_lengths.masked_fill_(mask0, 1)
        input_lengths_list = input_lengths.data.cpu().numpy().tolist()
        sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist()  # list of sorted input_lengths
        sort_ixs = np.argsort(input_lengths_list)[::-1].tolist()  # list of int sort_ixs, descending
        s2r = {s: r for r, s in enumerate(sort_ixs)}  # O(n)
        recover_ixs = [s2r[s] for s in range(batch_size)]  # list of int recover ixs

        # sort input_labels by descending order
        embedded = embedded[sort_ixs]

        # embed
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)

        # forward rnn
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(embedded)

        # recover
        # embedded (batch, seq_len, word_vec_size)
        embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
        embedded = embedded[recover_ixs]

        # recover rnn
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
        output = output[recover_ixs]

        # recover hidden
        if self.rnn_type == 'lstm':
            hidden = hidden[0]  # we only use hidden states for the final hidden representation
        hidden = hidden[:, recover_ixs, :]  # (num_layers * num_dirs, batch, hidden_size)
        hidden = hidden.transpose(0, 1).contiguous()  # (batch, num_layers * num_dirs, hidden_size)
        hidden = hidden.view(hidden.size(0), -1)  # (batch, num_layers * num_dirs * hidden_size)

        return output, hidden, embedded

class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b

class _netW(nn.Module):
    def __init__(self, ntoken, ninp, dropout):
        super(_netW, self).__init__()
        self.word_embed = nn.Embedding(ntoken, ninp).cuda()
        self.Linear = share_Linear(self.word_embed.weight).cuda()
        self.init_weights()
        self.d = dropout

    def init_weights(self):
        initrange = 0.1
        self.word_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, format ='index'):
        if format == 'onehot':
            out = F.dropout(self.Linear(input), self.d, training=self.training)
        elif format == 'index':
            out = F.dropout(self.word_embed(input), self.d, training=self.training)

        return out

    # def init_hidden(self, bsz):
    #     weight = next(self.parameters()).data
    #     if self.rnn_type == 'LSTM':
    #         return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
    #                 Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
    #     else:
    #         return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

class _netD(nn.Module):
    """
    Given the real/wrong/fake answer, use a RNN (LSTM) to embed the answer.
    """

    def __init__(self, rnn_type, ninp, nhid, nlayers, ntoken, dropout):
        super(_netD, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken
        self.ninp = ninp
        self.d = dropout

        # self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        self.rnn = RNNEncoder(word_size=ninp,
                              hidden_size=int(nhid / 2),
                              bidirectional=True,
                              drop_prob=dropout if nlayers > 1 else 0,
                              n_layers=nlayers,
                              rnn_type='lstm')
        self.W1 = nn.Linear(self.nhid, self.nhid)
        self.W2 = nn.Linear(self.nhid, 1)
        self.fc = nn.Linear(nhid, ninp)

    def forward(self, input_feat, idx, vocab_size):

        ansL = (idx != 0).sum(1)
        output, hidden, _ = self.rnn(input_feat, ansL)
        # mask = idx.data.eq(0)  # generate the mask
        # mask[idx.data == vocab_size] = 1 # also set the last token to be 1
        # if isinstance(input_feat, Variable):
        #     mask = Variable(mask, volatile=input_feat.volatile)

        # Doing self attention here.
        atten = self.W2(F.dropout(F.tanh(self.W1(output.view(-1, self.nhid))), self.d, training=self.training)).view(
            idx.size(0), -1)
        # atten.masked_fill_(mask, -99999)
        weight = F.softmax(atten, dim=1).view(idx.size(0), 1, -1)
        feat = torch.bmm(weight, output).view(-1, self.nhid)
        feat = F.dropout(feat, self.d, training=self.training)
        transform_output = F.tanh(self.fc(feat))

        return transform_output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def init_bi_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

class _netD_bi(nn.Module):
    """
    Given the real/wrong/fake answer, use a RNN (LSTM) to embed the answer.
    """
    def __init__(self, rnn_type, ninp, nhid, nlayers, ntoken, dropout):
        super(_netD_bi, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken
        self.ninp = ninp
        self.d = dropout

        self.rnn =nn.LSTM(ninp, int(nhid/2), nlayers, dropout=dropout, bidirectional=True)
        self.W1 = nn.Linear(self.nhid, self.nhid)
        self.W2 = nn.Linear(self.nhid, 1)
        self.fc = nn.Linear(nhid, ninp)

    def forward(self, input_feat, idx, hidden, vocab_size):

        output, _ = self.rnn(input_feat, hidden)


        mask = idx.data.eq(0)  # generate the mask
        mask[idx.data == vocab_size] = 1 # also set the last token to be 1
        if isinstance(input_feat, Variable):
            mask = Variable(mask, volatile=input_feat.volatile)

        # Doing self attention here.
        atten = self.W2(F.dropout(F.tanh(self.W1(output.view(-1, self.nhid))), self.d, training=self.training)).view(idx.size())
        atten.masked_fill_(mask, -99999)
        weight = F.softmax(atten.t(),dim=1).view(-1,1,idx.size(0))
        feat = torch.bmm(weight, output.transpose(0,1)).view(-1,self.nhid)
        feat = F.dropout(feat, self.d, training=self.training)
        transform_output = F.tanh(self.fc(feat))

        return transform_output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers*2, bsz, int(self.nhid/2)).zero_()),
                    Variable(weight.new(self.nlayers*2, bsz, int(self.nhid/2)).zero_()))
        else:
            return Variable(weight.new(self.nlayers*2, bsz, int(self.nhid/2)).zero_())

    def init_bi_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

class  LMCriterion(nn.Module):

    def __init__(self):
        super(LMCriterion, self).__init__()

    def forward(self, input, target):
        logprob_select = torch.gather(input, 1, target)

        mask = target.data.gt(0)  # generate the mask
        if isinstance(input, Variable):
            mask = Variable(mask, volatile=input.volatile)
        
        out = torch.masked_select(logprob_select, mask)

        loss = -torch.sum(out) # get the average loss.
        return loss

class nPairLoss(nn.Module):
    """
    Given the right, fake, wrong, wrong_sampled embedding, use the N Pair Loss
    objective (which is an extension to the triplet loss)

    Loss = log(1+exp(feat*wrong - feat*right + feat*fake - feat*right)) + L2 norm.

    Improved Deep Metric Learning with Multi-class N-pair Loss Objective (NIPS)
    """
    def __init__(self, ninp, margin):
        super(nPairLoss, self).__init__()
        self.ninp = ninp
        self.margin = np.log(margin)

    def forward(self, feat, right, wrong, batch_wrong, fake=None, fake_diff_mask=None):

        num_wrong = wrong.size(1)
        batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        wrong_dis = torch.bmm(wrong, feat)
        batch_wrong_dis = torch.bmm(batch_wrong, feat)

        wrong_score = torch.sum(torch.exp(wrong_dis - right_dis.expand_as(wrong_dis)),1) \
                + torch.sum(torch.exp(batch_wrong_dis - right_dis.expand_as(batch_wrong_dis)),1)

        loss_dis = torch.sum(torch.log(wrong_score + 1))
        loss_norm = right.norm() + feat.norm() + wrong.norm() + batch_wrong.norm()


        loss = (loss_dis + 0.1 * loss_norm) / batch_size

        return loss

class nPairLoss3(nn.Module):
    """
    Given the right, fake, wrong, wrong_sampled embedding, use the N Pair Loss
    objective (which is an extension to the triplet loss)

    Loss = log(1+exp(feat*wrong - feat*right + feat*fake - feat*right)) + L2 norm.

    Improved Deep Metric Learning with Multi-class N-pair Loss Objective (NIPS)
    """
    def __init__(self, ninp, margin):
        super(nPairLoss3, self).__init__()
        self.ninp = ninp
        self.margin = np.log(margin)

    def forward(self, feat, right, wrong, batch_wrong, fake=None, fake_diff_mask=None):

        num_wrong = wrong.size(1)
        batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        wrong_dis = torch.bmm(wrong, feat)
        batch_wrong_dis = torch.bmm(batch_wrong, feat)
        dis = torch.cat((wrong_dis, batch_wrong_dis), dim=1)

        wrong_score = torch.sum((torch.exp((dis - right_dis.expand_as(dis)))),1)


        loss_dis = torch.sum(torch.log(wrong_score + 1))
        loss_norm = right.norm() + feat.norm() + wrong.norm() #+ batch_wrong.norm()


        loss = (loss_dis + 0.1 * loss_norm) / batch_size

        return loss


class nPairLoss2(nn.Module):

    def __init__(self, ninp, margin):
        super(nPairLoss2, self).__init__()
        self.ninp = ninp
        self.margin = np.log(margin)

    def forward(self, feat, right, wrong):

        num_wrong = wrong.size(1)
        batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        wrong_dis = torch.bmm(wrong, feat)

        wrong_score = torch.sum(torch.exp(wrong_dis - right_dis.expand_as(wrong_dis)), 1)


        loss_dis = torch.sum(torch.log(wrong_score + 1))
        loss_norm = right.norm() + feat.norm() + wrong.norm()


        loss = (loss_dis + 0.1 * loss_norm) / batch_size
        return loss

class G_loss(nn.Module):
    """
    Generator loss:
    minimize right feature and fake feature L2 norm.
    maximinze the fake feature and wrong feature.
    """
    def __init__(self, ninp):
        super(G_loss, self).__init__()
        self.ninp = ninp

    def forward(self, feat, right, fake):

        #num_wrong = wrong.size(1)
        batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        #wrong_dis = torch.bmm(wrong, feat)
        #batch_wrong_dis = torch.bmm(batch_wrong, feat)
        fake_dis = torch.bmm(fake.view(-1, 1, self.ninp), feat)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)

        fake_score = torch.exp(right_dis - fake_dis)
        loss_fake = torch.sum(torch.log(fake_score + 1))

        loss_norm = feat.norm() + fake.norm() + right.norm()
        loss = (loss_fake + 0.1 * loss_norm) / batch_size

        return loss, loss_fake.data[0]/batch_size

class gumbel_sampler(nn.Module):
    def __init__(self):
        super(gumbel_sampler, self).__init__()

    def forward(self, input, noise, temperature=0.5):

        eps = 1e-20
        noise.data.add_(eps).log_().neg_()
        noise.data.add_(eps).log_().neg_()
        y = (input + noise) / temperature
        y = F.softmax(y)

        max_val, max_idx = torch.max(y, y.dim()-1)
        y_hard = y == max_val.view(-1,1).expand_as(y)
        y = (y_hard.float() - y).detach() + y

        # log_prob = input.gather(1, max_idx.view(-1,1)) # gather the logprobs at sampled positions

        return y, max_idx.view(1, -1)#, log_prob

class AxB(nn.Module):
    def __init__(self, nhid):
        super(AxB, self).__init__()
        self.nhid = nhid

    def forward(self, nhA, nhB):
        mat = torch.bmm(nhB.view(-1, 100, self.nhid), nhA.view(-1,self.nhid,1))
        return mat.view(-1,100)

class gc_sampler(nn.Module):

    def __init__(self, ninp, margin):
        super(gc_sampler, self).__init__()
        self.ninp = ninp


    def forward(self, feat, wrong):

        # num_wrong = wrong.size(1)
        # batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        # right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        wrong_dis = torch.bmm(wrong, feat)
        # gc_feat = F.softmax(torch.cat((right_dis, wrong_dis), dim=1), dim=1).squeeze(-1)
        gc_feat = F.softmax(wrong_dis, dim=1).squeeze(-1)
        m = Categorical(gc_feat)
        action = m.sample()
        a_prob = m.log_prob(action)

        return action, a_prob, m

class gc_sampler_v2(nn.Module):

    def __init__(self, ninp, sample_num):
        super(gc_sampler_v2, self).__init__()
        self.ninp = ninp
        self.num = sample_num

    def forward(self, feat, right, wrong):


        feat = feat.view(-1, self.ninp, 1)
        ans_rw = torch.cat((right.view(-1, 1, self.ninp), wrong), dim=1)
        rw_dis = torch.bmm(ans_rw, feat)
        gc_prob = F.softmax(rw_dis, dim=1).squeeze(-1)

        sorted_prob , gc_idx = torch.sort(gc_prob[:, 1:], 1, descending=True)
        sample_prob = sorted_prob[:, :self.num]
        sample_idx = gc_idx[:, :self.num]

        return sample_prob, sample_idx

class gt_sampler(nn.Module):

    def __init__(self, ninp, sample_num):
        super(gt_sampler, self).__init__()
        self.ninp = ninp
        self.num = sample_num

    def forward(self, feat, right, wrong):


        feat = feat.view(-1, self.ninp, 1)
        ans_rw = torch.cat((right.view(-1, 1, self.ninp), wrong), dim=1)
        rw_dis = torch.bmm(ans_rw, feat)
        gc_prob = F.softmax(rw_dis, dim=1).squeeze(-1)
        gt_prob = gc_prob[:, 0]
        return gt_prob

def gc_encoder(img_input, question, history, answerT, ques_input, his_input, ans_target, memory, rnd, netW, netE, netD):
    batch_size = question.size(0)

    ques = question[:, rnd, :].t()
    his = history[:, :rnd + 1, :].clone().view(-1, 24).t()
    tans = answerT[:, rnd, :].t()
    ques_input.data.resize_(ques.size()).copy_(ques)
    his_input.data.resize_(his.size()).copy_(his)
    ans_target.data.resize_(tans.size()).copy_(tans)
    ques_emb = netW(ques_input, format='index')
    his_emb = netW(his_input, format='index')

    ques_hidden = netE.init_bi_hidden(batch_size)
    hist_hidden = netE.init_bi_hidden(his_emb.size(1))

    real_hidden = netD.init_hidden(batch_size)


    featD, memory, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                      ques_hidden, hist_hidden, memory, rnd + 1)
    ans_real_emb = netW(ans_target, format='index')
    real_feat = netD(ans_real_emb, ans_target, real_hidden, 10364)
    score = torch.bmm(real_feat.view(-1, 1, 300), featD.view(-1, 300, 1))


    return score


def gc_encoder_v2(img_input, question, history, answerT, opt_answerT, batch_sample_idx, ques_input, \
           his_input, ans_target, wrong_ans_input, memory, rnd, netW, netE, netD):
    batch_size = question.size(0)

    ques = question[:, rnd, :].t()
    his = history[:, :rnd + 1, :].clone().view(-1, 24).t()
    tans = answerT[:, rnd, :].t()
    wrong_ans = opt_answerT[:, rnd, :].clone().view(-1, 9).t()

    ques_input.data.resize_(ques.size()).copy_(ques)
    his_input.data.resize_(his.size()).copy_(his)

    ans_target.data.resize_(tans.size()).copy_(tans)
    wrong_ans_input.data.resize_(wrong_ans.size()).copy_(wrong_ans)


    ques_emb = netW(ques_input, format='index')
    his_emb = netW(his_input, format='index')

    ques_hidden = netE.init_bi_hidden(batch_size)
    hist_hidden = netE.init_bi_hidden(his_emb.size(1))


    featD, memory, _ = netE(ques_emb, his_emb, img_input, \
                                      ques_hidden, hist_hidden, memory, rnd + 1)

    ans_real_emb = netW(ans_target, format='index')
    ans_wrong_emb = netW(wrong_ans_input, format='index')

    real_hidden = netD.init_hidden(batch_size)
    wrong_hidden = netD.init_hidden(ans_wrong_emb.size(1))

    real_feat = netD(ans_real_emb, ans_target, real_hidden, 10364)
    wrong_feat = netD(ans_wrong_emb, wrong_ans_input, wrong_hidden, 10364)

    batch_wrong_feat = wrong_feat.index_select(0, batch_sample_idx.view(-1))
    wrong_feat = wrong_feat.view(batch_size, -1, 300)
    batch_wrong_feat = batch_wrong_feat.view(batch_size, -1, 300)

    return featD, real_feat, wrong_feat, batch_wrong_feat, memory

class nPairLoss_gc(nn.Module):
    """
    Given the right, fake, wrong, wrong_sampled embedding, use the N Pair Loss
    objective (which is an extension to the triplet loss)

    Loss = log(1+exp(feat*wrong - feat*right + feat*fake - feat*right)) + L2 norm.

    Improved Deep Metric Learning with Multi-class N-pair Loss Objective (NIPS)
    """
    def __init__(self, ninp, margin):
        super(nPairLoss_gc, self).__init__()
        self.ninp = ninp
        self.margin = np.log(margin)

    def forward(self, feat, right, wrong, batch_wrong, fake=None, fake_diff_mask=None):

        num_wrong = wrong.size(1)
        batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        wrong_dis = torch.bmm(wrong, feat)
        batch_wrong_dis = torch.bmm(batch_wrong, feat)

        if batch_wrong:
            wrong_score = torch.sum(torch.exp(wrong_dis - right_dis.expand_as(wrong_dis)),1) \
                    + torch.sum(torch.exp(batch_wrong_dis - right_dis.expand_as(batch_wrong_dis)),1)

            loss_dis = torch.log(wrong_score + 1)
            loss_norm = right.norm() + feat.norm() + wrong.norm() + batch_wrong.norm()
        else:
            wrong_score = torch.sum(torch.exp(wrong_dis - right_dis.expand_as(wrong_dis)), 1)
            loss_dis = torch.log(wrong_score + 1)
            loss_norm = right.norm() + feat.norm() + wrong.norm()

        loss = (loss_dis + 0 * loss_norm) / batch_size
        return loss, loss_norm
        # if fake:
        #     fake_dis = torch.bmm(fake.view(-1, 1, self.ninp), feat)
        #     fake_score = torch.masked_select(torch.exp(fake_dis - right_dis), fake_diff_mask)
        #
        #     margin_score = F.relu(torch.log(fake_score + 1) - self.margin)
        #     loss_fake = torch.sum(margin_score)
        #     loss_dis += loss_fake
        #     loss_norm += fake.norm()

        loss = (loss_dis + 0.1 * loss_norm) / batch_size
        # if fake:
        #     return loss, loss_fake.data[0] / batch_size
        # else:
        #     return loss_dis, loss_norm

class nPairLoss_ft(nn.Module):
    """
    Given the right, fake, wrong, wrong_sampled embedding, use the N Pair Loss
    objective (which is an extension to the triplet loss)

    Loss = log(1+exp(feat*wrong - feat*right + feat*fake - feat*right)) + L2 norm.

    Improved Deep Metric Learning with Multi-class N-pair Loss Objective (NIPS)
    """

    def __init__(self, ninp, margin):
        super(nPairLoss_ft, self).__init__()
        self.ninp = ninp
        self.margin = np.log(margin)

    def forward(self, feat, right, wrong, batch_wrong, fake=None, fake_diff_mask=None):

        num_wrong = wrong.size(1)
        batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        wrong_dis = torch.bmm(wrong, feat)


        if batch_wrong:
            batch_wrong_dis = torch.bmm(batch_wrong, feat)
            wrong_score = torch.sum(torch.exp(wrong_dis - right_dis.expand_as(wrong_dis)), 1) \
                      + torch.sum(torch.exp(batch_wrong_dis - right_dis.expand_as(batch_wrong_dis)), 1)

            loss_dis = torch.sum(torch.log(wrong_score + 1))

        else:
            wrong_score = torch.sum(torch.exp(wrong_dis - right_dis.expand_as(wrong_dis)), 1)
            loss_dis = torch.sum(torch.log(wrong_score + 1))


        loss = (loss_dis) / batch_size
        return loss

class XELoss(nn.Module):

    def __init__(self, ninp, margin):
        super(XELoss, self).__init__()
        self.ninp = ninp


    def forward(self, feat, right, wrong):

        num_wrong = wrong.size(1)
        batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        wrong_dis = torch.bmm(wrong, feat)

        score = torch.cat((right_dis, wrong_dis),dim=1).squeeze(-1)
        prob_rw = F.softmax(score, dim=1)
        prob_r = prob_rw[:,0].contiguous()
        count = prob_rw.gt(prob_r.view(-1, 1).expand_as(prob_rw))
        rank = count.sum(1) + 1
        rankdata=rank.data
        # rank.type('torch.cuda.FloatTensor')
        # rank = 2-torch.reciprocal(rank)
        prob = -torch.log(prob_r)
        loss=0
        loss_gc = torch.ones_like(prob)
        for bi in range(batch_size):
            loss = loss + prob[bi]*(2-1/float(rankdata[bi]))
            loss_gc[bi] = prob[bi]*(2-1/float(rankdata[bi]))
        loss = loss/batch_size

        return loss, loss_gc

class MRRLoss(nn.Module):

    def __init__(self, ninp, margin):
        super(MRRLoss, self).__init__()
        self.ninp = ninp


    def forward(self, feat, right, wrong):

        num_wrong = wrong.size(1)
        batch_size = feat.size(0)

        feat = feat.view(-1, self.ninp, 1)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        wrong_dis = torch.bmm(wrong, feat)

        score = torch.cat((right_dis, wrong_dis),dim=1).squeeze(-1)
        prob_rw = F.softmax(score, dim=1)
        prob_r = prob_rw[:,0].contiguous()
        count = prob_rw.gt(prob_r.view(-1, 1).expand_as(prob_rw))
        count = count.float()
        rank = count.sum(1) + 1

        rank.reciprocal_()
        rankdata=-1*rank



        return None, rankdata