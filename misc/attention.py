from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
from misc.fc import FCNet
import math

class TanhAttention(nn.Module):
    def __init__(self, dim, num_hid, dropout=0.5):
        super(TanhAttention,self).__init__()
        self.num_hid = num_hid
        self.fc1 = nn.Linear(dim, num_hid)
        self.fc2 = nn.Linear(dim, num_hid)
        self.fc3 = nn.Linear(dim, num_hid)
        self.w = nn.Linear(num_hid,1)
        self.d = dropout

    def forward(self, v, q1, q2):
        """
        v: [batch, k1, vdim]

        """
        batch , att_num, _ = v.size()

        v_proj = self.fc1(v).view(batch,-1,self.num_hid)

        if q1 is not None:
            q1_proj = self.fc2(q1).view(batch,-1,self.num_hid)

            if q2 is not None:
                q2_proj = self.fc3(q2).view(batch,-1,self.num_hid)
                score = F.tanh(v_proj+q1_proj.expand_as(v_proj)+q2_proj.expand_as(v_proj))
            else:
                score = F.tanh(v_proj + q1_proj.expand_as(v_proj))
        else:
            score = F.tanh(v_proj)

        weight = F.softmax(self.w(score).view(-1, 1, att_num),dim=2)
        if self.d > 0.01:
            weight = F.dropout(weight, p=self.d, training=self.training)
        feature = torch.bmm(weight, v)
        weighted_v = weight.transpose(1,2).expand_as(v) * v

        return weight, feature

class TanhAttention4(nn.Module): ######sigmoid
    def __init__(self, dim, num_hid, dropout=0.5):
        super(TanhAttention4,self).__init__()
        self.num_hid = num_hid
        self.fc1 = nn.Linear(dim, num_hid)
        self.fc2 = nn.Linear(dim, num_hid)
        self.fc3 = nn.Linear(dim, num_hid)
        self.w = nn.Linear(num_hid,1)
        self.d = dropout

    def forward(self, v, q1, q2):
        """
        v: [batch, k1, vdim]

        """
        batch ,att_num,_ = v.size()

        v_proj = self.fc1(v).view(batch,-1,self.num_hid)

        if q1 is not None:
            q1_proj = self.fc2(q1).view(batch,-1,self.num_hid)

            if q2 is not None:
                q2_proj = self.fc3(q2).view(batch,-1,self.num_hid)
                score = F.tanh(v_proj+q1_proj.expand_as(v_proj)+q2_proj.expand_as(v_proj))
            else:
                score = F.tanh(v_proj + q1_proj.expand_as(v_proj))
        else:
            score = F.tanh(v_proj)

        weight = F.sigmoid(self.w(score).view(-1, 1, att_num))
        if self.d > 0.01:
            weight = F.dropout(weight, p=self.d, training=self.training)
        feature = torch.bmm(weight, v)
        weighted_v = weight.transpose(1,2).expand_as(v) * v

        return weight, feature

class TanhAttention_C(nn.Module):
    def __init__(self, dim, num_hid, dropout=0.5):
        super(TanhAttention_C,self).__init__()
        self.num_hid = num_hid
        self.op = nn.Parameter(torch.zeros([1,64,1]))
        nn.init.constant(self.op, 1)
        self.fc1 = nn.Linear(dim, num_hid)
        self.fc2 = nn.Linear(dim, num_hid)
        self.w = nn.Linear(64,1)
        self.d = dropout

    def forward(self, v, q1, q2):
        """
        v: [batch, k1, vdim]

        """

        batch = v.size(0)

        # op = self.op.expand([batch, 64, 1])

        q1_proj = self.fc1(q1).view(batch, -1, self.num_hid).expand([batch, 64, self.num_hid])
        q2_proj = self.fc2(q2).view(batch, -1, self.num_hid).expand([batch, 64, self.num_hid])
        score = F.tanh(torch.bmm(self.op.expand([batch, 64, 1]), v) + q1_proj + q2_proj)
        score = score.transpose(1, 2)
        weight = F.sigmoid(self.w(score)).transpose(1, 2)
        feature = weight * v


        return feature

class Attentiontanh(nn.Module):
    def __init__(self, dim, num_hid, dropout=0.5):
        super(Attentiontanh, self).__init__()
        self.num_hid = num_hid
        self.fc1 = nn.Linear(dim, num_hid)
        self.fc2 = nn.Linear(dim, num_hid)
        self.w = nn.Linear(num_hid, 1)
        self.d = dropout

    def forward(self, v, q):

        batch, att_num, _ = v.size()

        v_proj = self.fc1(v).view(batch, -1, self.num_hid)


        q_proj = self.fc2(q).view(batch, -1, self.num_hid)

        score = F.tanh(v_proj + q_proj.expand_as(v_proj))

        weight = F.softmax(self.w(score).view(-1, 1, att_num), dim=2)
        if self.d > 0.01:
            weight = F.dropout(weight, p=self.d, training=self.training)
        feature = torch.bmm(weight, v)

        return weight, feature


# def sum_attention(nnet, query, value, mask=None, dropout=None):
# 	scores = nnet(query).transpose(-2, -1)
# 	if mask is not None:
# 		scores.data.masked_fill_(mask.data.eq(0), -1e9)
#
# 	p_attn = F.softmax(scores, dim=-1)
# 	if dropout is not None:
# 		p_attn = dropout(p_attn)
#
# 	return torch.matmul(p_attn, value), p_attn
#
# class SummaryAttn(nn.Module):
#
# 	def __init__(self, dim, num_attn, dropout, is_cat=False):
# 		super(SummaryAttn, self).__init__()
# 		self.linear = nn.Sequential(
# 				nn.Linear(dim, dim),
# 				nn.ReLU(inplace=True),
# 				nn.Linear(dim, num_attn),
# 			)
# 		self.h = num_attn
# 		self.is_cat = is_cat
# 		self.attn = None
# 		self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
#
# 	def forward(self, query, value, mask=None):
# 		if mask is not None:
# 			mask = mask.unsqueeze(1)
# 		batch = query.size(0)
#
# 		weighted, self.attn = sum_attention(self.linear, query, value, mask=mask, dropout=self.dropout)
# 		weighted = weighted.view(batch, -1) if self.is_cat else weighted.mean(dim=1)
#
# 		return weighted
#
# class NewAttention(nn.Module):
#     def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
#         super(NewAttention, self).__init__()
#
#         self.v_proj = FCNet([v_dim, num_hid])
#         self.q_proj = FCNet([q_dim, num_hid])
#         self.dropout = nn.Dropout(dropout)
#         self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)
#
#     def forward(self, v, q):
#         """
#         v: [batch, k1, vdim]
#         q: [batch, k2, qdim]
#         """
#         att_w = []
#         for i in range(q.data.size()[1]):
#             index = Variable(torch.LongTensor([i]).cuda())
#             att_w.append(self.logits(v, torch.index_select(q, 1, index)).squeeze(dim = 2))
#         # logits = self.logits(v, q)
#         # w = nn.functional.softmax(logits, 1)
#         att_w = torch.stack(att_w,2)
#         att_w_v = torch.sum(att_w,dim = 2)
#         att_w_v = nn.functional.softmax(att_w_v, 1)
#         att_w_q = torch.sum(att_w,dim = 1)
#         att_w_q = nn.functional.softmax(att_w_q, 1)
#         return att_w_v,att_w_q
#
#     def logits(self, v, q):
#         batch, k, _ = v.size()
#         v_proj = self.v_proj(v) # [batch, k, qdim]
#         # q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
#         q_proj = self.q_proj(q)
#         q_proj = q_proj.expand_as(v_proj)
#         joint_repr = v_proj * q_proj
#         joint_repr = self.dropout(joint_repr)
#         logits = self.linear(joint_repr)
#         return logits
#
# class SimpleAttention(nn.Module):
#     def __init__(self, d_k, d_hid, dropout=0.1):
#         super(SimpleAttention, self).__init__()
#         self.d_k = d_k
#         self.dropout = nn.Dropout(dropout)
#         self.W_q = nn.Linear(d_k, d_hid)
#         self.W_k = nn.Linear(d_k, d_hid)
#
#     def forward(self, query, key, value, mask=None, drop=False):
#         """
#         v: [batch, k1, vdim]
#         q: [batch, k2, qdim]
#         """
#         query = self.W_q(query)
#         key = self.W_k(key)
#
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
#         # if drop is True:
#         #     scores = F.dropout(scores, self.dropout, training=self.training)
#
#         if mask is not None:
#             mask = mask.expand_as(scores)
#             scores.data.masked_fill_(mask.eq(0), -1e9)
#
#         p_attn = nn.functional.softmax(scores, dim=-1)
#         if drop is True:
#             p_attn = self.dropout(p_attn)
#         weighted_feat = torch.matmul(p_attn, value)
#
#
#         return weighted_feat, scores
#
#
# class Attention(nn.Module):
#     def __init__(self, channel_q, dim, num_hid, dropout=0.2):
#         super(Attention, self).__init__()
#         self.num_hid = num_hid
#         self.q_proj = weight_norm(nn.Linear(dim, num_hid), dim=None)
#         self.v_proj = weight_norm(nn.Linear(dim, num_hid), dim=None)
#         self.d = dropout
#         self.dropout = nn.Dropout(dropout)
#         self.linear = weight_norm(nn.Linear(channel_q, 1), dim=None)
#         self.channel_q = channel_q
#     def forward(self, v, q):
#         """
#         v: [batch, k1, vdim]
#         q: [batch, k2, qdim]
#         """
#         v_proj = self.v_proj(v)
#         q_proj = self.q_proj(q)
#         atten_map = torch.matmul(v_proj, q_proj.transpose(-2, -1))/math.sqrt(self.num_hid)
#         if self.channel_q >1:
#             score = self.linear(atten_map).squeeze(dim=-1)
#             # score = atten_map.sum(2)
#             weight = F.softmax(score)
#         else:
#             score = atten_map.squeeze()
#             weight = F.softmax(score)
#
#         if self.d > 0.01:
#             weight = self.dropout(weight)
#         # output_v = torch.matmul(weight, v)
#
#         return weight, score
#
#
#
# class TanhAttention_q(nn.Module):
#     def __init__(self, dim, num_hid, dropout=0.5):
#         super(TanhAttention_q,self).__init__()
#         self.num_hid = num_hid
#         self.fc1 = nn.Linear(dim, num_hid)
#         self.fc2 = nn.Linear(dim, num_hid)
#         self.fc3 = nn.Linear(dim, num_hid)
#         self.w = nn.Linear(num_hid,1)
#         self.d = dropout
#
#     def forward(self, v, q1, q2, mask=None):
#         """
#         v: [batch, k1, vdim]
#
#         """
#         batch , att_num, _ = v.size()
#
#         v_proj = self.fc1(v).view(batch,-1,self.num_hid)
#
#         if q1 is not None:
#             q1_proj = self.fc2(q1).view(batch,-1,self.num_hid)
#
#             if q2 is not None:
#                 q2_proj = self.fc3(q2).view(batch,-1,self.num_hid)
#                 score = F.tanh(v_proj+q1_proj.expand_as(v_proj)+q2_proj.expand_as(v_proj))
#             else:
#                 score = F.tanh(v_proj + q1_proj.expand_as(v_proj))
#         else:
#             score = F.tanh(v_proj)
#
#         score = self.w(score).squeeze().t()
#         score.masked_fill_(mask, -99999)
#         weight = F.softmax(score.t(),dim=1).view(-1, 1, att_num)
#         if self.d > 0.01:
#             weight = F.dropout(weight, p=self.d, training=self.training)
#         feature = torch.bmm(weight, v)
#         weighted_v = weight.transpose(1,2).expand_as(v) * v
#
#         return weight, feature
#
# class TanhAttention_dmn(nn.Module):
#     def __init__(self, dim, num_hid, dropout=0.5):
#         super(TanhAttention_dmn,self).__init__()
#         self.num_hid = num_hid
#         self.fc1 = nn.Linear(dim, num_hid)
#         self.fc2 = nn.Linear(dim, num_hid)
#         self.w = nn.Linear(num_hid,1)
#         self.d = dropout
#
#     def forward(self, v, q1):
#         """
#         v: [batch, k1, vdim]
#
#         """
#         batch, att_num, _ = v.size()
#
#         v_proj = self.fc1(v).view(batch,-1,self.num_hid)
#
#         if q1 is not None:
#             q1_proj = self.fc2(q1).view(batch,-1,self.num_hid)
#             score = F.tanh(v_proj + q1_proj.expand_as(v_proj))
#         else:
#             score = F.tanh(v_proj)
#
#         weight = F.softmax(self.w(score).view(-1, 1, att_num),dim=2)
#         if self.d > 0.01:
#             weight = F.dropout(weight, p=self.d, training=self.training)
#
#         return weight.squeeze(1)
#
#
# class TanhAttention_relu(nn.Module):
#     def __init__(self, dim, num_hid, dropout=0.5):
#         super(TanhAttention_relu,self).__init__()
#         self.num_hid = num_hid
#         self.fc1 = nn.Linear(dim, num_hid)
#         self.fc2 = nn.Linear(dim, num_hid)
#         self.fc3 = nn.Linear(dim, num_hid)
#         self.w = nn.Linear(num_hid,1)
#         self.d = dropout
#
#     def forward(self, v, q1, q2):
#         """
#         v: [batch, k1, vdim]
#
#         """
#         batch ,att_num,_ = v.size()
#
#         v_proj = self.fc1(v).view(batch,-1,self.num_hid)
#
#         if q1 is not None:
#             q1_proj = self.fc2(q1).view(batch,-1,self.num_hid)
#             q1_proj = F.relu(q1_proj)
#             if q2 is not None:
#                 q2_proj = self.fc3(q2).view(batch,-1,self.num_hid)
#                 q2_proj = F.relu(q2_proj)
#                 score = F.tanh(v_proj+q1_proj.expand_as(v_proj)+q2_proj.expand_as(v_proj))
#             else:
#                 score = F.tanh(v_proj + q1_proj.expand_as(v_proj))
#         else:
#             score = F.tanh(v_proj)
#
#         weight = F.softmax(self.w(score).view(-1, 1, att_num),dim=2)
#         if self.d > 0.01:
#             weight = F.dropout(weight, p=self.d, training=self.training)
#         feature = torch.bmm(weight, v)
#
#         return weight, feature
#
# class TanhAttention_multihead(nn.Module):
#     def __init__(self, dim, num_hid, dropout=0.5):
#         super(TanhAttention,self).__init__()
#         self.num_hid = num_hid
#         self.fc1 = nn.Linear(dim, num_hid)
#         self.fc2 = nn.Linear(dim, num_hid)
#         self.fc3 = nn.Linear(dim, num_hid)
#         self.w = nn.Linear(self.m_hid, 1)
#         self.d = dropout
#         self.m_hid = 256
#     def forward(self, v, q1, q2):
#         """
#         v: [batch, k1, vdim]
#
#         """
#         batch ,att_num,_ = v.size()
#
#         v_proj = self.fc1(v).view(batch, -1, 4, self.m_hid).transpose(1,2)
#
#         if q1 is not None:
#             q1_proj = self.fc2(q1).view(batch, -1, 4, self.m_hid).transpose(1,2)
#
#             if q2 is not None:
#                 q2_proj = self.fc3(q2).view(batch, -1, 4, self.m_hid).transpose(1,2)
#                 score = F.tanh(v_proj+q1_proj.expand_as(v_proj)+q2_proj.expand_as(v_proj))
#             else:
#                 score = F.tanh(v_proj + q1_proj.expand_as(v_proj))
#         else:
#             score = F.tanh(v_proj)
#
#         weight = F.softmax(self.w(score).view(-1, 4, att_num),dim=2)
#         weight = torch.mean(weight, dim=1)
#         if self.d > 0.01:
#             weight = F.dropout(weight, p=self.d, training=self.training)
#
#         feature = torch.bmm(weight, v)
#
#         return weight, feature
#
#
#
# class TanhAttention4_relu(nn.Module): ######sigmoid
#     def __init__(self, dim, num_hid, dropout=0.5):
#         super(TanhAttention4_relu,self).__init__()
#         self.num_hid = num_hid
#         self.fc1 = nn.Linear(dim, num_hid)
#         self.fc2 = nn.Linear(dim, num_hid)
#         self.fc3 = nn.Linear(dim, num_hid)
#         self.w = nn.Linear(num_hid,1)
#         self.d = dropout
#
#     def forward(self, v, q1, q2):
#         """
#         v: [batch, k1, vdim]
#
#         """
#         batch ,att_num,_ = v.size()
#
#         v_proj = self.fc1(v).view(batch,-1,self.num_hid)
#
#         if q1 is not None:
#             q1_proj = self.fc2(q1).view(batch,-1,self.num_hid)
#             q1_proj = F.relu(q1_proj)
#             if q2 is not None:
#                 q2_proj = self.fc3(q2).view(batch,-1,self.num_hid)
#                 q2_proj = F.relu(q2_proj)
#                 score = F.tanh(v_proj+q1_proj.expand_as(v_proj)+q2_proj.expand_as(v_proj))
#             else:
#                 score = F.tanh(v_proj + q1_proj.expand_as(v_proj))
#         else:
#             score = F.tanh(v_proj)
#
#         weight = F.sigmoid(self.w(score).view(-1, 1, att_num))
#         if self.d > 0.01:
#             weight = F.dropout(weight, p=self.d, training=self.training)
#         feature = torch.bmm(weight, v)
#
#         return weight, feature
#
#
# class TanhAttention_C(nn.Module):
#     def __init__(self, dim, num_hid, dropout=0.5):
#         super(TanhAttention_C,self).__init__()
#         self.num_hid = num_hid
#         self.op = nn.Parameter(torch.zeros([1,64,1]))
#         nn.init.constant(self.op, 1)
#         self.fc1 = nn.Linear(dim, num_hid)
#         self.fc2 = nn.Linear(dim, num_hid)
#         self.w = nn.Linear(64,1)
#         self.d = dropout
#
#     def forward(self, v, q1, q2):
#         """
#         v: [batch, k1, vdim]
#
#         """
#
#         batch = v.size(0)
#
#         # op = self.op.expand([batch, 64, 1])
#
#         q1_proj = self.fc1(q1).view(batch, -1, self.num_hid).expand([batch, 64, self.num_hid])
#         q2_proj = self.fc2(q2).view(batch, -1, self.num_hid).expand([batch, 64, self.num_hid])
#         score = F.tanh(torch.bmm(self.op.expand([batch, 64, 1]), v) + q1_proj + q2_proj)
#         score = score.transpose(1, 2)
#         weight = F.sigmoid(self.w(score)).transpose(1, 2)
#         feature = weight * v
#
#
#         return feature
#
# class TanhAttention_C_Softmax(nn.Module):
#     def __init__(self, dim, num_hid, dropout=0.5):
#         super(TanhAttention_C_Softmax,self).__init__()
#         self.num_hid = num_hid
#         self.op = nn.Parameter(torch.zeros([1,64,1]))
#         nn.init.constant(self.op, 1)
#         self.fc1 = nn.Linear(dim, num_hid)
#         self.fc2 = nn.Linear(dim, num_hid)
#         self.w = nn.Linear(64,1)
#         self.d = dropout
#
#     def forward(self, v, q1, q2):
#         """
#         v: [batch, k1, vdim]
#
#         """
#
#         batch = v.size(0)
#
#
#
#         q1_proj = self.fc1(q1).view(batch, -1, self.num_hid).expand([batch, 64, self.num_hid])
#         q2_proj = self.fc2(q2).view(batch, -1, self.num_hid).expand([batch, 64, self.num_hid])
#         score = F.tanh(torch.bmm(self.op.expand([batch, 64, 1]), v) + q1_proj + q2_proj)
#         score = score.transpose(1, 2)
#         weight = F.softmax(self.w(score)).transpose(1, 2)
#         feature = weight * v
#
#
#         return feature
#
# class TanhAttention2(nn.Module):   ####memory
#     def __init__(self, dim, num_hid, dropout=0.3):
#         super(TanhAttention2,self).__init__()
#         self.num_hid = num_hid
#         self.fc1 = nn.Linear(dim, num_hid)
#         self.fc2 = nn.Linear(dim, num_hid)
#         self.fc3 = nn.Linear(dim, num_hid)
#         self.fcm = nn.Linear(dim, num_hid)
#         self.w = nn.Linear(num_hid,1)
#         self.d = dropout
#
#     def forward(self, v, q1, q2, m):
#         """
#         v: [batch, k1, vdim]
#
#         """
#         batch ,att_num,_ = v.size()
#
#         v_proj = self.fc1(v).view(batch,-1,self.num_hid)
#
#         if q1 is not None:
#             q1_proj = self.fc2(q1).view(batch,-1,self.num_hid)
#
#             if q2 is not None:
#                 q2_proj = self.fc3(q2).view(batch,-1,self.num_hid)
#
#                 if m is not None:
#                     m_proj = self.fcm(m).view(batch,-1,self.num_hid)
#                     score = F.tanh(v_proj + q1_proj.expand_as(v_proj) + q2_proj.expand_as(v_proj) + m_proj.expand_as(v_proj))
#                 else:
#                     score = F.tanh(v_proj+q1_proj.expand_as(v_proj)+q2_proj.expand_as(v_proj))
#             else:
#                 score = F.tanh(v_proj + q1_proj.expand_as(v_proj))
#         else:
#             score = F.tanh(v_proj)
#
#         weight = F.softmax(self.w(score).view(-1, 1, att_num),dim=2)
#         if self.d > 0.01:
#             weight = F.dropout(weight, p=self.d, training=self.training)
#         feature = torch.bmm(weight, v)
#
#         return weight, feature
#
# class TanhAttention3(nn.Module): ####bias distance
#     def __init__(self, dim, num_hid, dropout=0.5, flag=0):
#         super(TanhAttention3,self).__init__()
#         self.num_hid = num_hid
#         self.fc1 = nn.Linear(dim, num_hid)
#         self.fc2 = nn.Linear(dim, num_hid)
#         self.fc3 = nn.Linear(dim, num_hid)
#         self.w = nn.Linear(num_hid,1)
#         self.d = dropout
#         self.use_bias = flag
#         self.D = nn.Parameter(torch.FloatTensor(1))
#         nn.init.constant(self.D, val=1)
#     def forward(self, v, q1, q2):
#         """
#         v: [batch, k1, vdim]
#
#         """
#         batch ,att_num,_ = v.size()
#
#         if self.use_bias == 0:
#             bias = 0
#         else:
#             bias = torch.linspace(0, 1 - att_num, att_num)
#             bias = torch.exp(bias)
#             bias = bias * self.D
#
#
#
#
#         v_proj = self.fc1(v).view(batch,-1,self.num_hid)
#
#         if q1 is not None:
#             q1_proj = self.fc2(q1).view(batch,-1,self.num_hid)
#
#             if q2 is not None:
#                 q2_proj = self.fc3(q2).view(batch,-1,self.num_hid)
#                 score = F.tanh(v_proj+q1_proj.expand_as(v_proj)+q2_proj.expand_as(v_proj))
#             else:
#                 score = F.tanh(v_proj + q1_proj.expand_as(v_proj))
#         else:
#             score = F.tanh(v_proj)
#
#         score = score + bias
#
#
#         weight = F.softmax(self.w(score).view(-1, 1, att_num),dim=2)
#         if self.d > 0.01:
#             weight = F.dropout(weight, p=self.d, training=self.training)
#         feature = torch.bmm(weight, v)
#
#         return weight, feature
#
#
#
# class DotCoAttention(nn.Module):
#     def __init__(self,dim, hid, dropout=0.3, split=0):
#         super(DotCoAttention,self).__init__()
#         self.dim = dim
#         self.hid = hid
#         self.dropout = dropout
#         self.fc_v = FCNet([dim, hid])
#         self.fc_q = FCNet([dim, hid])
#         self.fc_h = FCNet([dim, hid])
#         self.D = nn.Parameter(torch.FloatTensor(1,hid))
#         nn.init.xavier_normal(self.D)
#         self.split = split
#
#     def forward(self,v, q, h):
#         v_proj = self.fc_v(v)
#         q_proj = self.fc_q(q)
#         h_proj = self.fc_h(h)
#
#         if  self.split ==0:
#             return self.dotatt(q_proj, h_proj, v_proj)
#         elif self.split ==1:
#             return self.dotatt(v_proj, q_proj, h_proj)
#         else:
#             return self.dotatt(h_proj, v_proj, q_proj)
#
#     def dotatt(self, v, q1, q2):
#         # q = torch.cat((q1, q2), dim=1)
#         D1 = self.D.expand_as(q1)
#         q1 = D1*q1
#         q1 = q1.transpose(-1, -2)
#         D2 = self.D.expand_as(q2)
#         q2 = D2 * q2
#         q2 = q2.transpose(-1, -2)
#         att_map1 = torch.bmm(v, q1)
#         att_map2 = torch.bmm(v, q2)
#         weight1 = torch.mean(att_map1, 2)
#         weight2 = torch.mean(att_map2, 2)
#         weight = (weight1+weight2)/2
#         weight = F.softmax(weight, dim=1).unsqueeze(2)
#         if self.dropout> 0.01:
#             weight = F.dropout(weight, p=self.dropout, training=self.training)
#
#         output = weight * v
#         return output
#
#
#
#
# class Attention6(nn.Module):
#     '''
#     X*D*Y,duicheng
#
#     '''
#     def __init__(self, dim, hid,dropout=0.5):
#         self.dim = dim
#         self.hid = hid
#         self.dropout = dropout
#         self.fc = FCNet(dim, hid)
#         self.D = nn.Parameter(torch.FloatTensor(hid))
#
#
#     def forward(self,v, q):
#         v_proj = self.fc(v)
#         q_proj = self.fc(q)
#         D = self.D.expand_as(v_proj)
#         v_proj =v_proj*D
#         score = torch.matmul(v_proj,q_proj.transpose(-2,-1)) / math.sqrt(self.hid)
#         weight = F.softmax(score, dim=2)
#         if self.dropout>0.01:
#             weight = nn.Dropout(weight, p=self.dropout)
#         return weight
#
# class Attention7(nn.Module):
#     '''
#     W1X * W2Y
#     gra: return tanh()
#
#     '''
#     def __init__(self, dim, hid,dropout=0.5):
#         super(Attention7, self).__init__()
#         self.dim = dim
#         self.hid = hid
#         self.dropout = dropout
#         self.fc1 = FCNet([dim, hid])
#         self.fc2 = FCNet([dim, hid])
#
#     def forward(self,v, q):
#         v_proj = self.fc1(v)
#         q_proj = self.fc2(q)
#         score = torch.matmul(v_proj,q_proj.transpose(-2,-1)) / math.sqrt(self.hid)
#         weight = F.softmax(score.transpose(-2,-1), dim=2)
#         if self.dropout>0.01:
#             weight =F.dropout(weight, self.dropout, training=self.training)
#         v_att = torch.bmm(weight, v) # or v_proj
#         return weight , v_att
#
# class Attention8(nn.Module):
#     def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
#         super(Attention8, self).__init__()
#
#         self.v_proj = FCNet([v_dim, num_hid])
#         self.q_proj = FCNet([q_dim, num_hid])
#         self.dropout = nn.Dropout(dropout)
#         self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)
#
#     def forward(self, v, q):
#         """
#         v: [batch, k, vdim]
#         q: [batch, qdim]
#         """
#         logits = self.logits(v, q)
#         w = nn.functional.softmax(logits, 1).transpose(-1, -2)
#         feature = torch.bmm(w, v)
#
#         return w, feature
#
#     def logits(self, v, q):
#         batch, k, _ = v.size()
#         v_proj = self.v_proj(v) # [batch, k, qdim]
#         q_proj = self.q_proj(q).repeat(1, k, 1)
#         joint_repr = v_proj * q_proj
#         joint_repr = self.dropout(joint_repr)
#         logits = self.linear(joint_repr)
#         return logits
#
# class gru(nn.Module):
#     def __init__(self, idim, hdim, dropout=0.2):
#         super(gru, self).__init__()
#         self.idim = idim
#         self.hdim = hdim
#         self.fcz = nn.Linear(idim+hdim, 1, bias=False)
#         self.fcr = nn.Linear(idim+hdim, 1, bias=False)
#         self.w = nn.Linear(idim+hdim, hdim, bias=False)
#         self.d = dropout
#     def forward(self, x, h):
#         h = h.view(-1,self.idim)
#         x = x.view(-1, self.hdim)
#         feat = torch.cat((h, x),dim=1)
#         concat1 = F.dropout(feat, p=self.d, training=self.training)
#         concat2 = F.dropout(feat, p=self.d, training=self.training)
#         z = F.sigmoid(self.fcz(concat1)).expand_as(h)
#         r = F.sigmoid(self.fcr(concat2)).expand_as(h)
#         feat2 = F.dropout(torch.cat((r * h, x), dim=1), p=self.d, training=self.training)
#         h2 = F.tanh(self.w(feat2))
#         output = h - z*h+z*h2
#
#         return output
#
# class gru_dmn(nn.Module):
#     def __init__(self, idim, hdim, dropout=0.2):
#         super(gru_dmn, self).__init__()
#         self.idim = idim
#         self.hdim = hdim
#         self.fcr = nn.Linear(idim+hdim, 1, bias=False)
#         self.w = nn.Linear(idim+hdim, hdim, bias=False)
#         self.d = dropout
#     def forward(self, x, h, z):
#         h = h.view(-1,self.idim)
#         x = x.contiguous().view(-1, self.hdim)
#         feat = torch.cat((h, x),dim=1)
#         concat2 = F.dropout(feat, p=self.d, training=self.training)
#         r = F.sigmoid(self.fcr(concat2)).expand_as(h)
#         feat2 = F.dropout(torch.cat((r * h, x), dim=1), p=self.d, training=self.training)
#         h2 = F.tanh(self.w(feat2))
#         output = h - z*h+z*h2
#
#         return output

