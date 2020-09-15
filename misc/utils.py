import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
"""
Some utility Functions.
"""

def repackage_hidden_volatile(h):
    if type(h) == Variable:
        return Variable(h.data, volatile=True)
    else:
        return tuple(repackage_hidden_volatile(v) for v in h)

def repackage_hidden(h, batch_size):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data.resize_(h.size(0), batch_size, h.size(2)).zero_())
    else:
        return tuple(repackage_hidden(v, batch_size) for v in h)

def clip_gradient(model):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        p.grad.data.clamp_(-5, 5)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    if epoch<14:
        lr = lr
    elif epoch<18:
        lr = 0.5*lr
    else:
        lr = 0.25*lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
def decode_txt(itow, x):
    """Function to show decode the text."""
    out = []
    for b in range(x.size(1)):
        txt = ''
        for t in range(x.size(0)):
            idx = x[t,b]
            if idx == 0 or idx == len(itow)+1:
                break
            txt += itow[str(int(idx))]
            txt += ' '
        out.append(txt)

    return out

def l2_norm(input):
    """
    input: feature that need to normalize.
    output: normalziaed feature.
    """
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

class DynamicRNN(nn.Module):
    def __init__(self, rnn_model):
        super().__init__()
        self.rnn_model = rnn_model

    def forward(self, seq_input, seq_lens, initial_state=None):
        """A wrapper over pytorch's rnn to handle sequences of variable length.
        Arguments
        ---------
        seq_input : torch.Tensor
            Input sequence tensor (padded) for RNN model.
            Shape: (batch_size, max_sequence_length, embed_size)
        seq_lens : torch.LongTensor
            Length of sequences (b, )
        initial_state : torch.Tensor
            Initial (hidden, cell) states of RNN model.
        Returns
        -------
            A single tensor of shape (batch_size, rnn_hidden_size) corresponding
            to the outputs of the RNN model at the last time step of each input
            sequence.
        """
        max_sequence_length = seq_input.size(1)
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(
            sorted_seq_input, lengths=sorted_len, batch_first=True)

        if initial_state is not None:
            hx = initial_state
            sorted_hx = [x.index_select(1, fwd_order) for x in hx]
            assert hx[0].size(0) == self.rnn_model.num_layers
        else:
            hx = None

        self.rnn_model.flatten_parameters()
        outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, hx)

        # pick hidden and cell states of last layer
        h_n = h_n[-1].index_select(dim=0, index=bwd_order)
        c_n = c_n[-1].index_select(dim=0, index=bwd_order)

        outputs = pad_packed_sequence(
            outputs, batch_first=True, total_length=max_sequence_length
        )
        return outputs, (h_n, c_n)

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(lens.contiguous().view(-1), 0, descending=True)
        _, bwd_order = torch.sort(fwd_order)
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order

def sample_batch_neg(answerIdx, negAnswerIdx, sample_idx, num_sample):
    """
    input:
    answerIdx: batch_size
    negAnswerIdx: batch_size x opt.negative_sample

    output:
    sample_idx = batch_size x num_sample
    """

    batch_size = answerIdx.size(0)
    num_neg = negAnswerIdx.size(0) * negAnswerIdx.size(1)
    negAnswerIdx = negAnswerIdx.clone().view(-1)
    for b in range(batch_size):
        gt_idx = answerIdx[b]
        for n in range(num_sample):
            while True:
                rand = int(random.random() * num_neg)
                neg_idx = negAnswerIdx[rand]
                if gt_idx != neg_idx:
                    sample_idx.data[b, n] = rand
                    break
