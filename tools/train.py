from __future__ import print_function

import argparse
import os
import random
import sys

sys.path.append(os.getcwd())

import pdb
import time
import numpy as np
import json
import sys

sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from misc.glove import load_glove
from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, \
    decode_txt, sample_batch_neg, l2_norm

import misc.dataLoader as dl
import misc.model as model
from misc.encoder_r import _netE
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--input_img_h5', default='data/img_train_resnet.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_img_val_h5', default='data/img_val_resnet.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_ques_h5', default='script/visdial_data_1.0.h5', help='path to dataset, now hdf5 file')
parser.add_argument('--input_json', default='script/visdial_params_1.0.json', help='path to dataset, now hdf5 file')
parser.add_argument('--outf', default='./save_1.0/D', help='folder to output images and model checkpoints')
parser.add_argument('--decoder', default='D', help='what decoder to use.')
parser.add_argument('--model_path', default='', help='folder to output images and model checkpoints')
parser.add_argument('--num_val', default=1000, help='number of image split out as validation set.')

parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--negative_sample', type=int, default=20, help='folder to output images and model checkpoints')
parser.add_argument('--neg_batch_sample', type=int, default=30, help='folder to output images and model checkpoints')
parser.add_argument('--start_epoch', type=int, default=1, help='start of epochs to train for')


parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--save_iter', type=int, default=5, help='number of epochs to train for')

parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--lr', type=float, default=0.004, help='learning rate for, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')


parser.add_argument('--conv_feat_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=512, help='input batch size')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--ninp', type=int, default=300, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1024, help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=int, default=0.5, help='number of layers')
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--margin', type=float, default=2, help='number of epochs to train for')
parser.add_argument('--log_interval', type=int, default=50, help='how many iterations show the log info')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)

cudnn.benchmark = True

torch.cuda.set_device(0)



if opt.model_path != '':
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path)
    opt.manualSeed = checkpoint['opt'].manualSeed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.manualSeed)
    model_path = opt.model_path
    opt.start_epoch = checkpoint['epoch']
    opt.model_path = model_path
    opt.batchSize = 100
    # create new folder.
    t = datetime.datetime.now()
    cur_time = '%s-%s-%s' % (t.month, t.day, t.hour)
    save_path = os.path.join(opt.outf, 'enc' + cur_time)
    try:
        os.makedirs(save_path)
    except OSError:
        pass


else:
    # create new folder.
    t = datetime.datetime.now()
    cur_time = '%s-%s-%s' % (t.month, t.day, t.hour)
    save_path = os.path.join(opt.outf, 'enc' + cur_time)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

####################################################################################
# Data Loader
####################################################################################

dataset = dl.train(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
                   input_json=opt.input_json, negative_sample=opt.negative_sample,
                   num_val=opt.num_val, data_split='train', split=0)


dataset_val = dl.validate(input_img_h5=opt.input_img_val_h5, input_ques_h5=opt.input_ques_h5,
                          input_json=opt.input_json, negative_sample=opt.negative_sample,
                          num_val=opt.num_val, data_split='val')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=8)


dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=10,
                                             shuffle=False, num_workers=8)

####################################################################################
# Build the Model
####################################################################################
n_neg = opt.negative_sample
vocab_size = dataset.vocab_size
ques_length = dataset.ques_length
ans_length = dataset.ans_length
his_length = dataset.ans_length + dataset.ques_length
itow = dataset.itow
opt.word_size = opt.ninp
img_feat_size = 2048

netE = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size).cuda()
netW = model._netW(vocab_size, opt.ninp, opt.dropout).cuda()
netD = model._netD(opt.model, opt.ninp, opt.nhid, opt.nlayers, vocab_size, opt.dropout).cuda()
critD = model.nPairLoss(opt.ninp, opt.margin).cuda()
opt.glove = 'data/glove'
word2idx = {word: int(i) for i, word in itow.items()}
if opt.model_path == '':
    glove_weight = load_glove(glove=opt.glove, vocab=word2idx, opt=opt)
    print(glove_weight.shape)
    assert glove_weight.shape == netW.word_embed.weight.size()
    netW.word_embed.weight.data.set_(torch.cuda.FloatTensor(glove_weight))
    print("Load word vectors ...Done.")

if opt.model_path != '':  # load the pre-trained model.
    netW.load_state_dict(checkpoint['netW'])
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])


####################################################################################
# training model
####################################################################################
def train(epoch):
    netW.train()
    netE.train()
    netD.train()

    lr = adjust_learning_rate(optimizer, epoch, opt.lr)

    ques_hidden = netE.init_bi_hidden(opt.batchSize)
    hist_hidden = netE.init_bi_hidden(opt.batchSize)

    real_hidden = netD.init_hidden(opt.batchSize)
    wrong_hidden = netD.init_hidden(opt.batchSize)


    data_iter = iter(dataloader)
    num = len(dataloader)

    average_loss = 0
    total_loss = 0
    count = 0
    i = 0
    while i < num:

        data = data_iter.next()
        image, history, question, answer, answerLen, answerIdx, questionL, \
        opt_answerT, opt_answerLen, opt_answerIdx = data

        batch_size = question.size(0)
        image = image.view(-1, img_feat_size)
        img_input.data.resize_(image.size()).copy_(image)



        for rnd in range(10):
            netW.zero_grad()
            netE.zero_grad()
            netD.zero_grad()


            # get the corresponding round QA and history.
            ques = question[:, rnd, :]
            his = history[:, :rnd + 1, :].clone().view(-1, his_length)

            tans = answer[:, rnd, :]
            # tans = answerT[:, rnd, :]
            wrong_ans = opt_answerT[:, rnd, :].clone().view(-1, ans_length)


            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)

            # ans_input.data.resize_(ans.size()).copy_(ans)
            ans_target.data.resize_(tans.size()).copy_(tans)
            wrong_ans_input.data.resize_(wrong_ans.size()).copy_(wrong_ans)

            # sample in-batch negative index
            batch_sample_idx.data.resize_(batch_size, opt.neg_batch_sample).zero_()
            sample_batch_neg(answerIdx[:, rnd], opt_answerIdx[:, rnd, :], batch_sample_idx, opt.neg_batch_sample)

            ques_emb = netW(ques_input, format='index')
            his_emb = netW(his_input, format='index')

            featD,_ = netE(ques_emb, his_emb, img_input, ques_input, his_input, rnd + 1)


            ans_real_emb = netW(ans_target, format='index')
            ans_wrong_emb = netW(wrong_ans_input, format='index')

            # real_hidden = repackage_hidden(real_hidden, batch_size)
            # wrong_hidden = repackage_hidden(wrong_hidden, ans_wrong_emb.size(1))

            real_feat = netD(ans_real_emb, ans_target, vocab_size)
            wrong_feat = netD(ans_wrong_emb, wrong_ans_input, vocab_size)

            batch_wrong_feat = wrong_feat.index_select(0, batch_sample_idx.view(-1))
            wrong_feat = wrong_feat.view(batch_size, -1, opt.ninp)
            batch_wrong_feat = batch_wrong_feat.view(batch_size, -1, opt.ninp)

            nPairLoss = critD(featD, real_feat, wrong_feat, batch_wrong_feat)

            average_loss += nPairLoss.data[0]
            nPairLoss.backward()

            optimizer.step()
            count += 1

        i += 1
        if i % opt.log_interval == 0:
            average_loss /= count
            print("step {} / {} (epoch {}), loss {:.3f}, lr = {:.6f}" \
                  .format(i, len(dataloader), epoch, average_loss, lr))
            total_loss = total_loss + average_loss
            average_loss = 0
            count = 0




    return total_loss, lr


def val():
    netE.eval()
    netW.eval()
    netD.eval()

    n_neg = 100
    data_iter_val = iter(dataloader_val)
    ques_hidden = netE.init_bi_hidden(opt.batchSize)
    hist_hidden = netE.init_bi_hidden(opt.batchSize)

    opt_hidden = netD.init_hidden(opt.batchSize)
    i = 0

    average_loss = 0
    rank_all_tmp = []

    while i < len(dataloader_val):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
        opt_answerT, answer_ids, answerLen, opt_answerLen, img_id = data

        batch_size = question.size(0)
        image = image.view(-1, img_feat_size)
        # image = l2_norm(image)
        img_input.data.resize_(image.size()).copy_(image)




        for rnd in range(10):

            ques = question[:, rnd, :]
            his = history[:, :rnd + 1, :].clone().view(-1, his_length)

            opt_ans = opt_answerT[:, rnd, :].clone().view(-1, ans_length)
            gt_id = answer_ids[:, rnd]

            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)

            opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)
            gt_index.data.resize_(gt_id.size()).copy_(gt_id)


            ques_emb = netW(ques_input, format='index')
            his_emb = netW(his_input, format='index')

            featD,rel_his = netE(ques_emb, his_emb, img_input, ques_input, his_input, rnd + 1)

            opt_ans_emb = netW(opt_ans_input, format='index')

            opt_feat = netD(opt_ans_emb, opt_ans_input, vocab_size)
            opt_feat = opt_feat.view(batch_size, -1, opt.ninp)

            # ans_emb = ans_emb.view(ans_length, -1, 100, opt.nhid)
            featD = featD.view(-1, opt.ninp, 1)
            score = torch.bmm(opt_feat, featD)
            score = score.view(-1, 100)

            for b in range(batch_size):
                gt_index.data[b] = gt_index.data[b] + b * 100

            gt_score = score.view(-1).index_select(0, gt_index)
            sort_score, sort_idx = torch.sort(score, 1, descending=True)

            count = sort_score.gt(gt_score.view(-1, 1).expand_as(sort_score))
            rank = count.sum(1) + 1


            rank_all_tmp += list(rank.view(-1).data.cpu().numpy())

        i += 1

    return rank_all_tmp


####################################################################################
# Main
####################################################################################
img_input = torch.FloatTensor(opt.batchSize).cuda()
ques_input = torch.LongTensor(ques_length, opt.batchSize).cuda()
his_input = torch.LongTensor(his_length, opt.batchSize).cuda()

# answer input
ans_input = torch.LongTensor(ans_length, opt.batchSize).cuda()
ans_target = torch.LongTensor(ans_length, opt.batchSize).cuda()
wrong_ans_input = torch.LongTensor(ans_length, opt.batchSize).cuda()
sample_ans_input = torch.LongTensor(1, opt.batchSize).cuda()
opt_ans_input = torch.LongTensor(ans_length, opt.batchSize).cuda()

batch_sample_idx = torch.LongTensor(opt.batchSize).cuda()
fake_diff_mask = torch.ByteTensor(opt.batchSize).cuda()
fake_len = torch.LongTensor(opt.batchSize).cuda()
# noise_input = torch.FloatTensor(opt.batchSize).cuda()
gt_index = torch.LongTensor(opt.batchSize).cuda()

ques_input = Variable(ques_input)
img_input = Variable(img_input)
his_input = Variable(his_input)

ans_input = Variable(ans_input)
ans_target = Variable(ans_target)
wrong_ans_input = Variable(wrong_ans_input)
sample_ans_input = Variable(sample_ans_input)

# noise_input = Variable(noise_input)
batch_sample_idx = Variable(batch_sample_idx)
fake_diff_mask = Variable(fake_diff_mask)
opt_ans_input = Variable(opt_ans_input)
gt_index = Variable(gt_index)

optimizer = optim.Adam([{'params': netW.parameters()},
                        {'params': netE.parameters()},
                        {'params': netD.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.999))

history = []



for epoch in range(opt.start_epoch, opt.niter):

    t = time.time()
    # train_loss, lr = train(epoch)
    # print('Epoch: %d learningRate %4f train loss %4f Time: %3f' % (epoch, lr, train_loss, time.time() - t))
    #
    # train_his = {'loss': train_loss}

    if epoch > 0:
        print('Evaluating ... ')
        rank_all = val()
        R1 = np.sum(np.array(rank_all) == 1) / float(len(rank_all))
        R5 = np.sum(np.array(rank_all) <= 5) / float(len(rank_all))
        R10 = np.sum(np.array(rank_all) <= 10) / float(len(rank_all))
        ave = np.sum(np.array(rank_all)) / float(len(rank_all))
        mrr = np.sum(1 / (np.array(rank_all, dtype='float'))) / float(len(rank_all))
        print('%d/%d: mrr: %f R1: %f R5 %f R10 %f Mean %f' % (epoch, len(dataloader_val), mrr, R1, R5, R10, ave))
        val_his = {'R1': R1, 'R5': R5, 'R10': R10, 'Mean': ave, 'mrr': mrr}

        opt.save_iter = 1
    else:

        val_his = {}
    # history.append({'epoch': epoch, 'train': train_his, 'val': val_his})
    # saving the model.
    if epoch % opt.save_iter == 0:
        torch.save({'epoch': epoch,
                    'opt': opt,
                    'netW': netW.state_dict(),
                    'netD': netD.state_dict(),
                    'netE': netE.state_dict()},
                   '%s/epoch_%d.pth' % (save_path, epoch))

        json.dump(history, open('%s/log.json' % (save_path), 'w'))
