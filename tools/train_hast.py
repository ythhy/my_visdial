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
# import progressbar
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

import misc.dataLoader_hast as dl
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
parser.add_argument('--negative_sample', type=int, default=40, help='folder to output images and model checkpoints')
parser.add_argument('--negative_sample_gcst', type=int, default=5, help='folder to output images and model checkpoints')
parser.add_argument('--sample_gcst', type=int, default=100, help='folder to output images and model checkpoints')
parser.add_argument('--neg_batch_sample', type=int, default=1, help='folder to output images and model checkpoints')
parser.add_argument('--start_epoch', type=int, default=1, help='start of epochs to train for')
parser.add_argument('--teacher_forcing', type=int, default=1, help='start of epochs to train for')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--save_iter', type=int, default=5, help='number of epochs to train for')

parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--lr', type=float, default=0.00004, help='learning rate for, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--verbose', action='store_true', help='show the sampled caption')

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
    checkpoint2 = torch.load(opt.model_path2)
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
    save_path = os.path.join(opt.outf, 'r_v_5' + cur_time)
    try:
        os.makedirs(save_path)
    except OSError:
        pass


else:
    # create new folder.
    t = datetime.datetime.now()
    cur_time = '%s-%s-%s' % (t.month, t.day, t.hour)
    save_path = os.path.join(opt.outf, 'tttt' + cur_time)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

####################################################################################
# Data Loader
####################################################################################

dataset = dl.train(input_img_h5=opt.input_img_h5, input_ques_h5=opt.input_ques_h5,
                   input_json=opt.input_json, negative_sample=opt.negative_sample, gcst_sample=opt.sample_gcst,
                   data_split='train')


dataset_val = dl.validate(input_img_h5=opt.input_img_val_h5, input_ques_h5=opt.input_ques_h5,
                          input_json=opt.input_json, negative_sample=opt.negative_sample,
                          num_val=opt.num_val, data_split='val')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=10,
                                             shuffle=False, num_workers=int(opt.workers))

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
critD = model.nPairLoss2(opt.ninp, opt.margin).cuda()
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


def subeval(image, fake_his, fake_ques, fake_answerIdx, fake_opt_ans, memory, rnd_gc):
    netW.eval()
    netE.eval()
    netD.eval()
    batch_size = fake_ques.size(0)
    fake_image = image.view(-1, img_feat_size)
    fake_img_input = Variable(fake_image, volatile=True).cuda()
    fake_ques_input = Variable(fake_ques, volatile=True).cuda().long()
    fake_his_input = Variable(fake_his, volatile=True).cuda().long()
    fake_his_input = fake_his_input.view(-1, his_length).cuda().long()
    fake_opt_ans_input = Variable(fake_opt_ans, volatile=True).cuda().long().contiguous().view(-1, ans_length)
    fake_gt_id = fake_answerIdx[:, rnd_gc]
    fake_gt_index = Variable(fake_gt_id, volatile=True).cuda().long()

    ques_emb = netW(fake_ques_input, format='index')
    his_emb = netW(fake_his_input, format='index')

    featD, _, _ = netE(ques_emb, his_emb, fake_img_input, fake_ques_input, fake_his_input, memory, rnd_gc + 1)

    opt_ans_emb = netW(fake_opt_ans_input, format='index')

    opt_feat = netD(opt_ans_emb, fake_opt_ans_input, vocab_size)
    opt_feat = opt_feat.view(batch_size, -1, opt.ninp)

    featD = featD.view(-1, opt.ninp, 1)
    score = torch.bmm(opt_feat, featD)
    score = score.view(-1, 100)
    for b in range(batch_size):
        fake_gt_index.data[b] = fake_gt_index.data[b] + b * 100

    gt_score = score.view(-1).index_select(0, fake_gt_index)
    sort_score, sort_idx = torch.sort(score, 1, descending=True)

    count = sort_score.gt(gt_score.view(-1, 1).expand_as(sort_score))
    rank = (count.sum(1) + 1).float()  ##rank(10,)
    reward = torch.reciprocal(rank).data

    return reward, None

####################################################################################
# training model
####################################################################################
def train(epoch):

    netW.train()
    netE.train()
    netD.train()

    lr = adjust_learning_rate(optimizer, epoch, 0.0004)

    data_iter = iter(dataloader)
    num = len(dataloader)

    average_loss = 0
    total_loss = 0
    count = 0
    i = 0
    while i < num:
        data = data_iter.next()



        def choice_prob(featD, real_feat, wrong_feat):
            feat = torch.cat((real_feat.view(-1,1,opt.ninp), wrong_feat),dim=1)
            featD = featD.view(real_feat.size(0), -1, 1)
            score = torch.bmm(feat, featD).squeeze(-1)
            prob = F.softmax(score, dim=1)
            return prob


        image, history, question, answer, answerLen, answerIdx, questionL, \
        opt_answerT, opt_answerLen, opt_answerIdx, gcst_ans = data

        batch_size = question.size(0)
        image = image.view(-1, img_feat_size)
        img_input.data.resize_(image.size()).copy_(image)

        his_m = Variable(torch.zeros(batch_size, 1, 1024)).cuda()
        img_m = Variable(torch.zeros(batch_size, 1, 1024)).cuda()
        memory = [his_m, img_m]

        for rnd in range(0, 9):
            netW.zero_grad()
            netE.zero_grad()
            netD.zero_grad()


            # get the corresponding round QA and history.
            ques = question[:, rnd, :]  # bs,16
            his = history[:, :rnd + 1, :].clone().view(-1, his_length)

            tans = answer[:, rnd, :]  # (bs,8)
            wrong_ans = opt_answerT[:, rnd, :].clone().view(-1, ans_length)


            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)

            ans_target.data.resize_(tans.size()).copy_(tans)
            wrong_ans_input.data.resize_(wrong_ans.size()).copy_(wrong_ans)

            ques_emb = netW(ques_input, format='index')
            his_emb = netW(his_input, format='index')

            featD, memory, _ = netE(ques_emb, his_emb, img_input,
                                           ques_input, his_input, memory, rnd + 1)


            ans_real_emb = netW(ans_target, format='index')
            ans_wrong_emb = netW(wrong_ans_input, format='index')

            real_feat = netD(ans_real_emb, ans_target, vocab_size)
            wrong_feat = netD(ans_wrong_emb, wrong_ans_input, vocab_size)
            wrong_feat = wrong_feat.view(batch_size, -1, opt.ninp)

            prob = choice_prob(featD, real_feat, wrong_feat)
            log_prob_gt = torch.log(prob[:, 0])  # log(gt)
            prob_w_nosort = prob[:, 1:].detach().data
            prob_w, indices = torch.sort(prob_w_nosort, 1, descending=True)
            prob_w = prob_w[:, :opt.negative_sample_gcst]
            indices = indices[:, :opt.negative_sample_gcst]

            for rnd_gc in range(rnd + 1, 10):
                reward_fake = torch.zeros_like(prob_w)

                gt_ques = question[:, rnd_gc, :].clone()
                gt_his = history[:, :rnd_gc + 1, :].clone()
                gt_opt_ans = gcst_ans[:, rnd_gc, :].clone()
                gt_answerIdx = answerIdx.clone()
                Reward_gt, _ = subeval(image, gt_his, gt_ques, gt_answerIdx, gt_opt_ans, memory, rnd_gc)

                fake_opt_ans = gcst_ans[:, rnd_gc, :].clone()
                ques_forfake = question[:, rnd, :].clone()
                fake_ques = question[:, rnd_gc, :].clone()
                fake_answerIdx = answerIdx.clone()
                for w_id in range(prob_w.size(1)):
                    fake_his = history[:, :rnd_gc+1, :].clone()
                    fake_his[:, rnd+1, :] = torch.zeros_like(fake_his[:, -1, :])
                    for b_i in range(fake_his.size(0)):
                        fake_ans = opt_answerT[:, rnd, indices[b_i, w_id], :].clone()
                        quesL = (ques_forfake[b_i, :] != 0).sum()
                        fake_his[b_i, rnd+1, :quesL] = ques_forfake[b_i, :quesL]
                        ansL = (fake_ans[b_i, :] != 0).sum()
                        if ansL+quesL<25:
                            fake_his[b_i, rnd+1, quesL:quesL+ansL] = fake_ans[b_i, :ansL]
                        else:
                            fake_his[b_i, rnd+1, quesL:] = fake_ans[b_i, :24-quesL]

                    reward_fake[:, w_id], _ = subeval(image, fake_his, fake_ques, fake_answerIdx, fake_opt_ans, memory, rnd_gc)


                    netW.train()
                    netE.train()
                    netD.train()

                    Reward_fake = (prob_w*reward_fake).sum(1)
                    Reward = (Reward_gt - Reward_fake)
                    Reward_copy = torch.zeros_like(Reward)
                    Reward_copy.copy_(Reward)
                    R = Variable(Reward_copy).cuda()
                    Loss_GCST = (log_prob_gt*R).sum() / (batch_size * (9 - rnd_gc))
                    nPairLoss = critD(featD, real_feat, wrong_feat)
                    Loss = nPairLoss + Loss_GCST
                    if Loss.data[0]>0:
                        average_loss += Loss.data[0]
                        Loss.backward()

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
    i = 0

    rank_all_tmp = []

    while i < len(dataloader_val):
        data = data_iter_val.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
        opt_answerT, answer_ids, answerLen, opt_answerLen, img_id = data

        batch_size = question.size(0)
        image = image.view(-1, img_feat_size)
        # image = l2_norm(image)
        img_input.data.resize_(image.size()).copy_(image)

        his_m = Variable(torch.zeros(batch_size, 1, 1024)).cuda()
        img_m = Variable(torch.zeros(batch_size, 1, 1024)).cuda()
        memory = [his_m, img_m]


        for rnd in range(10):
            # get the corresponding round QA and history.

            ques = question[:, rnd, :]
            his = history[:, :rnd + 1, :].clone().view(-1, his_length)

            opt_ans = opt_answerT[:, rnd, :].clone().view(-1, ans_length)
            gt_id = answer_ids[:, rnd]

            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)

            opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)
            gt_index.data.resize_(gt_id.size()).copy_(gt_id)
            # opt_len = opt_answerLen[:, rnd, :].clone().view(-1)

            ques_emb = netW(ques_input, format='index')
            his_emb = netW(his_input, format='index')


            featD, memory, weightq = netE(ques_emb, his_emb, img_input, ques_input, his_input,
                                       memory, rnd + 1)

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
opt_ans_input = torch.LongTensor(ans_length, opt.batchSize).cuda()
gt_index = torch.LongTensor(opt.batchSize).cuda()

ques_input = Variable(ques_input)
img_input = Variable(img_input)
his_input = Variable(his_input)

ans_input = Variable(ans_input)
ans_target = Variable(ans_target)
wrong_ans_input = Variable(wrong_ans_input)



opt_ans_input = Variable(opt_ans_input)
gt_index = Variable(gt_index)

optimizer = optim.Adam([{'params': netW.parameters()},
                        {'params': netE.parameters()},
                        {'params': netD.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.999))

history = []



for epoch in range(opt.start_epoch, opt.niter):

    t = time.time()
    train_loss, lr = train(epoch)
    print('Epoch: %d learningRate %4f train loss %4f Time: %3f' % (epoch, lr, train_loss, time.time() - t))

    train_his = {'loss': train_loss}

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
    history.append({'epoch': epoch, 'train': train_his, 'val': val_his})
    # saving the model.
    if epoch % opt.save_iter == 0:
        torch.save({'epoch': epoch,
                    'opt': opt,
                    'netW': netW.state_dict(),
                    'netD': netD.state_dict(),
                    'netE': netE.state_dict()},
                   '%s/epoch_%d.pth' % (save_path, epoch))

        json.dump(history, open('%s/log.json' % (save_path), 'w'))
