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

from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt
import misc.dataLoader as dl
import misc.model as model
from misc.encoder_r import _netE
import datetime
import h5py
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='', help='folder to output images and model checkpoints')
parser.add_argument('--input_img_h5', default='data/img_test_resnet.h5', help='')
parser.add_argument('--input_ques_h5', default='script/new2_visdial_data_test_1.0.h5', help='visdial_data.h5')
parser.add_argument('--input_json', default='script/visdial_params_test_1.0.json', help='visdial_params.json')
parser.add_argument('--output_dir', default='./result', help='result.json')

parser.add_argument('--model_path', default='/home/yth/visDial_iccv/save_1.0/D/r_np33-7-19/epoch_19.pth', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--batchsize', default=1, help='batchsize')
parser.add_argument('--split', default='test', help='split')

opt = parser.parse_args()

opt.manualSeed = random.randint(1, 10000) # fix seed
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

####################################################################################
# Data Loader
####################################################################################
input_img_h5 = opt.input_img_h5
input_ques_h5 = opt.input_ques_h5
input_json = opt.input_json


print("=> loading checkpoint '{}'".format(opt.model_path))
checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
model_path = opt.model_path
data_dir = opt.data_dir
output_dir = opt.output_dir
split = opt.split
opt = checkpoint['opt']
opt.start_epoch = checkpoint['epoch']
opt.batchSize = 1
opt.data_dir = data_dir
opt.model_path = model_path
opt.output_dir = output_dir
opt.split = split
opt.cuda = True

if opt.split =='test':
    dataset_test = dl.test(input_img_h5=input_img_h5, input_ques_h5=input_ques_h5,
                    input_json=input_json,  data_split = 'test')


    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers))
else:
    dataset_test = dl.validate(input_img_h5=input_img_h5, input_ques_h5=input_ques_h5,
                              input_json=input_json, negative_sample=opt.negative_sample,
                              num_val=opt.num_val, data_split='val')

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                                 shuffle=False, num_workers=int(opt.workers))
####################################################################################
# Build the Model
####################################################################################

nn_neg = opt.negative_sample
vocab_size = dataset_test.vocab_size
ques_length = dataset_test.ques_length
ans_length = dataset_test.ans_length
his_length = dataset_test.ans_length + dataset_test.ques_length
itow = dataset_test.itow
opt.word_size = opt.ninp
img_feat_size = 2048

netE = _netE(opt.model, opt.ninp, opt.nhid, opt.nlayers, opt.dropout, img_feat_size).cuda()
netW = model._netW(vocab_size, opt.ninp, opt.dropout).cuda()
netD = model._netD(opt.model, opt.ninp, opt.nhid, opt.nlayers, vocab_size, opt.dropout).cuda()


if opt.model_path != '': # load the pre-trained model.
    netW.load_state_dict(checkpoint['netW'])
    netE.load_state_dict(checkpoint['netE'])
    netD.load_state_dict(checkpoint['netD'])
    print('Loading model Success!')

if opt.cuda: # ship to cuda, if has GPU
    netW.cuda(), netE.cuda(), netD.cuda()


n_neg = 100
####################################################################################
# Some Functions
####################################################################################
score1 = np.zeros((8000, 100))
def eval():
    netW.eval()
    netE.eval()
    netD.eval()

    data_iter_test = iter(dataloader_test)
    i = 0
    result = []

    while i < len(dataloader_test):#len(1000):
        result_tmp = {}
        data = data_iter_test.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                opt_answerT, _, answerLen, opt_answerLen, img_id, rnd_idx= data

        batch_size = question.size(0)
        image = image.view(-1, 2048)
        img_input.data.resize_(image.size()).copy_(image)
        save_tmp = [[] for j in range(batch_size)]

        his_m = Variable(torch.zeros(batch_size, 1, 1024)).cuda()
        img_m = Variable(torch.zeros(batch_size, 1, 1024)).cuda()
        memory = [his_m, img_m]



        rnd_eval=int(rnd_idx-1)

        for rnd in range(rnd_eval+1):

            # get the corresponding round QA and history.
            ques = question[:, rnd, :]
            his = history[:, :rnd + 1, :].clone().view(-1, his_length)
            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)
            ques_emb = netW(ques_input, format='index')
            his_emb = netW(his_input, format='index')

            featD, memory, weight_q = netE(ques_emb, his_emb, img_input, \
                                           ques_input, his_input, memory, rnd + 1)

        opt_ans = opt_answerT[:, rnd, :].clone().view(-1, ans_length)

        # gt_index.data.resize_(gt_id.size()).copy_(gt_id)
        opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)

        opt_ans_emb = netW(opt_ans_input, format='index')
        opt_feat = netD(opt_ans_emb, opt_ans_input, vocab_size)
        opt_feat = opt_feat.view(batch_size, -1, opt.ninp)

        featD = featD.view(-1, opt.ninp, 1)
        score = torch.bmm(opt_feat, featD)
        score = score.view(-1, 100)

        score1[i,:] = score.data.cpu().numpy()

        sort_score, sort_idx = torch.sort(score, 1, descending=True)
        sort_idx = sort_idx + 1
        sort_idx = sort_idx.squeeze().cpu().data.type(torch.IntTensor)
        sort_result = list(sort_idx)
        sort_final = [1] * 100
        for rank, w in enumerate(sort_result):
            sort_final[w - 1] = rank + 1

        result_tmp['image_id'] = int(img_id)
        result_tmp['round_id'] = int(rnd + 1)
        result_tmp['ranks'] = sort_final
        # sort_result = [rank +'\n' for rank in sort_result]
        result.append(result_tmp)

        if i%100 == 0:
            print(i)
        i = i+1
    return result

def eval_val():
    netW.eval()
    netE.eval()
    netD.eval()

    data_iter_test = iter(dataloader_test)
    ques_hidden = netE.init_bi_hidden(opt.batchSize)
    hist_hidden = netE.init_bi_hidden(opt.batchSize)

    opt_hidden = netD.init_bi_hidden(opt.batchSize)
    i = 0
    display_count = 0
    average_loss = 0

    img_atten = torch.FloatTensor(100 * 30, 10, 7, 7)
    result = []
    while i < len(dataloader_test):#len(1000):
        data = data_iter_test.next()
        image, history, question, answer, answerT, questionL, opt_answer, \
                opt_answerT, answer_ids, answerLen, opt_answerLen, img_id  = data

        batch_size = question.size(0)
        image = image.view(-1, 2048)
        img_input.data.resize_(image.size()).copy_(image)
        # save_tmp = [[] for j in range(batch_size)]

        for rnd in range(10):
            result_tmp = {'image_id':1,
                          'round_id':1,
                          'ranks':[]
                          }
            # get the corresponding round QA and history.
            ques = question[:,rnd,:].t()
            his = history[:,:rnd+1,:].clone().view(-1, his_length).t()

            opt_ans = opt_answerT[:,rnd,:].clone().view(-1, ans_length).t()
            # gt_id = answer_ids[:,rnd]

            ques_input.data.resize_(ques.size()).copy_(ques)
            his_input.data.resize_(his.size()).copy_(his)

            # gt_index.data.resize_(gt_id.size()).copy_(gt_id)
            opt_ans_input.data.resize_(opt_ans.size()).copy_(opt_ans)

            # opt_len = opt_answerLen[:,rnd,:].clone().view(-1)

            ques_emb = netW(ques_input, format = 'index')
            his_emb = netW(his_input, format = 'index')

            ques_hidden = repackage_hidden(ques_hidden, batch_size)
            hist_hidden = repackage_hidden(hist_hidden, his_input.size(1))

            featD, ques_hidden = netE(ques_emb, his_emb, img_input, \
                                                                    ques_hidden, hist_hidden, rnd+1)

            #img_atten[i*batch_size:(i+1)*batch_size, rnd, :] = img_atten_weight.data.view(batch_size, 7, 7)

            opt_ans_emb = netW(opt_ans_input, format = 'index')
            opt_hidden = repackage_hidden(opt_hidden, opt_ans_input.size(1))
            opt_feat = netD(opt_ans_emb, opt_ans_input, opt_hidden, vocab_size)
            opt_feat = opt_feat.view(batch_size, -1, opt.ninp)

            featD = featD.view(-1, opt.ninp, 1)
            score = torch.bmm(opt_feat, featD)
            score = score.view(-1, 100)

            sort_score, sort_idx = torch.sort(score, 1, descending=True)
            sort_idx = sort_idx+1
            sort_idx = sort_idx.squeeze().cpu().data.type(torch.IntTensor)
            sort_result = list(sort_idx)
            sort_final = [1]*100
            for rank, w in enumerate(sort_result):
                sort_final[w-1] = rank+1

            result_tmp['image_id'] = int(img_id)
            result_tmp['round_id'] = int(rnd + 1)
            result_tmp['ranks'] = sort_final
            # sort_result = [rank +'\n' for rank in sort_result]
            result.append(result_tmp)



        i += 1
        if i%100 ==0:
            print(i)




    return result

####################################################################################
# Main
####################################################################################
img_input = torch.FloatTensor(opt.batchSize)
ques_input = torch.LongTensor(ques_length, opt.batchSize)
his_input = torch.LongTensor(his_length, opt.batchSize)

# answer input
opt_ans_input = torch.LongTensor(ans_length, opt.batchSize)
# fake_ans_input = torch.FloatTensor(ques_length, opt.batchSize, n_words)
sample_ans_input = torch.LongTensor(1, opt.batchSize)

# answer index location.
opt_index = torch.LongTensor(opt.batchSize)
fake_index = torch.LongTensor(opt.batchSize)

batch_sample_idx = torch.LongTensor(opt.batchSize)
# answer len
fake_len = torch.LongTensor(opt.batchSize)

# noise
noise_input = torch.FloatTensor(opt.batchSize)
gt_index = torch.LongTensor(opt.batchSize)


if opt.cuda:
    ques_input, his_input, img_input = ques_input.cuda(), his_input.cuda(), img_input.cuda()
    opt_ans_input = opt_ans_input.cuda()

    opt_index, fake_index =  opt_index.cuda(), fake_index.cuda()

    fake_len = fake_len.cuda()
    noise_input = noise_input.cuda()
    batch_sample_idx = batch_sample_idx.cuda()
    gt_index = gt_index.cuda()


ques_input = Variable(ques_input)
img_input = Variable(img_input)
his_input = Variable(his_input)

opt_ans_input = Variable(opt_ans_input)

sample_ans_input = Variable(sample_ans_input)

opt_index = Variable(opt_index)
fake_index = Variable(fake_index)

fake_len = Variable(fake_len)
noise_input = Variable(noise_input)
batch_sample_idx = Variable(batch_sample_idx)
gt_index = Variable(gt_index)

if opt.split =='test':
    result = eval()
else:
    result = eval_val()

f = h5py.File('score_r_np33-7-19.h5', 'w')
f.create_dataset('score', dtype='float32', data=score1)
f.close()
# t = datetime.datetime.now()
# cur_time = '%s-%s-%s-%s' %(t.month, t.day, t.hour, t.minute)
# file_name = opt.split+'r_v5_np2_ft.json'
# file_dir = os.path.join(opt.output_dir,file_name)
# # result = [result_tmp +'\n' for result_tmp in result]
# with open(file_dir, 'w') as f:
#     result_w = json.dumps(result,indent=2)
#     f.write(result_w)
# f.close()

