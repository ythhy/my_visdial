import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import h5py
import json
import pdb
import random
from misc.utils import repackage_hidden, clip_gradient, adjust_learning_rate, decode_txt

class train(data.Dataset): # torch wrapper
    def __init__(self, input_img_h5, input_ques_h5, input_json, negative_sample, data_split):
        split = data_split
        print('DataLoader loading: %s' %split)
        print('Loading image feature from %s' %input_img_h5)


        f = json.load(open(input_json, 'r'))
        self.itow = f['itow']
        self.img_info = f['img_'+split]

        # get the data split.
        total_num = len(self.img_info)


        s = 0
        e = int(total_num)

        self.img_info = self.img_info[s:e]

        print('%s number of data: %d' %(split, e-s))
        # load the data.
        self.imgs = h5py.File(input_img_h5, 'r')['resnet_feat']
        # self.imgs = f['images_'+split][s:e]
        # self.imgs = f['resnet_feat'][:]
        # f.close()

        print('Loading txt from %s' %input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = f['ques_'+split][s:e]
        self.ans = f['ans_'+split][s:e]
        self.cap = f['cap_'+split][s:e]

        self.ques_len = f['ques_len_'+split][s:e]
        self.ans_len = f['ans_len_'+split][s:e]
        self.cap_len = f['cap_len_'+split][s:e]

        self.ans_ids = f['ans_index_'+split][s:e]
        self.opt_ids = f['opt_'+split][s:e]
        self.opt_list = f['opt_list_'+split][:]
        self.opt_len = f['opt_len_'+split][:]
        f.close()

        self.ques_length = self.ques.shape[2]
        self.ans_length = self.ans.shape[2]
        self.his_length = self.ques_length + self.ans_length
        self.vocab_size = len(self.itow)+1

        print('Vocab Size: %d' % self.vocab_size)

        self.split = split
        self.rnd = 10
        self.negative_sample = negative_sample
        
    def __getitem__(self, index):
        # get the image

        img = torch.from_numpy(self.imgs[index])

        # get the history
        his = np.zeros((self.rnd, self.his_length))
        # his[0,self.his_length-self.cap_len[index]:] = self.cap[index,:self.cap_len[index]]
        his[0, 0:self.cap_len[index]] = self.cap[index, :self.cap_len[index]]

        ques = np.zeros((self.rnd, self.ques_length))
        ans = np.zeros((self.rnd, self.ans_length))

        opt_ans = np.zeros((self.rnd, self.negative_sample, self.ans_length))
        ans_len = np.zeros((self.rnd))
        ques_len = np.zeros((self.rnd))
        opt_ans_len = np.zeros((self.rnd, self.negative_sample))

        ans_idx = np.zeros((self.rnd))
        ans_idx2 = np.zeros((self.rnd),dtype=np.int64)
        opt_ans_idx = np.zeros((self.rnd, self.negative_sample))

        for i in range(self.rnd):
            # get the index
            q_len = self.ques_len[index, i]
            a_len = self.ans_len[index, i]
            qa_len = q_len + a_len

            if i+1 < self.rnd:
                his[i+1, 0:q_len] = self.ques[index, i, :q_len]
                his[i+1, q_len:q_len+a_len] = self.ans[index, i, :a_len]

            ques[i, 0:q_len] = self.ques[index, i, :q_len]




            ans[i, :a_len] = self.ans[index, i, :a_len]


            # ans_target[i, :a_len] = self.ans[index, i, :a_len]
            # ans_target[i, a_len] = self.vocab_size
            ans_len[i] = self.ans_len[index, i]
            ques_len[i] = self.ques_len[index, i]

            opt_ids = self.opt_ids[index, i] # since python start from 0
            # random select the negative samples.
            ans_idx[i] = opt_ids[self.ans_ids[index, i]]
            ans_idx2[i] = self.ans_ids[index, i]

            # for j, ids in enumerate(opt_ids):
            #     opt_len = self.opt_len[ids]
            #     opt_ans[i, j, :opt_len] = self.opt_list[ids,:opt_len]
            #     opt_ans_len[i, j] = opt_len




            # exclude the gt index.
            opt_ids = np.delete(opt_ids, ans_idx2[i], 0)
            random.shuffle(opt_ids)
            for j in range(self.negative_sample):
                ids = opt_ids[j]
                opt_ans_idx[i, j] = ids

                opt_len = self.opt_len[ids]

                opt_ans_len[i, j] = opt_len
                opt_ans[i, j, :opt_len] = self.opt_list[ids,:opt_len]

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)
        ans = torch.from_numpy(ans)
        # ans_target = torch.from_numpy(ans_target)
        ans_len = torch.from_numpy(ans_len)
        ques_len = torch.from_numpy(ques_len)
        opt_ans_len = torch.from_numpy(opt_ans_len)
        opt_ans = torch.from_numpy(opt_ans)
        ans_idx = torch.from_numpy(ans_idx)
        ans_idx2 = torch.from_numpy(ans_idx2)
        opt_ans_idx = torch.from_numpy(opt_ans_idx)
        return img, his, ques, ans, ans_len, ans_idx2, ques_len, \
                opt_ans, opt_ans_len, opt_ans_idx

    def __len__(self):
        return self.ques.shape[0]

class validate(data.Dataset): # torch wrapper
    def __init__(self, input_img_h5, input_ques_h5, input_json, negative_sample, num_val, data_split):

        print('DataLoader loading: %s' %data_split)
        print('Loading image feature from %s' %input_img_h5)

        if data_split == 'test':
            split = 'test'
        else:
            split = 'val' # train and val split both corresponding to 'train'

        f = json.load(open(input_json, 'r'))
        self.itow = f['itow']
        self.img_info = f['img_'+split]

        # get the data split.
        total_num = len(self.img_info)
        s = 0
        e = total_num

        self.img_info = self.img_info[s:e]
        print('%s number of data: %d' %(data_split, e-s))

        # load the data.
        f = h5py.File(input_img_h5, 'r')
        self.imgs = f['resnet_feat'][s:e]
        f.close()

        print('Loading txt from %s' %input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = f['ques_'+split][s:e]
        self.ans = f['ans_'+split][s:e]
        self.cap = f['cap_'+split][s:e]

        self.ques_len = f['ques_len_'+split][s:e]
        self.ans_len = f['ans_len_'+split][s:e]
        self.cap_len = f['cap_len_'+split][s:e]

        self.ans_ids = f['ans_index_'+split][s:e]
        self.opt_ids = f['opt_'+split][s:e]
        self.opt_list = f['opt_list_'+split][:]
        self.opt_len = f['opt_len_'+split][:]
        f.close()

        self.ques_length = self.ques.shape[2]
        self.ans_length = self.ans.shape[2]
        self.his_length = self.ques_length + self.ans_length
        self.vocab_size = len(self.itow)+1

        print('Vocab Size: %d' % self.vocab_size)
        self.split = split
        self.rnd = 10
        self.negative_sample = negative_sample

    def __getitem__(self, index):

        # get the image
        img_id = self.img_info[index]['imgId']
        img = torch.from_numpy(self.imgs[index])
        # get the history
        his = np.zeros((self.rnd, self.his_length))
        his[0,:self.cap_len[index]] = self.cap[index,:self.cap_len[index]]

        ques = np.zeros((self.rnd, self.ques_length))
        ans = np.zeros((self.rnd, self.ans_length))
        ans_target = np.zeros((self.rnd, self.ans_length))
        quesL = np.zeros((self.rnd, self.ques_length))

        opt_ans = np.zeros((self.rnd, 100, self.ans_length))
        ans_ids = np.zeros(self.rnd)
        opt_ans_target = np.zeros((self.rnd, 100, self.ans_length))

        ans_len = np.zeros((self.rnd))
        opt_ans_len = np.zeros((self.rnd, 100))


        for i in range(self.rnd):
            # get the index
            q_len = self.ques_len[index, i]
            a_len = self.ans_len[index, i]
            qa_len = q_len + a_len

            if i+1 < self.rnd:
                ques_ans = np.concatenate([self.ques[index, i, :q_len], self.ans[index, i, :a_len]])
                his[i+1, :qa_len] = ques_ans

            ques[i, :q_len] = self.ques[index, i, :q_len]
            quesL[i, :q_len] = self.ques[index, i, :q_len]
            ans[i, :a_len] = self.ans[index, i, :a_len]


            ans_target[i, :a_len] = self.ans[index, i, :a_len]


            ans_ids[i] = self.ans_ids[index, i] # since python start from 0
            opt_ids = self.opt_ids[index, i] # since python start from 0
            ans_len[i] = self.ans_len[index, i]
            ans_idx = self.ans_ids[index, i]

            for j, ids in enumerate(opt_ids):
                opt_len = self.opt_len[ids]
                opt_ans[i, j, :opt_len] = self.opt_list[ids,:opt_len]


                opt_ans_target[i, j,:opt_len] = self.opt_list[ids,:opt_len]
                opt_ans_len[i, j] = opt_len

        opt_ans = torch.from_numpy(opt_ans)
        opt_ans_target = torch.from_numpy(opt_ans_target)
        ans_ids = torch.from_numpy(ans_ids)

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)
        ans = torch.from_numpy(ans)
        ans_target = torch.from_numpy(ans_target)
        quesL = torch.from_numpy(quesL)

        ans_len = torch.from_numpy(ans_len)
        opt_ans_len = torch.from_numpy(opt_ans_len)

        return img, his, ques, ans, ans_target, quesL, opt_ans, \
                    opt_ans_target, ans_ids, ans_len, opt_ans_len, img_id


    def __len__(self):
        return self.ques.shape[0]

class test(data.Dataset):  # torch wrapper
    def __init__(self, input_img_h5, input_ques_h5, input_json,  data_split):

        print('DataLoader loading: %s' % data_split)
        print('Loading image feature from %s' % input_img_h5)

        # if data_split == 'test':
        #     split = 'test'
        # else:
        #     split = 'val'  # train and val split both corresponding to 'train'
        split = data_split

        f = json.load(open(input_json, 'r'))
        self.itow = f['itow']
        self.img_info = f['img_' + split]

        # get the data split.
        total_num = len(self.img_info)
        # if data_split == 'train':
        #     s = 0
        #     e = total_num - num_val
        # elif data_split == 'val':
        #     s = total_num - num_val
        #     e = total_num
        # else:
        #     s = 0
        #     e = total_num
        s = 0
        e = total_num

        self.img_info = self.img_info[s:e]
        print('%s number of data: %d' % (data_split, e - s))

        # load the data.
        f = h5py.File(input_img_h5, 'r')
        # self.imgs = f['images_'+split][s:e]
        self.imgs = f['resnet_feat'][s:e]
        f.close()

        print('Loading txt from %s' % input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = f['ques_' + split][s:e]
        self.ans = f['ans_' + split][s:e]
        self.cap = f['cap_' + split][s:e]

        self.ques_len = f['ques_len_' + split][s:e]
        self.ans_len = f['ans_len_' + split][s:e]
        self.cap_len = f['cap_len_' + split][s:e]

        # self.ans_ids = f['ans_index_' + split][s:e]
        self.opt_ids = f['opt_' + split][s:e]
        self.opt_list = f['opt_list_' + split][:]
        self.opt_len = f['opt_len_' + split][:]

        self.round_len = f['round_len_'+split][:]

        f.close()

        self.ques_length = self.ques.shape[2]
        self.ans_length = self.ans.shape[2]
        self.his_length = self.ques_length + self.ans_length
        self.vocab_size = len(self.itow) + 1

        print('Vocab Size: %d' % self.vocab_size)
        self.split = split
        self.rnd = 10
        # self.negative_sample = negative_sample

    def __getitem__(self, index):

        rnd = int(self.round_len[index])
        # get the image
        img_id = self.img_info[index]['imgId']
        img = torch.from_numpy(self.imgs[index])
        # get the history
        his = np.zeros((self.rnd, self.his_length))
        his[0, :self.cap_len[index]] = self.cap[index, :self.cap_len[index]]

        ques = np.zeros((self.rnd, self.ques_length))
        ans = np.zeros((self.rnd, self.ans_length))
        ans_target = np.zeros((self.rnd, self.ans_length))
        quesL = np.zeros((self.rnd, self.ques_length))

        opt_ans = np.zeros((self.rnd, 100, self.ans_length))
        # ans_ids = np.zeros(self.rnd)
        opt_ans_target = np.zeros((self.rnd, 100, self.ans_length))

        ans_len = np.zeros((self.rnd))
        opt_ans_len = np.zeros((self.rnd, 100))

        for i in range(self.rnd):
            # get the index
            q_len = self.ques_len[index, i]
            a_len = self.ans_len[index, i]
            qa_len = q_len + a_len

            if i + 1 < self.rnd:
                ques_ans = np.concatenate([self.ques[index, i, :q_len], self.ans[index, i, :a_len]])
                his[i + 1, :qa_len] = ques_ans

            ques[i, :q_len] = self.ques[index, i, :q_len]
            quesL[i, :q_len] = self.ques[index, i, :q_len]
            ans[i, :a_len] = self.ans[index, i, :a_len]

            ans_target[i, :a_len] = self.ans[index, i, :a_len]


            # ans_ids[i] = self.ans_ids[index, i]  # since python start from 0
            opt_ids = self.opt_ids[index, i]  # since python start from 0
            ans_len[i] = self.ans_len[index, i]
            # ans_idx = self.ans_ids[index, i]

            for j, ids in enumerate(opt_ids):
                opt_len = self.opt_len[ids]
                opt_ans[i, j, :opt_len] = self.opt_list[ids,:opt_len]


                opt_ans_target[i, j,:opt_len] = self.opt_list[ids,:opt_len]
                opt_ans_len[i, j] = opt_len

        opt_ans = torch.from_numpy(opt_ans)
        opt_ans_target = torch.from_numpy(opt_ans_target)
        # ans_ids = torch.from_numpy(ans_ids)

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)
        ans = torch.from_numpy(ans)
        ans_target = torch.from_numpy(ans_target)
        quesL = torch.from_numpy(quesL)

        ans_len = torch.from_numpy(ans_len)
        opt_ans_len = torch.from_numpy(opt_ans_len)

        return img, his, ques, ans, ans_target, quesL, opt_ans, \
               opt_ans_target, ans_len, ans_len, opt_ans_len, img_id, rnd

    def __len__(self):
        return self.ques.shape[0]


class trainall(data.Dataset):  # torch wrapper
    def __init__(self, input_img_h5, input_ques_h5, input_json, negative_sample):
        split_train = 'train'
        split_val = 'val'
        split = split_train+split_val
        print('DataLoader loading: %s' % split)
        print('Loading image feature from %s' % input_img_h5)

        f = json.load(open(input_json, 'r'))
        self.itow = f['itow']
        # self.img_info = f['img_' + split]
        #
        # # get the data split.
        # total_num = len(self.img_info)
        #
        # s = 0
        # e = int(total_num)
        #
        # self.img_info = self.img_info[s:e]

        # load the data.
        self.imgs = h5py.File(input_img_h5, 'r')['resnet_feat']
        # f.close()

        print('Loading txt from %s' % input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = np.concatenate((f['ques_' + split_train][:], f['ques_' + split_val]), axis=0)
        self.ans = np.concatenate((f['ans_' + split_train][:], f['ans_' + split_val]), axis=0)
        self.cap = np.concatenate((f['cap_' + split_train][:], f['cap_' + split_val]), axis=0)

        self.ques_len = np.concatenate((f['ques_len_' + split_train][:], f['ques_len_' + split_val]), axis=0)
        self.ans_len = np.concatenate((f['ans_len_' + split_train][:], f['ans_len_' + split_val]), axis=0)
        self.cap_len = np.concatenate((f['cap_len_' + split_train][:], f['cap_len_' + split_val]), axis=0)

        self.ans_ids = np.concatenate((f['ans_index_' + split_train][:], f['ans_index_' + split_val]), axis=0)
        self.opt_ids = np.concatenate((f['opt_' + split_train][:], f['opt_' + split_val]), axis=0)
        self.opt_list = np.concatenate((f['opt_list_' + split_train][:], f['opt_list_' + split_val]), axis=0)
        self.opt_len = np.concatenate((f['opt_len_' + split_train][:], f['opt_len_' + split_val]), axis=0)
        f.close()

        self.ques_length = self.ques.shape[2]
        self.ans_length = self.ans.shape[2]
        self.his_length = self.ques_length + self.ans_length
        self.vocab_size = len(self.itow) + 1

        print('Vocab Size: %d' % self.vocab_size)
        self.rnd = 10
        self.negative_sample = negative_sample

    def __getitem__(self, index):
        # get the image
        img = torch.from_numpy(self.imgs[index])

        # get the history
        his = np.zeros((self.rnd, self.his_length))
        # his[0,self.his_length-self.cap_len[index]:] = self.cap[index,:self.cap_len[index]]
        his[0, 0:self.cap_len[index]] = self.cap[index, :self.cap_len[index]]

        ques = np.zeros((self.rnd, self.ques_length))
        ans = np.zeros((self.rnd, self.ans_length))

        opt_ans = np.zeros((self.rnd, self.negative_sample, self.ans_length))
        ans_len = np.zeros((self.rnd))
        ques_len = np.zeros((self.rnd))
        opt_ans_len = np.zeros((self.rnd, self.negative_sample))

        ans_idx = np.zeros((self.rnd))
        ans_idx2 = np.zeros((self.rnd), dtype=np.int64)
        opt_ans_idx = np.zeros((self.rnd, self.negative_sample))

        for i in range(self.rnd):
            # get the index
            q_len = self.ques_len[index, i]
            a_len = self.ans_len[index, i]
            qa_len = q_len + a_len

            if i + 1 < self.rnd:
                his[i + 1, 0:q_len] = self.ques[index, i, :q_len]
                his[i + 1, q_len:q_len + a_len] = self.ans[index, i, :a_len]

            ques[i, 0:q_len] = self.ques[index, i, :q_len]

            ans[i, :a_len] = self.ans[index, i, :a_len]

            # ans_target[i, :a_len] = self.ans[index, i, :a_len]
            # ans_target[i, a_len] = self.vocab_size
            ans_len[i] = self.ans_len[index, i]
            ques_len[i] = self.ques_len[index, i]

            opt_ids = self.opt_ids[index, i]  # since python start from 0
            # random select the negative samples.
            ans_idx[i] = opt_ids[self.ans_ids[index, i]]
            ans_idx2[i] = self.ans_ids[index, i]
            # exclude the gt index.
            opt_ids = np.delete(opt_ids, ans_idx2[i], 0)
            random.shuffle(opt_ids)
            for j in range(self.negative_sample):
                ids = opt_ids[j]
                opt_ans_idx[i, j] = ids

                opt_len = self.opt_len[ids]

                opt_ans_len[i, j] = opt_len
                opt_ans[i, j, :opt_len] = self.opt_list[ids, :opt_len]
                # opt_ans[i, j, opt_len] = self.vocab_size

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)
        ans = torch.from_numpy(ans)
        # ans_target = torch.from_numpy(ans_target)
        ans_len = torch.from_numpy(ans_len)
        ques_len = torch.from_numpy(ques_len)
        opt_ans_len = torch.from_numpy(opt_ans_len)
        opt_ans = torch.from_numpy(opt_ans)
        ans_idx = torch.from_numpy(ans_idx)
        ans_idx2 = torch.from_numpy(ans_idx2)
        opt_ans_idx = torch.from_numpy(opt_ans_idx)
        return img, his, ques, ans, ans_len, ans_idx2, ques_len, \
               opt_ans, opt_ans_len, opt_ans_idx

    def __len__(self):
        return self.ques.shape[0]

class trainallXE(data.Dataset):  # torch wrapper
    def __init__(self, input_img_h5, input_ques_h5, input_json, negative_sample):
        split_train = 'train'
        split_val = 'val'
        split = split_train + split_val
        print('DataLoader loading: %s' % split)
        print('Loading image feature from %s' % input_img_h5)

        f = json.load(open(input_json, 'r'))
        self.itow = f['itow']
        # self.img_info = f['img_' + split]
        #
        # # get the data split.
        # total_num = len(self.img_info)
        #
        # s = 0
        # e = int(total_num)
        #
        # self.img_info = self.img_info[s:e]

        # load the data.
        self.imgs = h5py.File(input_img_h5, 'r')['resnet_feat']
        # f.close()

        print('Loading txt from %s' % input_ques_h5)
        f = h5py.File(input_ques_h5, 'r')
        self.ques = np.concatenate((f['ques_' + split_train][:], f['ques_' + split_val]), axis=0)
        self.ans = np.concatenate((f['ans_' + split_train][:], f['ans_' + split_val]), axis=0)
        self.cap = np.concatenate((f['cap_' + split_train][:], f['cap_' + split_val]), axis=0)

        self.ques_len = np.concatenate((f['ques_len_' + split_train][:], f['ques_len_' + split_val]), axis=0)
        self.ans_len = np.concatenate((f['ans_len_' + split_train][:], f['ans_len_' + split_val]), axis=0)
        self.cap_len = np.concatenate((f['cap_len_' + split_train][:], f['cap_len_' + split_val]), axis=0)

        self.ans_ids = np.concatenate((f['ans_index_' + split_train][:], f['ans_index_' + split_val]), axis=0)
        self.opt_ids = np.concatenate((f['opt_' + split_train][:], f['opt_' + split_val]), axis=0)
        self.opt_list = np.concatenate((f['opt_list_' + split_train][:], f['opt_list_' + split_val]), axis=0)
        self.opt_len = np.concatenate((f['opt_len_' + split_train][:], f['opt_len_' + split_val]), axis=0)
        f.close()

        self.ques_length = self.ques.shape[2]
        self.ans_length = self.ans.shape[2]
        self.his_length = self.ques_length + self.ans_length
        self.vocab_size = len(self.itow) + 1

        print('Vocab Size: %d' % self.vocab_size)
        self.rnd = 10
        self.negative_sample = 100

    def __getitem__(self, index):
        # get the image
        img = torch.from_numpy(self.imgs[index])

        # get the history
        his = np.zeros((self.rnd, self.his_length))
        # his[0,self.his_length-self.cap_len[index]:] = self.cap[index,:self.cap_len[index]]
        his[0, 0:self.cap_len[index]] = self.cap[index, :self.cap_len[index]]

        ques = np.zeros((self.rnd, self.ques_length))
        ans = np.zeros((self.rnd, self.ans_length))

        opt_ans = np.zeros((self.rnd, self.negative_sample, self.ans_length))
        ans_len = np.zeros((self.rnd))
        ques_len = np.zeros((self.rnd))
        opt_ans_len = np.zeros((self.rnd, self.negative_sample))

        ans_idx = np.zeros((self.rnd))
        ans_idx2 = np.zeros((self.rnd), dtype=np.int64)
        opt_ans_idx = np.zeros((self.rnd, self.negative_sample))

        for i in range(self.rnd):
            # get the index
            q_len = self.ques_len[index, i]
            a_len = self.ans_len[index, i]
            qa_len = q_len + a_len

            if i + 1 < self.rnd:
                his[i + 1, 0:q_len] = self.ques[index, i, :q_len]
                his[i + 1, q_len:q_len + a_len] = self.ans[index, i, :a_len]

            ques[i, 0:q_len] = self.ques[index, i, :q_len]

            ans[i, :a_len] = self.ans[index, i, :a_len]

            # ans_target[i, :a_len] = self.ans[index, i, :a_len]
            # ans_target[i, a_len] = self.vocab_size
            ans_len[i] = self.ans_len[index, i]
            ques_len[i] = self.ques_len[index, i]

            opt_ids = self.opt_ids[index, i]  # since python start from 0
            # random select the negative samples.
            ans_idx[i] = opt_ids[self.ans_ids[index, i]]
            ans_idx2[i] = self.ans_ids[index, i]
            for j, ids in enumerate(opt_ids):
                opt_len = self.opt_len[ids]
                opt_ans[i, j, :opt_len] = self.opt_list[ids,:opt_len]
                opt_ans_len[i, j] = opt_len

        his = torch.from_numpy(his)
        ques = torch.from_numpy(ques)
        ans = torch.from_numpy(ans)
        # ans_target = torch.from_numpy(ans_target)
        ans_len = torch.from_numpy(ans_len)
        ques_len = torch.from_numpy(ques_len)
        opt_ans_len = torch.from_numpy(opt_ans_len)
        opt_ans = torch.from_numpy(opt_ans)
        ans_idx = torch.from_numpy(ans_idx)
        ans_idx2 = torch.from_numpy(ans_idx2)
        opt_ans_idx = torch.from_numpy(opt_ans_idx)
        return img, his, ques, ans, ans_len, ans_idx2, ques_len, \
               opt_ans, opt_ans_len, opt_ans_idx

    def __len__(self):
        return self.ques.shape[0]