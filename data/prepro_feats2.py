from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import h5py
from random import shuffle, seed
import torchvision.models as model
import numpy as np
import torch
from torch.autograd import Variable
import skimage.io as io
from torchvision import transforms as trn
from skimage import transform
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc2.resnet_utils import myResnet
import misc2.resnet as resnet

net = getattr(resnet, 'resnet101')()
# net = model.resnet101(pretrained=True)

net.load_state_dict(torch.load(os.path.join('/home/yth/visDial/data/resnet101.pth')))
my_resnet = myResnet(net)
my_resnet.cuda()
my_resnet.eval()


"""
read img

"""
for split in ['train']:
    with h5py.File('img_'+split+'.h5') as f:
        datafile = json.load(open('visdial_1.0_'+split+'.json'))
        dialog = datafile['data']['dialogs']
        imgdir = '/home/yth/visDial/data/img/VisualDialog_'+split+'2018/'
        imgnum = len(dialog)
        map = torch.zeros(imgnum,7,7,2048)
        for i in range(imgnum):
            img_id = str(dialog[i]['image_id']).zfill(12)
            I = io.imread(imgdir+'VisualDialog_'+split+'2018_'+img_id+'.jpg')
            if len(I.shape) == 2:
                I = I[:, :, np.newaxis]
                I = np.concatenate((I, I, I), axis=2)

            I = I.astype('float32') / 255.0
            I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
            I = Variable(preprocess(I), volatile=True)
            tmp_fc, tmp_att = my_resnet(I, 7)
            map[i] = tmp_att.data
            if i%500 == 0 :
                print(i)
        resfeat = map.cpu().float().numpy()
        f.create_dataset('resnet_feat',data=resfeat)
        f.close()






