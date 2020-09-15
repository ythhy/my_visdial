import torchvision.models as model
from torch.autograd import Variable
import torch
import torch.nn as nn
from torchvision import transforms as trn
import skimage.io as io
from skimage import transform
import matplotlib.pyplot as plt

import numpy as np
import json
import h5py




for split in ['train']:
    with h5py.File('img_'+split+'_info.h5') as f:
        datafile = json.load(open('visdial_1.0_'+split+'.json'))
        dialog = datafile['data']['dialogs']
        imgdir = '/home/yth/visDial/data/img/VisualDialog_'+split+'2018/'
        imgnum = len(dialog)
        imginfo = np.zeros((imgnum),dtype=int)
        for i in range(imgnum):
            img_id = dialog[i]['image_id']
            imginfo[i] = img_id

        f.create_dataset('img_info',data=imginfo)
        if i%1000 ==0:
            print(i)
        f.close()