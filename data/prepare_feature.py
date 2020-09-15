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
import pdb


vgg19 = model.vgg19_bn(pretrained=True)

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(*list(vgg19.features.children()))
    def forward(self, x):
        x = self.features(x)
        return x

net = VGG19()

# pretrained_model = torch.load('/home/yth/visDial/data/vgg16_faster_rcnn_iter_1190000.pth')
# dict_model = net.state_dict().copy()
#
# model_list = list(net.state_dict().keys())
# trained_list = list (pretrained_model.keys())
#
# print(model_list)
#
# for i in range(26):
#     dict_model[model_list[i]] = pretrained_model[trained_list[i]]
#
# net.load_state_dict(dict_model)
net.cuda()
net.eval()
print(net)

preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



"""
read img

"""
for split in ['train']:
    with h5py.File('img_'+split+'_vgg19bn.h5') as f:
        datafile = json.load(open('visdial_1.0_'+split+'.json'))
        dialog = datafile['data']['dialogs']
        imgdir = '/home/yth/visDial/data/img/VisualDialog_'+split+'2018/'
        imgnum = len(dialog)
        map = torch.zeros(imgnum,7,7,512)
        for i in range(imgnum):
            img_id = str(dialog[i]['image_id']).zfill(12)
            img = io.imread(imgdir+'VisualDialog_'+split+'2018_'+img_id+'.jpg')
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate((img, img, img), axis=2)


            img = transform.resize(img,(224,224)).astype('float32')

            img = torch.from_numpy(img.transpose([2,0,1])).cuda()
            img = Variable(preprocess(img),volatile = True).unsqueeze(0)
            feat_map = net(img).transpose(1,2).transpose(2,3)
            map[i] = feat_map.data
            if i%500 == 0 :
                print(i)
        vggfeat = map.cpu().float().numpy()
        f.create_dataset('vgg_feat',data=vggfeat)
        f.close()






