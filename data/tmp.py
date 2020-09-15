import h5py
import numpy as np


f1 = h5py.File('img_val_resnet.h5','r')
f = f1['resnet_feat'][:]
f2 = h5py.File('img_val_resnet2.h5','w')

f_convert = f.astype('float32')

f2.create_dataset('resnet_feat',data=f_convert)
f1.close()
f2.close()
