
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:09:38 2023

@author: 28225
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
import pdb
import math
import nibabel as nib
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Process segmenatation labels
x = []
y = []
for i in range(50):
    if not os.path.exists('../Data_preprocess/Data/HK/case{}_segmentation'.format('%02d'%i)+'.nii.gz'):
      continue;
    t2w = nib.load('../Data_preprocess/Data/HK/case{}_segmentation'.format('%02d'%i)+'.nii.gz')
    t2w = t2w.get_fdata()
    for j in range(t2w.shape[2]):
        x.append(t2w[:,:,j].tolist())

# Turning the image and label lists to a numpy arrays
X = np.array(x)

test_f_np = np.load('../Data_preprocess/x_AWS.npy')
test_g_np = np.load('../Data_preprocess/y_AWS.npy')


#Normalize to 0~1
test_f_np=(test_f_np-np.min(test_f_np))
test_f_np=test_f_np/np.max(abs(test_f_np))
test_g_np=(test_g_np-np.min(test_g_np))
test_g_np=test_g_np/np.max(abs(test_g_np))
X=(X-np.min(X))
X=X/np.max(abs(X))

#To pytorch
test_f = torch.from_numpy(test_f_np).float()
test_g = torch.from_numpy(test_g_np).float()
X = torch.from_numpy(X).float()

test_f = torch.unsqueeze(test_f,1)
test_g = torch.unsqueeze(test_g,1)

#Downsample to 256x256
X = torch.unsqueeze(X,1)
X = torch.nn.functional.interpolate(X, size = (256, 256), mode='nearest').float()
X = torch.squeeze(X,1)
test_l_np=X.numpy()

test_dataset = torch.utils.data.TensorDataset(test_f,test_g)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,drop_last=False)

#Model
net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64], latent_dim=2, no_convs_fcomb=4, beta=10)
net.to(device)
net.eval()
net.load_state_dict(torch.load('./weight.pt'))

#Test
test_p=[]
num_batches = len(test_loader)#batch的数量
for step,(patch,mask) in enumerate(test_loader): 
    net(patch.to(device), segm=None,training=False)
    test_o = net.sample(testing=True)
    test_o = torch.squeeze(test_o,1)
    test_o_np = test_o.cpu().detach().numpy()
    for i in range(1):
        test_p.append(test_o_np[i])
test_p=np.array(test_p)



















