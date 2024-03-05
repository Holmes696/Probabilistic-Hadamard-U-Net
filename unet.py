
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:00:10 2023

@author: 28225
"""
import torch
import torch.nn as nn
#from unet_blocks import *
import torch.nn.functional as F
#from scipy.fft import dct, idct
import numpy as np
import pdb
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hadamard transform
def fwht(u, axis=-1, fast=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """  
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    if fast:
        x = u[..., np.newaxis]
        for d in range(m)[::-1]:
            x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    else:
        H = torch.tensor(hadamard(n), dtype=torch.float, device=u.device)
        y = u @ H
    if axis != -1:
        y = torch.transpose(y, -1, axis)
    return y

#Inverse Hadamard transform
def ifwht(u, axis=-1, fast=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """  
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    if fast:
        x = u[..., np.newaxis]
        for d in range(m)[::-1]:
            x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
        y = x.squeeze(-2) / n
    else:
        H = torch.tensor(hadamard(n), dtype=torch.float, device=u.device)
        y = u @ H /n
    if axis != -1:
        y = torch.transpose(y, -1, axis)
        
    return y

#Hard thresholding layer
class Thresholding(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.T = torch.nn.Parameter(torch.rand(self.num_features)/10)
              
    def forward(self, x):
        x= torch.copysign(torch.nn.functional.relu(torch.abs(x)-torch.abs(self.T)), x)
        y= x+torch.sign(x)*torch.abs(self.T)
        return y

def find_min_power(x, p=2):
    y = 1
    while y<x:
        y *= p
    return y

def TVLoss(x,weight):
    batch_size, c, h, w = x.size()
    tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
    return weight * (tv_h + tv_w) / (batch_size * c * h * w)

def KL_divergence(q):
    """
    Calculate the KL-divergence of (p,q)
    :param p:
    :param q:
    :return:
    """
    a=q.shape[0]
    b=q.shape[1]
    c=q.shape[2]
    d=q.shape[3]
    e=int(a*b*c*d/64)
    q=q.reshape(e,64)#/torch.max(abs(q))
    RHO = 0.001
    p = torch.FloatTensor([RHO for _ in range(64)]).to(device)
    p = torch.nn.functional.sigmoid(abs(p))  
    q = torch.nn.functional.sigmoid(abs(q)) 
    q = torch.sum(q, dim=0)/e  # dim:缩减的维度,q的第一维是batch维,即大小为batch_size大小,此处是将第j个神经元在batch_size个输入下所有的输出取平均
    s1 = torch.sum(p*torch.log(p/q))#计算第一部分
    s2 = torch.sum((1-p)*torch.log((1-p)/(1-q)))#计算第二部分

    return (s1+s2)

class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self,):
        super(Unet, self).__init__()
        
        #The first path
        self.height1=122
        self.width1=122
        self.height_pad1 = find_min_power(self.height1)  
        self.width_pad1 = find_min_power(self.width1)
        self.v1 = torch.nn.Parameter(torch.rand((self.height_pad1, self.width_pad1)))
        self.ST1 = Thresholding((self.height_pad1, self.width_pad1)) 
        
        self.height2=122
        self.width2=122
        self.height_pad2 = find_min_power(self.height2)  
        self.width_pad2 = find_min_power(self.width2)
        self.v2 = torch.nn.Parameter(torch.rand((self.height_pad2, self.width_pad2)))
        self.ST2 = Thresholding((self.height_pad2, self.width_pad2)) 
        
        self.height3=122
        self.width3=122
        self.height_pad3 = find_min_power(self.height3)  
        self.width_pad3 = find_min_power(self.width3)
        self.v3 = torch.nn.Parameter(torch.rand((self.height_pad3, self.width_pad3)))
        self.ST3 = Thresholding((self.height_pad3, self.width_pad3))

        self.conv1 = nn.Conv2d(1,4,16,2,padding=1)
        self.conv2 = nn.Conv2d(4,4,7, 1, padding=3)
        self.conv3 = nn.ConvTranspose2d(4,4,7, 1, padding=3)
        self.conv4 = nn.ConvTranspose2d(4,1,16, 2, padding=1)

    def forward(self, x, val):
        
        x1=self.conv1(x)
        if self.width_pad1>=self.width1 or self.height_pad1>=self.height1:
            x2 = torch.nn.functional.pad(x1, (0, self.width_pad1-self.width1, 0, self.height_pad1-self.height1))
        x3 = fwht(x2, axis=-1)
        x4 = fwht(x3, axis=-2)
        x5 = self.v1*x4
        x6 = self.ST1(x5)
        x7 = ifwht(x6, axis=-1)
        x8 = ifwht(x7, axis=-2)
        x9 = x8[..., :self.height1, :self.width1]
        
        x10=self.conv2(x9)
        if self.width_pad2>=self.width2 or self.height_pad2>=self.height2:
            x11 = torch.nn.functional.pad(x10, (0, self.width_pad2-self.width2, 0, self.height_pad2-self.height2))
        x12 = fwht(x11, axis=-1)
        x13 = fwht(x12, axis=-2)
        x14 = self.v2*x13
        x15 = self.ST2(x14)
        x16 = ifwht(x15, axis=-1)
        x17 = ifwht(x16, axis=-2)
        x18 = x17[..., :self.height2, :self.width2]
        
        x19=self.conv3(x18)
        x19=x19+x10
        if self.width_pad3>=self.width3 or self.height_pad3>=self.height3:
            x20 = torch.nn.functional.pad(x19, (0, self.width_pad3-self.width3, 0, self.height_pad3-self.height3))
        x21 = fwht(x20, axis=-1)
        x22 = fwht(x21, axis=-2)
        x23 = self.v3*x22
        x24 = self.ST3(x23)
        x224 = x24+x6
        x25 = ifwht(x224, axis=-1)
        x26 = ifwht(x25, axis=-2)
        x27 = x26[..., :self.height3, :self.width3]        
 
        x28=self.conv4(x27)

        x=x28*x
    
        loss_TV=TVLoss(x,np.random.uniform(0,0.1))
        loss_KL=KL_divergence(x6)+KL_divergence(x15)+KL_divergence(x24)
        loss = 1*loss_TV+0.1*loss_KL
        
        return x,loss

