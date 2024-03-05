# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 19:38:25 2023

@author: 28225
"""
import tensorflow.keras as keras # Importing required neural network module
import csv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt # Testing
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
# Just disables the warning, doesn't enable AVX/FMA
import os
import pdb
import SimpleITK as sitk
#matplotlib.use("TkAgg")
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Prepare for training dataset and testing dataset

x = []
y = []
for i in range(100):
    
   if not os.path.exists('./Data/N4_HK/prostate_{}'.format('%02d'%i)+'.nii.gz'):
      continue;
   # Loading each cropped image
   t2w_done = nib.load('./Data/N4_HK/prostate_{}'.format('%02d'%i)+'.nii.gz')
   t2w_done = t2w_done.get_fdata()
   t2w = nib.load('./Data/HK/case{}'.format('%02d'%i)+'.nii.gz')
   t2w = t2w.get_fdata()
  
   for j in range(t2w.shape[2]):
       x.append(t2w[:,:,j].tolist())
       y.append(t2w_done[:,:,j].tolist())

# Turning the image and label lists to a numpy arrays
X = np.array(x)
Y = np.array(y)
print(X.shape)

np.save('./x_HK',X)
np.save('./y_HK', Y)


x = []
y = []
for i in range(100):
    
   if not os.path.exists('./Data/N4_AWS/prostate_{}'.format('%02d'%i)+'.nii.gz'):
      continue;
   # Loading each cropped image
   t2w_done = nib.load('./Data/N4_AWS/prostate_{}'.format('%02d'%i)+'.nii.gz')
   t2w_done = t2w_done.get_fdata()
   t2w = nib.load('./Data/AWS/prostate_{}'.format('%02d'%i)+'.nii.gz')
   t2w = t2w.get_fdata()
  
   for j in range(t2w.shape[2]):
       x.append(t2w[:,:,j].tolist())
       y.append(t2w_done[:,:,j].tolist())

# Turning the image and label lists to a numpy arrays
X = np.array(x)
Y = np.array(y)
print(X.shape)
# pdb.set_trace()

np.save('./x_AWS',X)
np.save('./y_AWS', Y)


x = []
y = []
for i in range(100):
    
   if not os.path.exists('./Data/N4_HCRUDB/prostate_{}'.format('%02d'%i)+'.nii.gz'):
      continue;
   # Loading each cropped image
   t2w_done = nib.load('./Data/N4_HCRUDB/prostate_{}'.format('%02d'%i)+'.nii.gz')
   t2w_done = t2w_done.get_fdata()
   t2w = nib.load('./Data/HCRUDB/case{}'.format('%02d'%i)+'.nii.gz')
   t2w = t2w.get_fdata()
  
   for j in range(t2w.shape[2]):
       x.append(t2w[:,:,j].tolist())
       y.append(t2w_done[:,:,j].tolist())

# Turning the image and label lists to a numpy arrays
X = np.array(x)
Y = np.array(y)
print(X.shape)

np.save('./x_HCRUDB',X)
np.save('./y_HCRUDB', Y)


x = []
y = []
for i in range(100):
    
   if not os.path.exists('./Data/N4_UCL/prostate_{}'.format('%02d'%i)+'.nii.gz'):
      continue;
   # Loading each cropped image
   t2w_done = nib.load('./Data/N4_UCL/prostate_{}'.format('%02d'%i)+'.nii.gz')
   t2w_done = t2w_done.get_fdata()
   t2w = nib.load('./Data/UCL/case{}'.format('%02d'%i)+'.nii.gz')
   t2w = t2w.get_fdata()
  
   for j in range(t2w.shape[2]):
       x.append(t2w[:,:,j].tolist())
       y.append(t2w_done[:,:,j].tolist())

# Turning the image and label lists to a numpy arrays
X = np.array(x)
Y = np.array(y)
print(X.shape)

np.save('./x_train',X)
np.save('./y_train', Y)
