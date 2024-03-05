#-*-coding:utf-8-*-
import os
import shutil
import SimpleITK as sitk
import warnings
import glob
import numpy as np
import torch
from nipype.interfaces.ants import N4BiasFieldCorrection
import csv
import nibabel as nib
import pdb
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):

    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)
 
def normalize_image(in_file, out_file, bias_correction=True):
    if bias_correction:
        correct_bias(in_file, out_file)
    else:
        shutil.copy(in_file, out_file)
    return out_file


#Get input and ground truth 
#Ground truth processed by N4ITK

#Process AWS dataset
for i in range(100):
   
    in_file='./Data/AWS/prostate_{}'.format('%02d'%i)+'.nii.gz'
    #print(in_file)
    #pdb.set_trace()
    out_file='./Data/N4_AWS/prostate_{}'.format('%02d'%i)+'.nii.gz'
    if not os.path.exists(in_file):
        continue;
    t2w = nib.load(in_file)
    img_affine = t2w.affine
    t2w = t2w.get_fdata()
    # print(t2w.shape)
    # pdb.set_trace()
    t2w = t2w[:,:,:,0]
    t2w=t2w.transpose(2,0,1)
    
    t2w = torch.from_numpy(t2w).float()
    t2w = torch.unsqueeze(t2w,1)
    t2w = torch.nn.functional.interpolate(t2w, size = (256, 256), mode='nearest').float()
    t2w = torch.squeeze(t2w,1)
    t2w=t2w.numpy()
    
    t2w=t2w.transpose(1,2,0)
    
    pair_img = nib.Nifti1Pair(t2w, img_affine)
    nib.save(pair_img,in_file) 
    out=normalize_image(in_file,out_file)

#Process HK dataset
for i in range(100):
   
    in_file='./Data/HK/case{}'.format('%02d'%i)+'.nii.gz'
    #print(in_file)
    #pdb.set_trace()
    out_file='./Data/N4_HK/prostate_{}'.format('%02d'%i)+'.nii.gz'
    if not os.path.exists(in_file):
        continue;
    t2w = nib.load(in_file)
    img_affine = t2w.affine
    t2w = t2w.get_fdata()

    t2w=t2w.transpose(2,0,1)
    
    t2w = torch.from_numpy(t2w).float()
    t2w = torch.unsqueeze(t2w,1)
    t2w = torch.nn.functional.interpolate(t2w, size = (256, 256), mode='nearest').float()
    t2w = torch.squeeze(t2w,1)
    t2w=t2w.numpy()
    
    t2w=t2w.transpose(1,2,0)
  
    pair_img = nib.Nifti1Pair(t2w, img_affine)
    nib.save(pair_img,in_file) 
    out=normalize_image(in_file,out_file)
  
   
#Process HCRUDB dataset
for i in range(100):
   
    in_file='./Data/I2CVB/case{}'.format('%02d'%i)+'.nii.gz'
    #print(in_file)
    #pdb.set_trace()
    out_file='./Data/N4_I2CVB/prostate_{}'.format('%02d'%i)+'.nii.gz'
    if not os.path.exists(in_file):
        continue;
    t2w = nib.load(in_file)
    img_affine = t2w.affine
    t2w = t2w.get_fdata()
    
    t2w=t2w.transpose(2,0,1)
    
    t2w = torch.from_numpy(t2w).float()
    t2w = torch.unsqueeze(t2w,1)
    t2w = torch.nn.functional.interpolate(t2w, size = (256, 256), mode='nearest').float()
    t2w = torch.squeeze(t2w,1)
    t2w=t2w.numpy()
    
    t2w=t2w.transpose(1,2,0)
  
    pair_img = nib.Nifti1Pair(t2w, img_affine)
    nib.save(pair_img,in_file) 
    out=normalize_image(in_file,out_file)
   
   
#Process UCL dataset
for i in range(100):
   
    in_file='./Data/UCL/case{}'.format('%02d'%i)+'.nii.gz'
    #print(in_file)
    #pdb.set_trace()
    out_file='./Data/N4_UCL/prostate_{}'.format('%02d'%i)+'.nii.gz'
    if not os.path.exists(in_file):
        continue;
    t2w = nib.load(in_file)
    img_affine = t2w.affine
    t2w = t2w.get_fdata()
    
    t2w=t2w.transpose(2,0,1)
    
    t2w = torch.from_numpy(t2w).float()
    t2w = torch.unsqueeze(t2w,1)
    t2w = torch.nn.functional.interpolate(t2w, size = (256, 256), mode='nearest').float()
    t2w = torch.squeeze(t2w,1)
    t2w=t2w.numpy()
    
    t2w=t2w.transpose(1,2,0)
  
    pair_img = nib.Nifti1Pair(t2w, img_affine)
    nib.save(pair_img,in_file) 
    out=normalize_image(in_file,out_file)
   
   
