import scipy
from numpy import zeros, newaxis
from numpy import random as np_random

import torch
import numpy as np
import torch
from torch import nn
import numpy as np
import skimage
from PIL import Image,ImageChops
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Grayscale, Pad
import matplotlib.pyplot as plt
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_msssim import ms_ssim, ssim

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
import h5py
from utils.Forward import *

import time


def get_step_size(overlap_ratio,probe_size):
    move_ratio=1-overlap_ratio
    step_size=round(probe_size*move_ratio)
    return step_size



def SOTA_Deep_Methods_Process(amplitude_gt,phase_gt,overlap_ratio,probe,parameters,model_name="PINN",noise="clean"
                              ,seed=1):
    random_seed=seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    step_size=get_step_size(overlap_ratio,probe.shape[0])
    print("step size: ",step_size)
    
    
    actual_object = torch.complex(amplitude_gt*torch.cos(phase_gt),amplitude_gt*torch.sin(phase_gt))
    actual_object=actual_object.to(parameters["device"])
    probe=probe.to(parameters["device"])
    horizontal_size=int((actual_object.shape[0]-probe.shape[0])/step_size+1)
    vertical_size=int((actual_object.shape[1]-probe.shape[1])/step_size+1)
    detection_size = int(probe.shape[0])
    total_amp=torch.zeros((horizontal_size,vertical_size,detection_size,detection_size))
    object_act=torch.zeros((horizontal_size,vertical_size,probe.shape[0],probe.shape[1])).type(torch.complex64)

    for i in range(vertical_size):
        for j in range(horizontal_size):
            total_amp[j,i]=forward(actual_object[j*step_size:j*step_size+probe.shape[0],i*step_size:i*step_size+probe.shape[1]]
                                   ,probe)
            object_act[j,i]=actual_object[j*step_size:j*step_size+probe.shape[0],i*step_size:i*step_size+probe.shape[1]]
    
    
    gaussian_low =  np.random.normal(0,6,total_amp.shape)
    gaussian_low= torch.tensor(gaussian_low)

    gaussian_high =  np.random.normal(0,10,total_amp.shape)
    gaussian_high= torch.tensor(gaussian_high)

    
    diff_amp=total_amp.cpu().detach().numpy()
    poisson_low =np_random.poisson(lam=(diff_amp**2)/3,size=diff_amp.shape)
    poisson_low = torch.tensor(poisson_low)*3

    poisson_high =np_random.poisson(lam=(diff_amp**2)/10,size=diff_amp.shape)
    poisson_high = torch.tensor(poisson_high)*10
    
    if noise=="clean":
        total_amp=total_amp
    elif noise=="gaussian_low":
        total_amp=total_amp**2+gaussian_low
        total_amp=torch.clamp(total_amp, min=0)
        total_amp=torch.sqrt(total_amp)
    elif noise=="gaussian_high":
        total_amp=total_amp**2+gaussian_high
        total_amp=torch.clamp(total_amp, min=0)
        total_amp=torch.sqrt(total_amp)
    elif noise=="poisson_low":
        total_amp=torch.sqrt(poisson_low)
    elif noise=="poisson_high":
        total_amp=torch.sqrt(poisson_high)
    elif noise=="combined":
        total_amp=poisson_high+gaussian_high
        total_amp=torch.clamp(total_amp, min=0)
        total_amp=torch.sqrt(total_amp)
    
    if model_name == "PtychoNN":
        return total_amp,object_act
        
    else:
        horizontal_size=int((actual_object.shape[0]-probe.shape[0])/step_size+1)
        vertical_size=int((actual_object.shape[1]-probe.shape[1])/step_size+1)
        size=int(horizontal_size/2)
        Loss_amp = torch.zeros((size,size,4, detection_size,detection_size)).to(parameters["device"])
        input_amp = torch.zeros((size,size ,4, 128,128)).to(parameters["device"])
        resize = transforms.Resize([128,128],antialias=True)
        
        

        for i in range(size):
            for j in range(size):
                Loss_amp[j,i,0]=total_amp[2*j,2*i]
                Loss_amp[j,i,1]=total_amp[2*j+1,2*i]
                Loss_amp[j,i,2]=total_amp[2*j,2*i+1]
                Loss_amp[j,i,3]=total_amp[2*j+1,2*i+1]

                input_amp[j,i,0]=resize(total_amp[2*j,2*i].view(1,1,detection_size,detection_size))
                input_amp[j,i,1]=resize(total_amp[2*j+1,2*i].view(1,1,detection_size,detection_size))
                input_amp[j,i,2]=resize(total_amp[2*j,2*i+1].view(1,1,detection_size,detection_size))
                input_amp[j,i,3]=resize(total_amp[2*j+1,2*i+1].view(1,1,detection_size,detection_size))
                    
        
    return Loss_amp,input_amp
    
    
    
def SCAN_Iterative_Methods_Process(amplitude_gt,phase_gt,overlap_ratio,probe,parameters
                                   ,noise="clean",seed=1):
    random_seed=seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    step_size=get_step_size(overlap_ratio,probe.shape[0])
    print("step size: ",step_size)
    
    actual_object = torch.complex(amplitude_gt*torch.cos(phase_gt),amplitude_gt*torch.sin(phase_gt))
    actual_object=actual_object.to(parameters["device"])
    probe=probe.to(parameters["device"])
    resolution = parameters["resolution"]
    scanpts=int((amplitude_gt.shape[0]-probe.shape[0])/step_size+1)
    detection_size = int(probe.shape[0])
    actual_amp=torch.tensor(np.zeros((scanpts*scanpts,detection_size,detection_size)))
    points = torch.tensor(np.zeros((2,scanpts*scanpts)))
    
    for i in range(scanpts):  # vertical 
        for j in range(scanpts):  # horizontal
            actual_amp[i*scanpts+j]=forward(actual_object[j*step_size:j*step_size+probe.shape[0]
                                                          ,i*step_size:i*step_size+probe.shape[1]],probe)
            st = step_size*resolution[-1]*1e6*(scanpts-1)/2
            points[0][i*scanpts+j] = -st+step_size*resolution[-2]*1e6*j# x j  um
            points[1][i*scanpts+j] = -st+step_size*resolution[-1]*1e6*i# y i
    
    if parameters["device"]=="cuda":
        actual_amp=actual_amp.detach().cpu().numpy()
    else:
        actual_amp=actual_amp.numpy()
        
    
    gaussian_low =  np.random.normal(0,6,actual_amp.shape)
    gaussian_high = np.random.normal(0,10,actual_amp.shape)
    diff_amp=actual_amp
    poisson_low = np_random.poisson(lam=(diff_amp ** 2) / 3, size=diff_amp.shape)
    poisson_low=poisson_low*3
    poisson_high = np_random.poisson(lam=(diff_amp ** 2) / 10, size=diff_amp.shape)
    poisson_high=poisson_high*10
    
    if noise=="clean":
        actual_amp=actual_amp
    elif noise=="gaussian_low":
        actual_amp=actual_amp**2+gaussian_low
        actual_amp=actual_amp.clip(min=0)
        actual_amp=np.sqrt(actual_amp)
    elif noise=="gaussian_high":
        actual_amp=actual_amp**2+gaussian_high
        actual_amp=actual_amp.clip(min=0)
        actual_amp=np.sqrt(actual_amp)
    elif noise=="poisson_low":
        actual_amp=np.sqrt(poisson_low)
    elif noise=="poisson_high":
        actual_amp = np.sqrt(poisson_high)
    elif noise=="combined":
        actual_amp=poisson_high+gaussian_high
        actual_amp=actual_amp.clip(min=0)
        actual_amp=np.sqrt(actual_amp)
    
    
    
    with h5py.File(parameters["path_to_data"],'w') as f1:
        angle = f1.create_dataset("angle", data=parameters["angle"])
        ccd_pixel_um = f1.create_dataset("ccd_pixel_um", data=parameters["ccd_pixel_um"])
        diffamp = f1.create_dataset("diffamp", data=actual_amp)
        lambda_nm = f1.create_dataset("lambda_nm", data=parameters["lambda_nm"])
        points = f1.create_dataset("points", data=points)
        z_m = f1.create_dataset("z_m", data=parameters["z_m"])
        
        f1.close()
    
    f = h5py.File(parameters["path_to_data"], 'r')
    #os.remove(parameters["path_to_data"])
    
    return f