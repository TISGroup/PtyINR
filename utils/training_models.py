import scipy
from numpy import zeros, newaxis
# !pip install pytorch-msssim
from pytorch_msssim import ms_ssim, ssim 
from torchmetrics.image import PeakSignalNoiseRatio
import random
import os
import numpy as np
import h5py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Grayscale, Pad
import numpy as np
import skimage
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm 
import time
from torchsummary import summary
import h5py, os
import numpy as np
import numpy as np
import scipy.constants as C
import time
from torchmetrics.image import PeakSignalNoiseRatio, TotalVariation
import commentjson as json
from utils.forward import *
from utils.deep_models import *
import tinycudann as tcnn
from numpy import inf
from torch.optim.lr_scheduler import CosineAnnealingLR

config = {
    "encoding": {
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 15,
        "base_resolution": 16,
        "per_level_scale": 1.5
    },
    "network": {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2
    }
}

class PtychographyDataset(Dataset):
    def __init__(self, coordinates_x,coordinates_y, diffraction_patterns):
        """
        Args:
            coordinates (numpy array): Shape (40401, 2) - input coordinates.
            diffraction_patterns (numpy array): Shape (40401, 220, 220) - measured patterns.
        """
        self.coordinates_x = torch.tensor(coordinates_x, dtype=torch.int32)#.cuda()  # Convert coordinates to tensor
        self.coordinates_y = torch.tensor(coordinates_y, dtype=torch.int32)#.cuda()
        self.diffraction_patterns = torch.tensor(diffraction_patterns, dtype=torch.float32).cuda()  # Convert patterns to tensor

    def __len__(self):
        # Return the total number of samples
        return len(self.coordinates_x)

    def __getitem__(self, idx):
        # Retrieve a single coordinate and corresponding diffraction pattern
        return self.coordinates_x[idx],self.coordinates_y[idx], self.diffraction_patterns[idx]

    
def grid_return(resolution):
    H=resolution[0]
    W=resolution[1]
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H),  # y-axis (height)
        torch.linspace(-1, 1, W),  # x-axis (width)
        indexing='ij'  # 'ij' indexing for row-major order
    )
    
    # Stack and reshape into (H * W, 2) where each row contains (x, y) coordinates
    coords_2d = torch.stack([x_coords, y_coords], dim=-1).view(-1, 2)
    return coords_2d





def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid



        
def get_loss(parameters):
    loss_type=parameters["loss"]
    if loss_type=="L1":
        print("using L1 loss for training!")
        return nn.L1Loss()
    elif loss_type=="MSE":
        print("using MSE loss for training!")
        return nn.MSELoss()
    else:
        print("using SmoothL1 loss for training!")
        return nn.SmoothL1Loss(beta=parameters["beta_for_smoothl1"])



def get_model(parameters,obj_size,config):
    model_type=parameters["model_type"]
    if model_type=="siren":
        obj_net_amp=Siren(in_features=2, out_features=1, hidden_features=512, 
                  hidden_layers=3, outermost_linear=True, first_omega_0=parameters["first_omega"], hidden_omega_0=30)
        obj_net_phase=Siren(in_features=2, out_features=1, hidden_features=512, 
                  hidden_layers=3, outermost_linear=True, first_omega_0=parameters["first_omega"], hidden_omega_0=30)
        return 1,get_mgrid(obj_size),obj_net_amp,obj_net_phase
    else:
        print("Using ReLU for representing object instead of PtyINR default structure!")
        obj_net_amp=tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                                    encoding_config=config["encoding"], network_config=config["network"])
        obj_net_phase=tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                                    encoding_config=config["encoding"], network_config=config["network"])
        return 1e4,grid_return([obj_size,obj_size]),obj_net_amp,obj_net_phase



        
def recenter_probe(probe):

    center=int(probe.shape[0]/2)
    max_index=(probe.abs()==torch.max(probe.abs())).nonzero()
    shift=(center-max_index[0,0],center-max_index[0,1])
    recentered_probe = torch.roll(probe, shifts=tuple(shift), dims=(0, 1))

    return recentered_probe

def train_Pty_INR_SGD(f,parameters,probe):   
    ratio=parameters["regularized_loss_weight"]
    regularized_steps=parameters["regularized_steps"]
    show_every=parameters["show_every"]  
    if parameters["amp_shift"]==False:
        actual_amp=f['diffamp'][()]
    else:        
        actual_amp=f['diffamp'][()]
        actual_amp=np.fft.fftshift(actual_amp,axes=(-2,-1))
    

    
    actual_amp=actual_amp/parameters["diffraction_scale"]
    frames_count=actual_amp.shape[0]
    scan_size=int(np.sqrt(f['points'][()].shape[1]))
    pixel_size=parameters["pixel_size"]
    x_axis=np.empty([scan_size, scan_size])
    y_axis=np.empty([scan_size, scan_size])
    
    x_coord=f['points'][()][0].reshape(scan_size,scan_size)
    y_coord=f['points'][()][1].reshape(scan_size,scan_size)
    for i in range(scan_size):
        for j in range(scan_size):
            x_axis[i,j]=round((x_coord[i,j]-x_coord[0,0])/pixel_size)
            y_axis[i,j]=round((y_coord[i,j]-y_coord[0,0])/pixel_size)

    x_axis=(x_axis-x_axis.min()).reshape(-1)
    y_axis=(y_axis-y_axis.min()).reshape(-1)
    a=x_axis.max()
    b=y_axis.max()
    obj_size=int(max(a,b))+actual_amp.shape[1]
    
    print("The shape for diffraction patterns are ",actual_amp.shape)
    dataset = PtychographyDataset(x_axis,y_axis,actual_amp)
    dataloader = DataLoader(dataset, batch_size=parameters["batches"], shuffle=True, num_workers=0)

    criterion = get_loss(parameters)
    total_steps=parameters["total_steps"]
    min_loss = float('inf')
    pred_scale,obj_grid,obj_net_amp,obj_net_phase=get_model(parameters,obj_size,config)
    obj_net_amp=obj_net_amp.cuda()
    obj_net_phase=obj_net_phase.cuda()
    obj_grid=get_mgrid(obj_size).cuda()
    
    model_probe_amp = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                                    encoding_config=config["encoding"], network_config=config["network"])
    model_probe_phase = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                                      encoding_config=config["encoding"], network_config=config["network"])
    
    model_probe_amp=model_probe_amp.cuda()
    model_probe_phase=model_probe_phase.cuda()
    probe_grid=grid_return([actual_amp.shape[1],actual_amp.shape[1]]).cuda()
    probe_shape=actual_amp.shape[1]
    optim = torch.optim.Adam(lr=parameters["LR"], params=obj_net_amp.parameters())
    optim2 = torch.optim.Adam(lr=parameters["LR2"], params=obj_net_phase.parameters())
    optim3 = torch.optim.Adam(lr=parameters["LR3"], params=model_probe_amp.parameters())
    optim4 = torch.optim.Adam(lr=parameters["LR4"], params=model_probe_phase.parameters())
    accumlated_loss=0
    imshow=parameters["image_show"]
    probe_amp=torch.abs(probe)
    total_batches=len(dataloader)
    probe=probe.cuda()
    
    start = time.time()
    for step in range(total_steps):
        accumlated_loss=0
        for batch_idx, (x_axis,y_axis, measured_patterns) in enumerate(dataloader):
            loss=0
            predict_amp=obj_net_amp(obj_grid)*pred_scale
            obj_amp=torch.abs(predict_amp).view(obj_size,obj_size)
            obj_phase=obj_net_phase(obj_grid).view(obj_size,obj_size)*pred_scale
            object_act=torch.complex(obj_amp*torch.cos(obj_phase),obj_amp*torch.sin(obj_phase))

            if parameters["probe_known"]==False:
                predict2=model_probe_amp(probe_grid)
                probe_amp=torch.abs(predict2).view(probe_shape,probe_shape)*1e4
                probe_phase=model_probe_phase(probe_grid).view(probe_shape,probe_shape)*1e4
                probe=torch.complex(probe_amp*torch.cos(probe_phase),probe_amp*torch.sin(probe_phase))
                probe=probe/(torch.abs(probe).max())
                probe=recenter_probe(probe)
                

            x_indices = x_axis
            y_indices = y_axis
            x_limits = x_indices + probe.shape[0]
            y_limits = y_indices + probe.shape[1]
            
            # Extract all part_obj in a single batch operation
            batch_part_obj = torch.stack([
                object_act[x.item():x_l.item(), y.item():y_l.item()]
                for x, x_l, y, y_l in zip(x_indices, x_limits, y_indices, y_limits)
            ], dim=0)
            
            # Perform Fourier transforms in batch
            pre_amp_batch = forward(batch_part_obj, probe)
        
            loss1 = criterion(pre_amp_batch, measured_patterns)
            loss2 = torch.abs(probe).mean()
            if step>=regularized_steps:
                ratio=0
            loss= loss1+ratio*loss2
            accumlated_loss+=loss.item()*len(batch_part_obj)/frames_count
            loss.backward()
            optim.step()
            optim2.step()
            optim3.step()
            optim4.step()
            optim.zero_grad()
            optim2.zero_grad()
            optim3.zero_grad()
            optim4.zero_grad()
        
        
        if step % show_every==0:
            if imshow==True:
                print("Step %d, Total loss %0.6f" % (step, accumlated_loss))
                fig, axes = plt.subplots(1,4, figsize=(18,6))
                axes[0].imshow(torch.rot90(torch.abs(object_act),k=4).cpu().detach().numpy(),cmap="grey")
                axes[1].imshow(torch.rot90(torch.angle(object_act),k=4).cpu().detach().numpy(),cmap="magma_r")
                axes[2].imshow(torch.abs(probe).cpu().detach().numpy(),cmap="grey")
                axes[3].imshow(torch.angle(probe).cpu().detach().numpy(),cmap="magma")
                plt.show()
            else:     
                print("Step %d, Total loss %0.6f" % (step, accumlated_loss))
                plt.imsave(parameters["save_path"]+"amp.jpg",obj_amp.detach().cpu().numpy(),cmap="grey")
                plt.imsave(parameters["save_path"]+"phase.jpg",obj_phase.detach().cpu().numpy(),cmap="magma_r")
                plt.imsave(parameters["save_path"]+"probe_amp.jpg",torch.abs(probe).detach().cpu().numpy(),cmap="grey")
                plt.imsave(parameters["save_path"]+"probe_phase.jpg",torch.angle(probe).detach().cpu().numpy(),cmap="magma")
    

    
        if accumlated_loss < min_loss:
            min_loss = accumlated_loss
            best_obj = object_act
            best_probe = probe
            np.save(parameters["save_path"]+parameters["tag"]+"_obj"+".npy",best_obj.cpu().detach().numpy())
            np.save(parameters["save_path"]+parameters["tag"]+"_probe"+".npy",best_probe.cpu().detach().numpy())


            
    end = time.time()
    print("running time is ",end - start,"s")
    best_obj=best_obj.cpu().detach().numpy()
    best_probe=best_probe.cpu().detach().numpy()
    np.save(parameters["save_path"]+parameters["tag"]+"_obj"+".npy",best_obj)
    np.save(parameters["save_path"]+parameters["tag"]+"_probe"+".npy",best_probe)



def train_Pty_INR_GD(f,parameters,probe):           
    ratio=parameters["regularized_loss_weight"]
    regularized_steps=parameters["regularized_steps"]
    show_every=parameters["show_every"]  
    if parameters["amp_shift"]==False:
        actual_amp=f['diffamp'][()]
    else:        
        actual_amp=f['diffamp'][()]
        actual_amp=np.fft.fftshift(actual_amp,axes=(-2,-1))
    

    
    actual_amp=actual_amp/parameters["diffraction_scale"]
    frames_count=actual_amp.shape[0]
    scan_size=int(np.sqrt(f['points'][()].shape[1]))
    pixel_size=parameters["pixel_size"]
    x_axis=np.empty([scan_size, scan_size])
    y_axis=np.empty([scan_size, scan_size])
    
    x_coord=f['points'][()][0].reshape(scan_size,scan_size)
    y_coord=f['points'][()][1].reshape(scan_size,scan_size)
    for i in range(scan_size):
        for j in range(scan_size):
            x_axis[i,j]=round((x_coord[i,j]-x_coord[0,0])/pixel_size)
            y_axis[i,j]=round((y_coord[i,j]-y_coord[0,0])/pixel_size)

    x_axis=(x_axis-x_axis.min()).reshape(-1)
    y_axis=(y_axis-y_axis.min()).reshape(-1)
    a=x_axis.max()
    b=y_axis.max()
    obj_size=int(max(a,b))+actual_amp.shape[1]
    
    print("The shape for diffraction patterns are ",actual_amp.shape)
    dataset = PtychographyDataset(x_axis,y_axis,actual_amp)
    dataloader = DataLoader(dataset, batch_size=parameters["batches"], shuffle=True, num_workers=0)

    criterion = get_loss(parameters)
    total_steps=parameters["total_steps"]
    min_loss = float('inf')
    pred_scale,obj_grid,obj_net_amp,obj_net_phase=get_model(parameters,obj_size,config)
    obj_net_amp=obj_net_amp.cuda()
    obj_net_phase=obj_net_phase.cuda()
    obj_grid=get_mgrid(obj_size).cuda()
    
    model_probe_amp = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                                    encoding_config=config["encoding"], network_config=config["network"])
    model_probe_phase = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                                      encoding_config=config["encoding"], network_config=config["network"])
    
    model_probe_amp=model_probe_amp.cuda()
    model_probe_phase=model_probe_phase.cuda()
    probe_grid=grid_return([actual_amp.shape[1],actual_amp.shape[1]]).cuda()
    probe_shape=actual_amp.shape[1]
    optim = torch.optim.Adam(lr=parameters["LR"], params=obj_net_amp.parameters())
    optim2 = torch.optim.Adam(lr=parameters["LR2"], params=obj_net_phase.parameters())
    optim3 = torch.optim.Adam(lr=parameters["LR3"], params=model_probe_amp.parameters())
    optim4 = torch.optim.Adam(lr=parameters["LR4"], params=model_probe_phase.parameters())
    accumlated_loss=0
    imshow=parameters["image_show"]
    probe_amp=torch.abs(probe)
    total_batches=len(dataloader)
    probe=probe.cuda()
    
    start = time.time()
    for step in range(total_steps):
        accumlated_loss=0
        for batch_idx, (x_axis,y_axis, measured_patterns) in enumerate(dataloader):
            loss=0
            predict_amp=obj_net_amp(obj_grid)*pred_scale
            obj_amp=torch.abs(predict_amp).view(obj_size,obj_size)
            obj_phase=obj_net_phase(obj_grid).view(obj_size,obj_size)*pred_scale
            object_act=torch.complex(obj_amp*torch.cos(obj_phase),obj_amp*torch.sin(obj_phase))

            if parameters["probe_known"]==False:
                predict2=model_probe_amp(probe_grid)
                probe_amp=torch.abs(predict2).view(probe_shape,probe_shape)*1e4
                probe_phase=model_probe_phase(probe_grid).view(probe_shape,probe_shape)*1e4
                probe=torch.complex(probe_amp*torch.cos(probe_phase),probe_amp*torch.sin(probe_phase))
                probe=probe/(torch.abs(probe).max())
                probe=recenter_probe(probe)
                

            x_indices = x_axis
            y_indices = y_axis
            x_limits = x_indices + probe.shape[0]
            y_limits = y_indices + probe.shape[1]
            
            # Extract all part_obj in a single batch operation
            batch_part_obj = torch.stack([
                object_act[x.item():x_l.item(), y.item():y_l.item()]
                for x, x_l, y, y_l in zip(x_indices, x_limits, y_indices, y_limits)
            ], dim=0)
            
            # Perform Fourier transforms in batch
            pre_amp_batch = forward(batch_part_obj, probe)
        
            loss1 = criterion(pre_amp_batch, measured_patterns)
            loss1=loss1*len(batch_part_obj)/frames_count
            loss2 = torch.abs(probe).mean()/total_batches
            if step>=regularized_steps:
                ratio=0
            loss= loss1+ratio*loss2
            accumlated_loss+=loss.item()*len(batch_part_obj)/frames_count
            loss.backward()
        optim.step()
        optim2.step()
        optim3.step()
        optim4.step()
        optim.zero_grad()
        optim2.zero_grad()
        optim3.zero_grad()
        optim4.zero_grad()
        
        
        if step % show_every==0:
            if imshow==True:
                print("Step %d, Total loss %0.6f" % (step, accumlated_loss))
                fig, axes = plt.subplots(1,4, figsize=(18,6))
                axes[0].imshow(torch.rot90(torch.abs(object_act),k=4).cpu().detach().numpy(),cmap="grey")
                axes[1].imshow(torch.rot90(torch.angle(object_act),k=4).cpu().detach().numpy(),cmap="magma_r")
                axes[2].imshow(torch.abs(probe).cpu().detach().numpy(),cmap="grey")
                axes[3].imshow(torch.angle(probe).cpu().detach().numpy(),cmap="magma")
                plt.show()
            else:     
                print("Step %d, Total loss %0.6f" % (step, accumlated_loss))
                plt.imsave(parameters["save_path"]+"amp.jpg",obj_amp.detach().cpu().numpy(),cmap="grey")
                plt.imsave(parameters["save_path"]+"phase.jpg",obj_phase.detach().cpu().numpy(),cmap="magma_r")
                plt.imsave(parameters["save_path"]+"probe_amp.jpg",torch.abs(probe).detach().cpu().numpy(),cmap="grey")
                plt.imsave(parameters["save_path"]+"probe_phase.jpg",torch.angle(probe).detach().cpu().numpy(),cmap="magma")
    

    
        if accumlated_loss < min_loss:
            min_loss = accumlated_loss
            best_obj = object_act
            best_probe = probe
            np.save(parameters["save_path"]+parameters["tag"]+"_obj"+".npy",best_obj.cpu().detach().numpy())
            np.save(parameters["save_path"]+parameters["tag"]+"_probe"+".npy",best_probe.cpu().detach().numpy())


            
    end = time.time()
    print("running time is ",end - start,"s")
    best_obj=best_obj.cpu().detach().numpy()
    best_probe=best_probe.cpu().detach().numpy()
    np.save(parameters["save_path"]+parameters["tag"]+"_obj"+".npy",best_obj)
    np.save(parameters["save_path"]+parameters["tag"]+"_probe"+".npy",best_probe)
    


    
    
    
    
    
    
def train_model(parameters,f,probe):
    if parameters["train_method"] == "mini_batch":
        print("Using batch optimization")
        train_Pty_INR_SGD(f,parameters,probe)
    else:
        print("Using full data for optimization")
        train_Pty_INR_GD(f,parameters,probe)
