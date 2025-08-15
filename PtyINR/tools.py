import os
import time
import json  # Assuming commentjson behaves like json

# Scientific computing
import numpy as np
from numpy import zeros, newaxis, inf
import scipy
import h5py

# PyTorch and related libraries
import torch
from torch import nn
import torch.nn.functional as F


from PtyINR.train import *

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


        
def get_loss(parameters,rank):
    loss_type=parameters["loss"]
    if loss_type=="L1":
        if rank == 0:
            print("using L1 loss for training!")
        return nn.L1Loss()
    elif loss_type=="MSE":
        if rank == 0:
            print("using MSE loss for training!")
        return nn.MSELoss()
    else:
        if rank == 0:
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



