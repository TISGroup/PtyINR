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
import os
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
import ptypy, os
import ptypy.utils as u
import numpy as np
import scipy.constants as C
import time
from torchmetrics.image import PeakSignalNoiseRatio, TotalVariation
import commentjson as json
from utils.Forward import *
from utils.Deep_Models import *
import tinycudann as tcnn
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
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def psnr_amp(original, compressed):
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return [psnr,ssim(original,compressed,data_range=amp_act.max()-amp_act.min())] 


def psnr_phase(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = np.pi*2
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return [psnr,ssim(original,compressed,data_range=phase_act.max()-phase_act.min())]

def print_the_metric(amp_act,phase_act,obj):
    
    obj=obj[:241,:241]
    amp=np.abs(obj)[30:-30,30:-30]
    amp=amp*np.median(amp_act)/np.median(amp)
    amp=amp.clip(max=1)
    amp=amp.clip(min=0)
    phase=np.angle(obj)[30:-30,30:-30]
    phase=phase-np.median(phase)
    print(" : ",round(psnr_amp(amp,amp_act)[0],2),"/",round(psnr_amp(amp,amp_act)[1],2),"  "
          ,round(psnr_phase(phase,phase_act)[0],2),"/",round(psnr_phase(phase,phase_act)[1],2))

        
def get_sub_grid(scan_size,coord,x_axis,y_axis,obj_size,actual_amp):
    sub_grid=torch.zeros((scan_size*scan_size,actual_amp.shape[1],actual_amp.shape[2],2)).cuda()
    x_total=coord[:,0].view(obj_size,obj_size)
    y_total=coord[:,1].view(obj_size,obj_size)
    for step_h in range(scan_size):
        for step_v in range(scan_size):
            x = int(x_axis[step_h, step_v])
            y = int(y_axis[step_h, step_v])
            x_limit = x + actual_amp.shape[1]
            y_limit = y + actual_amp.shape[2]
            
            sub_grid[step_h*scan_size+step_v,:,:,0] = x_total[x:x_limit, y:y_limit]
            sub_grid[step_h*scan_size+step_v,:,:,1] = y_total[x:x_limit, y:y_limit]
    return sub_grid




    

def recenter_probe(probe):

    center=int(probe.shape[0]/2)
    max_index=(probe.abs()==torch.max(probe.abs())).nonzero()
    shift=(center-max_index[0,0],center-max_index[0,1])
    recentered_probe = torch.roll(probe, shifts=tuple(shift), dims=(0, 1))

    return recentered_probe





def train_Pty_INR(f,parameters,probe,c):
    with open("config_hash.json") as l:
        config = json.load(l)           
    ratio=parameters["regularized_loss_weight"]
    if parameters["amp_shift"]==False:
        actual_amp=f['diffamp'][()]
    else:
        actual_amp=np.fft.fftshift(f['diffamp'][()],axes=(-2,-1))

    actual_amp=torch.tensor(actual_amp).to(parameters["device"])
    actual_amp=actual_amp/parameters["diffraction_scale"]
    scan_size=int(np.sqrt(f['points'][()].shape[1]))
    pixel_size=parameters["pixel_size"]
    print("pixel size: ",pixel_size)
    x_axis=np.empty([scan_size, scan_size])
    y_axis=np.empty([scan_size, scan_size])
    
    x_coord=f['points'][()][0].reshape(scan_size,scan_size)
    y_coord=f['points'][()][1].reshape(scan_size,scan_size)
    for i in range(scan_size):
        for j in range(scan_size):
            x_axis[i,j]=round((x_coord[i,j]-x_coord[0,0])/pixel_size)
            y_axis[i,j]=round((y_coord[i,j]-y_coord[0,0])/pixel_size)

    x_axis=x_axis-x_axis.min()
    y_axis=y_axis-y_axis.min()
    a=x_axis.max()
    b=y_axis.max()
    obj_size=int(max(a,b))+actual_amp.shape[1]

    
    criterion = nn.L1Loss()
    total_steps=parameters["total_steps"]
    min_loss = float('inf')
    obj_net=Siren(in_features=2, out_features=2, hidden_features=512, 
                  hidden_layers=3, outermost_linear=True, first_omega_0=parameters["first_omega"], hidden_omega_0=30)
    obj_net=obj_net.cuda()
    obj_grid=get_mgrid(obj_size).cuda()
    model_probe_amp = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                                    encoding_config=config["encoding"], network_config=config["network"])
    model_probe_phase = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                                      encoding_config=config["encoding"], network_config=config["network"])
    
    model_probe_amp=model_probe_amp.cuda()
    model_probe_phase=model_probe_phase.cuda()
    probe_grid=grid_return([actual_amp.shape[1],actual_amp.shape[1]]).cuda()
    probe_shape=actual_amp.shape[1]
    optim = torch.optim.Adam(lr=parameters["LR"], params=obj_net.parameters())
    optim2 = torch.optim.Adam(lr=parameters["LR2"], params=model_probe_amp.parameters())
    optim3 = torch.optim.Adam(lr=parameters["LR2"], params=model_probe_phase.parameters())
    for step in range(total_steps):

        loss=0
        predict=obj_net(obj_grid)
        obj_amp=torch.abs(predict[:,0]).view(obj_size,obj_size)
        obj_phase=predict[:,1].view(obj_size,obj_size)
        object_act=torch.complex(obj_amp*torch.cos(obj_phase),obj_amp*torch.sin(obj_phase))
        
        predict2=model_probe_amp(probe_grid)#.view(112,112)
        probe_amp=torch.abs(predict2).view(probe_shape,probe_shape)*1e4
        probe_phase=model_probe_phase(probe_grid).view(probe_shape,probe_shape)*1e4
        probe=torch.complex(probe_amp*torch.cos(probe_phase),probe_amp*torch.sin(probe_phase))
        probe=probe/(torch.abs(probe).max())
        probe=recenter_probe(probe)
        x_indices = x_axis.reshape(-1)  # Flatten x_axis
        y_indices = y_axis.reshape(-1)  # Flatten y_axis
        x_limits = x_indices + probe.shape[0]
        y_limits = y_indices + probe.shape[1]
        
        # Extract all part_obj in a single batch operation
        batch_part_obj = torch.stack([
            object_act[int(x):int(x_l), int(y):int(y_l)]
            for x, x_l, y, y_l in zip(x_indices, x_limits, y_indices, y_limits)
        ], dim=0)
        
        # Perform Fourier transforms in batch
        pre_amp_batch = forward(batch_part_obj, probe)
    
        loss1 = criterion(pre_amp_batch, actual_amp)
        loss2 = torch.abs(probe).mean()
        if step>50:
            ratio=0
        loss= loss1+ratio*loss2
                
    


        
        optim.zero_grad()
        optim2.zero_grad()
        optim3.zero_grad()
        loss.backward()
        optim.step()
        optim2.step()
        optim3.step()

        
        
        if step % 50==0:
            print("Step %d, Total obj loss %0.6f" % (step, loss))
            plt.imsave(("amp.jpg"),obj_amp.detach().cpu().numpy())
            plt.imsave(("phase.jpg"),obj_phase.detach().cpu().numpy())
            plt.imsave(("probe_amp.jpg"),torch.abs(probe).detach().cpu().numpy())
            plt.imsave(("probe_phase.jpg"),torch.angle(probe).detach().cpu().numpy())
    

        
        if loss < min_loss:
            min_loss = loss
            best_density = obj_amp
            best_phase = obj_phase
    
    obj=torch.complex(best_density*torch.cos(best_phase),best_density*torch.sin(best_phase))
    obj=obj.cpu().detach().numpy()
    np.save(parameters["tag"]+".npy",obj)
    


    
def train_Iterative(f,parameters,probe,method):
    
    
    
    path_to_data_xy="temp2.h5"
    yarr = f['points'][1]
    xarr = f['points'][0]
    diff = f['diffamp'][:]**2
    z_m=f['z_m'][()]
    ccd_pixel_um=f['ccd_pixel_um'][()]
    
    if probe.shape[0]<diff.shape[-1]:
        shape_size= parameters["shape_size"]
        padsize = (shape_size-probe.shape[0])//2
        pad = nn.ZeroPad2d(padsize)
        probe_model=np.array(pad(probe).cpu().detach())
    else:
        probe_model=probe.cpu().numpy()
        print("probe shape: ",probe_model.shape)
    
    with h5py.File(path_to_data_xy,'w') as f1:
        posy_um = f1.create_dataset("posy_um", data=yarr)
        posx_um = f1.create_dataset("posx_um", data=xarr)
        diffinten = f1.create_dataset("diffinten", data=diff)
        z_m_2=f1.create_dataset("z_m", data=z_m)
        ccd_pixel_um_2=f1.create_dataset("ccd_pixel_um", data=ccd_pixel_um)
        
    
    
    
    ptypy.load_ptyscan_module("hdf5_loader")

    ptypy.load_gpu_engines(arch="cupy")

    p = u.Param()

    # Set verbose level to interactive
    p.verbose_level = "interactive"

    # Set io settings (no files saved)
    p.io = u.Param()
    p.io.autosave = u.Param(active=True)
    p.io.interaction = u.Param(active=True)
    # Path to final .ptyr output file 
    # using variables p.run, engine name and total nr. of iterations
    p.io.rfile =  "recons/%(run)s_%(engine)s_%(iterations)04d.ptyr"

    # Use non-threaded live plotting
    p.io.autoplot = u.Param()
    p.io.autoplot.active=True
    p.io.autoplot.threaded = False
    p.io.autoplot.layout = "jupyter"
    p.io.autoplot.interval = 1

    # Save intermediate .ptyr files (dumps) every 10 iterations
    p.io.autosave = u.Param()
    p.io.autosave.active = True
    p.io.autosave.interval = 10
    p.io.autosave.rfile = 'dumps/%(run)s_%(engine)s_%(iterations)04d.ptyr'

    # Live-plotting during the reconstruction
    p.io.autoplot = u.Param()
    p.io.autoplot.active=True
    p.io.autoplot.threaded = False
    p.io.autoplot.layout = "jupyter"
    p.io.autoplot.interval = 1

    # Define the scan model
    p.scans = u.Param()
    p.scans.scan_00 = u.Param()
    p.scans.scan_00.name = 'Full'

    # Initial illumination (based on simulated optics)
    p.scans.scan_00.illumination = u.Param()
    if parameters["probe_known"]==True:
        p.scans.scan_00.illumination.model = parameters["path_to_probe"]
        step_to_update_probe=parameters["total_steps"]+10
    p.scans.scan_00.illumination.aperture = u.Param()
    if parameters["probe_known"]==False:
        p.scans.scan_00.illumination.aperture.form = "circ"
        step_to_update_probe=0

    # Data loader
    p.scans.scan_00.data = u.Param()
    p.scans.scan_00.data.name = 'Hdf5Loader'

    # Read diffraction data
    p.scans.scan_00.data.intensities = u.Param()
    p.scans.scan_00.data.intensities.file = path_to_data_xy
    p.scans.scan_00.data.intensities.key = "diffinten"
    # p.scans.scan_00.data.intensities.file = path_to_data
    # p.scans.scan_00.data.intensities.key = "diffamp"

    # Read positions data
    p.scans.scan_00.data.positions = u.Param()
    p.scans.scan_00.data.positions.file = path_to_data_xy
    p.scans.scan_00.data.positions.slow_key = "posx_um"        # changed
    p.scans.scan_00.data.positions.slow_multiplier = 1e-6
    p.scans.scan_00.data.positions.fast_key = "posy_um"
    p.scans.scan_00.data.positions.fast_multiplier = 1e-6

    # Read meta data:  
    #lambda_nm = 0.12397975*1e-9 # 0.12397975*1e-9  0.08265323*1e-9
    lambda_nm=f['lambda_nm'][()]*1e-9
    print(lambda_nm)
    energy = C.h*C.c/lambda_nm/C.eV/1e3
    p.scans.scan_00.data.energy = energy

    # Read meta data: detector distance ???
    p.scans.scan_00.data.recorded_distance = u.Param()
    p.scans.scan_00.data.recorded_distance.file = path_to_data_xy
    p.scans.scan_00.data.recorded_distance.key = "z_m"
    p.scans.scan_00.data.recorded_distance.multiplier = 1

    # Read meta data: detector pixelsize
    p.scans.scan_00.data.recorded_psize = u.Param()
    p.scans.scan_00.data.recorded_psize.file = path_to_data_xy
    p.scans.scan_00.data.recorded_psize.key = "ccd_pixel_um"
    p.scans.scan_00.data.recorded_psize.multiplier = 1e-6

    # Define reconstruction engine (using DM)
    p.engines = u.Param()
    p.engines.engine = u.Param()

    
    # p.engines.engine.name = "DM_pycuda"
    # p.engines.engine.numiter = 1000
    # p.engines.engine.numiter_contiguous = 10
    # p.engines.engine.alpha = 0.95
    # p.engines.engine.probe_support = None
    # p.engines.engine.probe_update_start = 0
    

    if method =="EPIE":
        p.engines.engine.name = "EPIE_cupy"
        p.engines.engine.numiter = parameters["total_steps"]
        p.engines.engine.numiter_contiguous = 1
        p.engines.engine.alpha =parameters["a"]  
        p.engines.engine.beta = parameters["b"]
        p.engines.engine.object_norm_is_global = True  
        p.engines.engine.probe_update_start = step_to_update_probe#parameters["total_steps"]+10
    elif method == "RAAR":
        p.engines.engine.fourier_relax_factor=0
        p.engines.engine.subpix_start= parameters["total_steps"]+10
        p.engines.engine.name = method
        p.engines.engine.numiter = parameters["total_steps"]  
        p.engines.engine.numiter_contiguous = 10
        p.engines.engine.beta = parameters["RAAR_beta"]
        p.engines.engine.probe_support = None
        p.engines.engine.probe_update_start =step_to_update_probe #parameters["total_steps"]+10
        
    elif method == "ML_pycuda":
        p.engines.engine.name = method
        p.engines.engine.ML_type = "gaussian"
        p.engines.engine.numiter = parameters["total_steps"] 
        p.engines.engine.numiter_contiguous = 10
        p.engines.engine.reg_del2 = True 
        p.engines.engine.reg_del2_amplitude = 0.01 # 1
        p.engines.engine.scale_precond = True
        p.engines.engine.scale_probe_object = 1.
        p.engines.engine.probe_update_start = parameters["total_steps"]+10
    elif method == "ML":
        p.engines.engine.name = method
        p.engines.engine.ML_type = "poisson"
        p.engines.engine.numiter = parameters["total_steps"] 
        p.engines.engine.numiter_contiguous = 10
        p.engines.engine.reg_del2 = True                      # Whether to use a Gaussian prior (smoothing) regularizer
        p.engines.engine.reg_del2_amplitude = 0.1             # Amplitude of the Gaussian prior if used
        p.engines.engine.scale_precond = True
        p.engines.engine.scale_probe_object = 1.
        p.engines.engine.smooth_gradient = 20.
        p.engines.engine.smooth_gradient_decay = 1/50.
        p.engines.engine.floating_intensities = False
        p.engines.engine.probe_update_start = parameters["total_steps"]+10
        
    elif method == "DM":
        #p.engines.engine.fourier_relax_factor=0
        p.engines.engine.subpix_start= 0#parameters["total_steps"]+10
        p.engines.engine.name = "DM_cupy"
        p.engines.engine.numiter = parameters["total_steps"]  
        p.engines.engine.numiter_contiguous = 10
        p.engines.engine.alpha =parameters["DM_alpha"]
        p.engines.engine.probe_support = None
        p.engines.engine.probe_update_start =step_to_update_probe

    
    P = ptypy.core.Ptycho(p,level=5)
    S = list(P.obj.S.values())[0]
    probe=list(P.probe.S.values())[0]
    probe=probe.data[0].copy()
    obj = S.data[0].copy()
    
    np.save(parameters["tag"]+"_obj.npy",obj)
    np.save(parameters["tag"]+"_probe.npy",probe)
    
    
    
    
    
    
def train_model(parameters,f,probe,model_name="Pty_INR",trained=False,c=1):
    if model_name=="Pty_INR":
        train_Pty_INR(f,parameters,probe,c)
    else:
        train_Iterative(f,parameters,probe,model_name)