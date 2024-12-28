import numpy as np
import h5py
import torch

parameters={}

### Define Experiment Type
parameters["mode"]="simulated"
parameters["model_name"]="ePIE"

### Data Source File
parameters["simulate_data_source"]="data/crystal.npy"                       # only for simulation
f=h5py.File("data/scan_203104.h5", 'r')                                     # for getting the physical parameters for simulation
parameters["real_data_source"]="data/scan_203104.h5"                        # only for real data

### Result save directory
parameters["save_path"] = "result/"
### Simulation setting
parameters["overlap_ratio"]=0.95
parameters["path_to_data"]= "temp/temp3.h5"
parameters["path_to_data_make"]= "temp/temp4.h5"
### Probe
parameters["probe_known"]=False
parameters["path_to_probe"]="data/probe.npy"
probe=np.load(parameters["path_to_probe"])
probe=torch.tensor(probe)
if parameters["mode"]=="simulated":   #extract the middle 64*64
    total_size = probe.shape[0]
    idx=int((total_size-64)/2)
    probe=probe[idx:-idx,idx:-idx]


## PHYSICS PARAMETERS
if parameters["mode"]=="simulated":
    parameters["shape_size"]=64
else:
    parameters["shape_size"]=f['diffamp'][()].shape[-1]
parameters["pixel_size"]=f['lambda_nm'][()]*f['z_m'][()]/f['ccd_pixel_um'][()]/parameters["shape_size"]*1000
parameters["c_sam"]="fftshift"
parameters["c_det"]= np.array([parameters["shape_size"]/2, parameters["shape_size"]/2])
parameters["sh"]=np.array([int(parameters["shape_size"]),int(parameters["shape_size"])])
parameters["resolution"]=np.array([parameters["pixel_size"]*1e-6,parameters["pixel_size"]*1e-6])
parameters["psize"]=np.array(f['ccd_pixel_um'][()]*1e-6,f['ccd_pixel_um'][()]*1e-6)
parameters["lz"]=f['lambda_nm'][()]*f['z_m'][()]*1e-9
parameters["angle"]=f['angle'][()]
parameters["ccd_pixel_um"]=f['ccd_pixel_um'][()]
parameters["z_m"]=f['z_m'][()]
parameters["lambda_nm"]=f['lambda_nm'][()]
### Training Details
parameters["total_steps"] = 30

### for ePIE

parameters["a"]=0.5  
parameters["b"]=0.5  

### for DM
parameters["DM_alpha"]=1


### for RAAR
parameters["RAAR_beta"]=0.99


### for Pty_INR
if parameters["mode"]=="simulated":
    parameters["amp_shift"]=False
else:
    parameters["amp_shift"]=True
parameters["diffraction_scale"]=2000    # the diffraction pattern will be divided by this value
parameters["first_omega"]=60
parameters["loss"]="MSE"
parameters["beta_for_smoothl1"]=1
parameters["LR"]=1e-4            # for object amplitude       
parameters["LR2"]=1e-4         # for object phase 
parameters["LR3"]=5e-5           # for probe amplitude
parameters["LR4"]=5e-5         # for probe phase
parameters["device"]="cuda"
parameters["tag"]="NA"
parameters["regularized_loss_weight"]=2e-1        #1
parameters["regularized_steps"]=20
parameters["show_every"]=50
parameters["image_show"]=True
### for Pty_INR accumulation gradient descent
parameters["batches"]=[9800,9800]