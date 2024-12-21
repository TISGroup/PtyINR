import numpy as np
import h5py
import torch

parameters={}

### Define Experiment Type
parameters["mode"]="real"
parameters["model_name"]="Pty_INR"

### Data Source File
parameters["simulate_data_source"]="data/crystal.npy"                       # only for simulation
parameters["real_data_source"]="scan_203104.h5"
f=h5py.File("scan_203104.h5", 'r')
parameters["overlap_ratio"]=0.95
parameters["path_to_data"]= "temp3.h5"
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
parameters["total_steps"] = 1000

### for ePIE

parameters["a"]=1  
parameters["b"]=1  

### for DM
parameters["DM_alpha"]=1


### for RAAR
parameters["RAAR_beta"]=0.99


### for Pty_INR
parameters["diffraction_scale"]=2000    # the diffraction pattern will be divided by this value
if parameters["mode"]=="simulated":
    parameters["amp_shift"]=False
    parameters["diffraction_scale"]=1    # the diffraction pattern will be divided by this value
else:
    parameters["amp_shift"]=True
    parameters["diffraction_scale"]=2000    # the diffraction pattern will be divided by this value
parameters["first_omega"]=30
parameters["LR"]=3e-4
parameters["LR2"]=1.5e-4
parameters["device"]="cuda"
parameters["tag"]="NA"
parameters["regularized_loss_weight"]=2e-1        #1