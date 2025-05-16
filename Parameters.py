import numpy as np
import h5py
import os
import torch

parameters={}

##########################################################   Define experiment type (simulated or your own data)    #########################################

### for simulated experiment settings
parameters["mode"]="simulated"
parameters["overlap_ratio"]=0.95                                            # choose between 0.95 0.5 0.3
parameters["probe_known"]=False                                             # if False, PtyINR will also estimate the probe
parameters["path_to_probe"]="data/probe_for_sim.npy"                        # predefine known probes if available
parameters["simulate_data_source"]="data/crystal.npy"                       # only for simulation, the ground truth object
f={}                                                                        # for getting the physical parameters for simulation
f["lambda_nm"]=np.float64(0.1239797467401991)                     # the wavelength of the incoming wave(nm)
f['z_m']=np.float64(0.5)                                          # the distance from the detector to the sample(meters)
f['ccd_pixel_um']=np.float64(55.0)                                # the pixel size of the detector(um)
f['angle']=np.float64(-9.992007221626409*1e-15)                   # the angle of the incoming wave to the sample(rad)

### real data source file
parameters["real_data_source"]="data/real_data.h5"     # current we only support h5 file as standard input with headers "lambda_nm", "z_m", "ccd_pixel_um", "angle", "diff_amp", ""

### result save directory
parameters["save_path"] = "result/"


##########################################################   Model training parameters    #########################################

### those are defined as default, you may change based on your own data
parameters["diffraction_scale"]=1200    # the diffraction pattern will be divided by this value, this is to ensure the loss is not too large for optimization as LR is usually below 1
parameters["first_omega"]=30            # control the the extent of recovered details, raise this value if the data is sufficient and decrease if data is limited
parameters["loss"]="SmoothL1"                 # use SmoothL1 by default. choose between "MSE" "SmoothL1" "L1"
parameters["beta_for_smoothl1"]=1e-3        # control the extent the loss is more like MSE or L1, raise this value if you want to have better details but higher value will become more unstable
parameters["LR"]=8e-5            # for object amplitude       
parameters["LR2"]=8e-5         # for object phase 
parameters["LR3"]=5e-5           # for probe amplitude
parameters["LR4"]=5e-5         # for probe phase
parameters["device"]="cuda"    # by default, we use GPU for training
parameters["regularized_loss_weight"]=2e-1        # to regularize the probe function at the begining of training, adjust it higher if the reconstructed probe diverged
parameters["regularized_steps"]=20                # to control the training steps to regularize the probe, adjust it higher if the reconstructed probe diverged
parameters["show_every"]=50                       # the image will be showed every this step
parameters["image_show"]=True                     # whether to display the current training images
parameters["model_type"]="siren"                  # define the object neural network type, default is SIREN
parameters["tag"]="NA"         # the name to be saved for reconstructions


### for Pty_INR training strategy
parameters["train_method"]="mini_batch"        
parameters["batches"]=3600                        
parameters["total_steps"] = 3000   # total training steos


##########################################################   Do not modify from here, below are for setting parameters    #########################################
if parameters["mode"]=="simulated":
    parameters["amp_shift"]=False
else:
    parameters["amp_shift"]=True

probe=np.load(parameters["path_to_probe"])
probe=torch.tensor(probe)
if parameters["mode"]=="simulated":   # extract the middle 64*64
    total_size = probe.shape[0]
    idx=int((total_size-64)/2)
    probe=probe[idx:-idx,idx:-idx]
else:
    f=h5py.File(parameters["real_data_source"], 'r')


## Probe size                                                      
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