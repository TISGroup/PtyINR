import numpy as np
import h5py
import os
import torch

parameters={}

##########################################################   Important: Choose reconstruction type    ########################################################

parameters["mode"]="simulated"                                              # choose "simulated" or "real", choosing "real" means using your own data

##########################################################   Define experiment type (simulated or your own data)    #########################################

### for simulated experiment settings
parameters["overlap_ratio"]=0.95     # choose between 0.95 0.5 0.3, for simplicity of calculation, we used illumination overlap ratio (1-step_size/probe_array_size) here, corresponding to 40%, -540%, -800% actual probe overlap ratios (1-step_size/probe_size)
parameters["probe_known"]=False                                             # if False, PtyINR will also estimate the probe
parameters["path_to_probe"]="data/probe_for_sim.npy"                        # predefine known probes if available
parameters["simulate_data_source"]="data/tungsten_sample.npy"                       # only for simulation, the ground truth object
parameters["noise_tag"] = "clean"                               #choose between clean, gaussian, poisson, combined
f={}                                                                        # for getting the physical parameters for simulation
f["lambda_nm"]=np.float64(0.1239797467401991)                     # the wavelength of the incoming wave(nm)
f['z_m']=np.float64(0.5)                                          # the distance from the detector to the sample(meters)
f['ccd_pixel_um']=np.float64(55.0)                                # the pixel size of the detector(um)
f['angle']=np.float64(-9.992007221626409*1e-15)                   # the angle of the incoming wave to the sample(rad)

### real data source file
parameters["real_data_source"]="data/generated_diffraction_pattern.h5"     # current we only support h5 file as standard input with headers "lambda_nm", "z_m", "ccd_pixel_um", "diffamp", "points"

### result save directory
parameters["save_path"] = "result/"            # please set a folder to save the reconstruction results, preferably a leaf folder of current address 

##########################################################   Model training parameters    #########################################

### those are defined as default, you may change based on your own data
parameters["diffraction_scale"]=600    # the diffraction pattern will be divided by this value, this is to ensure the loss is not too large for optimization as LR is usually below 1
parameters["first_omega"]=30            # control the the extent of recovered details, raise this value if the data is sufficient and decrease if data is limited
parameters["loss"]="SmoothL1"                 # use SmoothL1 by default. choose between "MSE" "SmoothL1" "L1"
parameters["beta_for_smoothl1"]=1e-2        # control the extent the loss is more like MSE or L1, raise this value if you want to have better details but higher value will become more unstable
parameters["LR"]=2e-5            # for object amplitude       
parameters["LR2"]=2e-5         # for object phase 
parameters["LR3"]=1e-5           # for probe amplitude
parameters["LR4"]=1e-5         # for probe phase
parameters["regularized_loss_weight"]=1e-2        # to regularize the probe function at the begining of training, adjust it higher if the reconstructed probe diverged
parameters["regularized_steps"]=50                # to control the training steps to regularize the probe, adjust it higher if the reconstructed probe diverged
parameters["show_every"]=100                       # the image will be showed every this step
parameters["image_show"]=False                     # whether to display the intermediate training images
parameters["model_type"]="siren"                  # define the object neural network type, default is SIREN
parameters["tag"]="NA"         # the name to be saved for reconstructions

### for Pty_INR training strategy
parameters["batches"]=3600                       # batch size 
parameters["total_steps"] = 6000   # total training steps
parameters["CUDA_VISIBLE_DEVICES"]="0"            # choose specific GPU(s) for training

##########################################################   Do not modify from here, below are for setting physical parameters    #########################################
if parameters["mode"]=="simulated":
    parameters["amp_shift"]=False
else:
    parameters["amp_shift"]=True

probe=np.load(parameters["path_to_probe"])
amp1=np.rot90(np.abs(probe),k=2)
phase1=np.rot90(-np.angle(probe),k=2)
probe=amp1*np.exp(1j*phase1)
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

if not os.path.isdir(parameters["save_path"]):
    raise FileNotFoundError(f"Defined reconstruction save path does not exist! Please check again in parameters.py")

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
