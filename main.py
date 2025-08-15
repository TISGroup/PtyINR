import numpy as np
import tinycudann as tcnn
from parameters import *
from PtyINR.forward import *
from PtyINR.data_simulation_and_evaluation import*
from PtyINR.train import *
import h5py
import commentjson as json

os.environ["CUDA_VISIBLE_DEVICES"]=parameters["CUDA_VISIBLE_DEVICES"]

mode=parameters["mode"]
parameters["image_show"]=False 
if mode=="simulated":
    overlap_ratio= parameters["overlap_ratio"]
    crystal=np.load(parameters["simulate_data_source"])[303:,303:]   #241*241
    probe=np.load(parameters["path_to_probe"])
    probe=probe[10:74,10:74]             #64*64
    probe=torch.tensor(probe)
    step_size=round((1-parameters["overlap_ratio"])*probe.shape[0])
    pad_number=step_size-(crystal.shape[0]-probe.shape[0])%step_size
    
    if pad_number!=step_size:
        pad=nn.ZeroPad2d((0, pad_number, 0, pad_number))
        case_obj=torch.tensor(crystal)
        case_obj=pad(case_obj)
        case_obj=case_obj.numpy()
    else:
        case_obj=crystal
        
    if ((case_obj.shape[0]-probe.shape[0])/step_size+1)%2!=0:
        case_obj=torch.tensor(case_obj)
        pad=nn.ZeroPad2d((0, step_size, 0, step_size))
        case_obj=pad(case_obj)
        case_obj=case_obj.numpy()
    
    print("test object shape: ",case_obj.shape)
    print("overlap ratio: ",parameters["overlap_ratio"])
    parameters["obj_size"]=case_obj.shape[0]
    h5=diffraction_pattern_generate(amplitude_gt=torch.tensor(np.abs(case_obj)),phase_gt=torch.tensor(np.angle(case_obj))
                                   ,overlap_ratio=0.95,probe=probe,parameters=parameters)

    parameters["tag"]="Simulated_PtyINR"
    train_model(parameters)
else:
    parameters["tag"]="Experimental_PtyINR"
    train_model(parameters)
