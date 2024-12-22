import numpy as np
import tinycudann as tcnn
from utils.Deep_Models import *
from Parameters import *
from utils.Forward import *
from utils.Simulate_Data_Process import*
from utils.Training_models import *
import h5py
import commentjson as json

model_name = parameters["model_name"]
mode=parameters["mode"]
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
    probe=probe/torch.abs(probe).max()
    h5=SCAN_Iterative_Methods_Process(amplitude_gt=torch.tensor(np.abs(case_obj)),phase_gt=torch.tensor(np.angle(case_obj))
                                   ,overlap_ratio=0.95,probe=probe,parameters=parameters)
    model=parameters["model_name"]
    if model == "ePIE":
        parameters["tag"]="Simulated_ePIE_recovered"
        train_model(parameters,h5,probe,model_name="EPIE",trained=False)
    elif model == "DM":
        parameters["tag"]="Simulated_DM_recovered"
        train_model(parameters,h5,probe,model_name="DM",trained=False)
    elif model == "RAAR":
        parameters["tag"]="Simulated_RAAR_recovered"
        train_model(parameters,h5,probe,model_name="RAAR",trained=False)
    elif model == "Pty_INR":
        parameters["tag"]="Simulated_SCAN_recovered"
        train_model(parameters,h5,probe,model_name="Pty_INR",trained=False)
    else:
        print("no such model, please check again the name!")
else:
    h5= h5py.File(parameters["real_data_source"], 'r')
    model=parameters["model_name"]
    if model == "ePIE":
        parameters["tag"]="Real_Data_ePIE_recovered"
        train_model(parameters,h5,probe,model_name="ePIE",trained=False)
    elif model == "DM":
        parameters["tag"]="Real_Data_DM_recovered"
        train_model(parameters,h5,probe,model_name="DM",trained=False)
    elif model == "RAAR":
        parameters["tag"]="Real_Data_RAAR_recovered"
        train_model(parameters,h5,probe,model_name="RAAR",trained=False)
    elif model == "Pty_INR":
        parameters["tag"]="Real_Data_Pty_INR_recovered"
        train_model(parameters,h5,probe,model_name="Pty_INR",trained=False)
    else:
        print("no such model, please check again the name!")