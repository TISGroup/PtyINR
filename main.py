from parameters import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(parameters["CUDA_VISIBLE_DEVICES"])
from PtyINR.train import *
import matplotlib.pyplot as plt
from PtyINR.data_simulation_and_evaluation import*

def _prepare_simulated_data(params):

    # Load crystal (complex) and probe, crop to target sizes
    crystal = np.load(params["simulate_data_source"])[303:, 303:]  # 241x241
    probe = np.load(params["path_to_probe"])[10:74, 10:74]          # 64x64

    # Compute scan step size from overlap
    step_size=round((1-parameters["overlap_ratio"])*probe.shape[0])
    
    pad_number=step_size-(241-probe.shape[0])%step_size
    
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

    print("test object shape: ", case_obj.shape)
    print("overlap ratio: ", params["overlap_ratio"])
    print("step size: ", step_size)

    # Update parameters
    params["obj_size"] = int(case_obj.shape[0])

    # Generate diffraction patterns (writes to files/buffers as implemented in your project)

    amp = torch.tensor(np.abs(case_obj))
    phs = torch.tensor(np.angle(case_obj))
    prb = torch.tensor(probe)

    _ = diffraction_pattern_generate(
        amplitude_gt=amp,
        phase_gt=phs,
        overlap_ratio=params["overlap_ratio"],
        probe=prb,
        parameters=params,
        noise = parameters["noise_tag"]
    )


def main():
    # Respect GPU selection before importing torch
    # if "CUDA_VISIBLE_DEVICES" in parameters:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(parameters["CUDA_VISIBLE_DEVICES"])

    # Defer heavy imports until after environment is set

    # Use spawn for safety with CUDA/DDP

    # Silence inline image display
    parameters["image_show"] = False

    mode = parameters["mode"]
    if mode == "simulated":
        _prepare_simulated_data(parameters)
        parameters["tag"] = "Simulated_PtyINR"
    else:
        parameters["tag"] = "Experimental_PtyINR"

    # Kick off training (this function may use mp.spawn internally)
    train_model(parameters)


if __name__ == "__main__":
    main()