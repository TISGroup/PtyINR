from parameters import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(parameters["CUDA_VISIBLE_DEVICES"])  # bind training to specific GPU(s)
from PtyINR.train import *  # provides train_model
import matplotlib.pyplot as plt
from PtyINR.data_simulation_and_evaluation import *  # provides diffraction_pattern_generate


def _prepare_simulated_data(params):
    # Load complex object and probe; crop to fixed working sizes
    crystal = np.load(params["simulate_data_source"])[303:, 303:]  # 241x241
    probe = np.load(params["path_to_probe"])[24:-24, 24:-24]  # 64x64

    # Step size from overlap: step ≈ (1 - overlap) * probe_size
    step_size = round((1 - params["overlap_ratio"]) * probe.shape[0])

    # Pad object so scan grid fits exactly
    pad_number = step_size - (241 - probe.shape[0]) % step_size

    if pad_number != step_size:
        pad = nn.ZeroPad2d((0, pad_number, 0, pad_number))  # pad right and bottom
        case_obj = torch.tensor(crystal)
        case_obj = pad(case_obj)
        case_obj = case_obj.numpy()
    else:
        case_obj = crystal

    # Ensure even number of scan positions along each axis, just to ensure it could be further divided
    if ((case_obj.shape[0] - probe.shape[0]) / step_size + 1) % 2 != 0:
        case_obj = torch.tensor(case_obj)
        pad = nn.ZeroPad2d((0, step_size, 0, step_size))
        case_obj = pad(case_obj)
        case_obj = case_obj.numpy()

    print("test object shape: ", case_obj.shape)
    print("overlap ratio: ", params["overlap_ratio"])
    params["obj_size"] = case_obj.shape[0]

    # Convert complex object to amplitude and phase for simulation
    amp = torch.tensor(np.abs(case_obj))
    phs = torch.tensor(np.angle(case_obj))
    prb = torch.tensor(probe)

    _ = diffraction_pattern_generate(
        amplitude_gt=amp,
        phase_gt=phs,
        overlap_ratio=params["overlap_ratio"],
        probe=prb,
        parameters=params,
        noise=params["noise_tag"]
    )


def main():

    mode = parameters["mode"]
    if mode == "simulated":
        _prepare_simulated_data(parameters)
        parameters["tag"] = "Simulated_PtyINR"
    else:
        parameters["tag"] = "Experimental_PtyINR"

    train_model(parameters)  # start training


if __name__ == "__main__":
    main()