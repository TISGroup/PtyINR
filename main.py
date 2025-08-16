from parameters import *
from PtyINR.train import *
import matplotlib.pyplot as plt
from PtyINR.data_simulation_and_evaluation import*

def _prepare_simulated_data(params):

    # Load crystal (complex) and probe, crop to target sizes
    crystal = np.load(params["simulate_data_source"])[303:, 303:]  # 241x241
    probe = np.load(params["path_to_probe"])[10:74, 10:74]          # 64x64

    # Compute scan step size from overlap
    step_size = round((1 - params["overlap_ratio"]) * probe.shape[0])

    # Pad case_obj on right and bottom so scan grid divides evenly
    # First pad so (obj - probe) % step == 0
    rem = (crystal.shape[0] - probe.shape[0]) % step_size
    pad_number = (step_size - rem) if rem != 0 else 0
    if pad_number > 0:
        case_obj = np.pad(crystal, ((0, pad_number), (0, pad_number)), mode="constant")
    else:
        case_obj = crystal

    # Ensure number of scan positions per axis is even
    n_steps = (case_obj.shape[0] - probe.shape[0]) / step_size + 1
    if int(n_steps) % 2 != 0:
        case_obj = np.pad(case_obj, ((0, step_size), (0, step_size)), mode="constant")

    print("test object shape: ", case_obj.shape)
    print("overlap ratio: ", params["overlap_ratio"])
    print("step size: ", step_size)

    # Update parameters
    params["obj_size"] = int(case_obj.shape[0])

    # Generate diffraction patterns (writes to files/buffers as implemented in your project)

    amp = torch.tensor(np.abs(case_obj), dtype=torch.float32)
    phs = torch.tensor(np.angle(case_obj), dtype=torch.float32)
    prb = torch.tensor(probe, dtype=torch.float32)

    _ = diffraction_pattern_generate(
        amplitude_gt=amp,
        phase_gt=phs,
        overlap_ratio=params["overlap_ratio"],
        probe=prb,
        parameters=params,
    )


def main():
    # Respect GPU selection before importing torch
    if "CUDA_VISIBLE_DEVICES" in parameters:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(parameters["CUDA_VISIBLE_DEVICES"])

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