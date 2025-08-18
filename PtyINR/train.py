import os
import time
import json

# Scientific computing
import numpy as np
from numpy import zeros, newaxis, inf
import scipy
import h5py

# PyTorch and related libraries
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, TensorDataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Scikit-learn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt

# PtyINR related libraries
from PtyINR.siren import *
from PtyINR.forward import *
from PtyINR.tools import *
import tinycudann as tcnn

config = {
    "encoding": {
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 15,
        "base_resolution": 16,
        "per_level_scale": 1.5
    },
    "network": {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2
    }
}


class PtychographyDataset(Dataset):
    def __init__(self, coordinates_x, coordinates_y, diffraction_patterns, rank):

        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.coordinates_x = torch.tensor(coordinates_x,
                                          dtype=torch.int32)   # Convert coordinates to tensor
        self.coordinates_y = torch.tensor(coordinates_y, dtype=torch.int32)
        self.diffraction_patterns = torch.tensor(diffraction_patterns, dtype=torch.float32).to(
            device)  # Convert patterns to tensor

    def __len__(self):
        # Number of scan positions
        return len(self.coordinates_x)

    def __getitem__(self, idx):
        # Return indices and corresponding pattern
        return self.coordinates_x[idx], self.coordinates_y[idx], self.diffraction_patterns[idx]


def _downsample_for_tb(img_hw: torch.Tensor, max_side=512) -> torch.Tensor:
    # Area downsample for TB visualization to cap max side length
    H, W = img_hw.shape[-2], img_hw.shape[-1]
    scale = max(H, W) / max_side
    if scale > 1.0:
        nh, nw = int(round(H / scale)), int(round(W / scale))
        img_hw = F.interpolate(img_hw[None, None, ...], size=(nh, nw), mode="area")[0, 0]
    return img_hw


def _to_tb_image_single_channel(img_hw: torch.Tensor) -> torch.Tensor:
    # Normalize to [0,1] and add channel dim (CHW) for TB
    img_hw = img_hw - img_hw.min()
    denom = img_hw.max().clamp_min(1e-12)
    img_hw = img_hw / denom
    return img_hw[None, ...]  # 1xHxW


def train_Pty_INR_SGD(rank, world_size, parameters):

    # load the diffraction pattern source
    f = h5py.File(parameters["real_data_source"], 'r')
    ### --- DDP SETUP ---
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # Ensure this port is free

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # TensorBoard writer (rank 0 only)
    writer = None
    if rank == 0:
        logdir = os.path.join(parameters["save_path"], "tb")
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=logdir, max_queue=10, flush_secs=20)

    ### --- Data Loading ---
    # Load measured amplitudes and scan coordinates
    ratio = parameters["regularized_loss_weight"]
    regularized_steps = parameters["regularized_steps"]
    show_every = parameters["show_every"]

    actual_amp = f['diffamp'][()]
    if parameters["amp_shift"]:
        actual_amp = np.fft.fftshift(actual_amp, axes=(-2, -1))
    actual_amp = actual_amp / parameters["diffraction_scale"]
    frames_count = actual_amp.shape[0]
    print(f"[GPU {rank}] diffraction patterns shape {actual_amp.shape}", flush=True)

    # Convert physical coords to pixel indices and compute object size
    scan_size = int(np.sqrt(f['points'][()].shape[1]))
    pixel_size = parameters["pixel_size"]

    x_coord = f['points'][()][0].reshape(scan_size, scan_size)
    y_coord = f['points'][()][1].reshape(scan_size, scan_size)

    x_axis = np.round((x_coord - x_coord[0, 0]) / pixel_size).reshape(-1)
    y_axis = np.round((y_coord - y_coord[0, 0]) / pixel_size).reshape(-1)

    x_axis -= x_axis.min()
    y_axis -= y_axis.min()

    obj_size = int(max(x_axis.max(), y_axis.max())) + actual_amp.shape[1]

    # Batch sizes per rank
    global_batch = int(parameters["batches"])
    per_rank_batch = max(1, global_batch // world_size)
    if rank == 0:
        eff_global = per_rank_batch * world_size
        print(f"[DDP] Requested global batch={global_batch} -> per-rank batch={per_rank_batch} "
              f"(effective global={eff_global})", flush=True)

    dataset = PtychographyDataset(x_axis, y_axis, actual_amp, rank)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=per_rank_batch, sampler=sampler, drop_last=False)

    ### --- Models ---
    pred_scale, obj_grid, obj_net_amp, obj_net_phase = get_model(parameters, obj_size, config)
    obj_net_amp = obj_net_amp.to(device)
    obj_net_phase = obj_net_phase.to(device)
    obj_grid = obj_grid.to(device)

    obj_net_amp = DDP(obj_net_amp, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    obj_net_phase = DDP(obj_net_phase, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    model_probe_amp = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                                    encoding_config=config["encoding"],
                                                    network_config=config["network"]).to(device)
    model_probe_phase = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=1,
                                                      encoding_config=config["encoding"],
                                                      network_config=config["network"]).to(device)

    model_probe_amp = DDP(model_probe_amp, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    model_probe_phase = DDP(model_probe_phase, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    probe_grid = grid_return([actual_amp.shape[1], actual_amp.shape[1]]).to(device)
    probe_shape = actual_amp.shape[1]
    if parameters["probe_known"] == True:
        probe = np.load("path_to_probe")
        probe = torch.tensor(probe).to(device)

    ### --- Optimizers ---
    optim = torch.optim.Adam(obj_net_amp.parameters(), lr=parameters["LR"])
    optim2 = torch.optim.Adam(obj_net_phase.parameters(), lr=parameters["LR2"])
    optim3 = torch.optim.Adam(model_probe_amp.parameters(), lr=parameters["LR3"])
    optim4 = torch.optim.Adam(model_probe_phase.parameters(), lr=parameters["LR4"])
    criterion = get_loss(parameters, rank)

    ### --- Training Loop ---
    min_loss = float('inf')
    best_state = None
    total_steps = parameters["total_steps"]
    regularized_steps = parameters["regularized_steps"]
    ratio = parameters["regularized_loss_weight"]
    pre_fft, post_fft = get_pre_post_fft()
    pre_fft = pre_fft.to(device)
    post_fft = post_fft.to(device)

    dx = torch.arange(probe_shape).view(1, probe_shape, 1)
    dy = torch.arange(probe_shape).view(1, 1, probe_shape)

    start_time = time.time()  # timing
    for step in range(total_steps):
        dataloader.sampler.set_epoch(step)
        accumlated_loss = 0

        for x_batch, y_batch, measured in dataloader:
            # Predict object amplitude/phase and form complex object
            predict_amp = obj_net_amp(obj_grid) * pred_scale
            obj_amp = torch.abs(predict_amp).view(obj_size, obj_size)
            obj_phase = obj_net_phase(obj_grid).view(obj_size, obj_size) * pred_scale
            object_act = torch.complex(obj_amp * torch.cos(obj_phase), obj_amp * torch.sin(obj_phase))

            # Probe estimation (if unknown)
            if not parameters["probe_known"]:
                probe_amp = torch.abs(model_probe_amp(probe_grid)).view(probe_shape, probe_shape) * 1e4
                probe_phase = model_probe_phase(probe_grid).view(probe_shape, probe_shape) * 1e4
                probe = torch.complex(probe_amp * torch.cos(probe_phase), probe_amp * torch.sin(probe_phase))
                probe = probe / torch.abs(probe).max()
                probe = recenter_probe(probe)

            # Extract object patches at scan positions
            x_indices = x_batch
            y_indices = y_batch
            x_limits = x_indices + probe.shape[0]
            y_limits = y_indices + probe.shape[1]

            batch_obj = torch.stack([
                object_act[x.item():x_l.item(), y.item():y_l.item()]
                for x, x_l, y, y_l in zip(x_indices, x_limits, y_indices, y_limits)
            ])

            # Forward model and loss
            pre_amp_batch = forward(batch_obj, probe, pre_fft, post_fft)
            loss1 = criterion(pre_amp_batch, measured)
            loss2 = torch.abs(probe).mean()
            loss = loss1 + (ratio if step < regularized_steps else 0) * loss2

            accumlated_loss += loss.item() * len(batch_obj) / frames_count
            loss.backward()

            # Optimizer steps
            optim.step()
            optim2.step()
            optim3.step()
            optim4.step()

            optim.zero_grad()
            optim2.zero_grad()
            optim3.zero_grad()
            optim4.zero_grad()

        # Average loss across ranks
        loss_tensor = torch.tensor([accumlated_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_accum_loss = (loss_tensor.item() / world_size)

        # Save best recon on rank 0
        if rank == 0 and global_accum_loss < min_loss:
            min_loss = global_accum_loss
            np.save(parameters["save_path"] + parameters["tag"] + "_obj" + ".npy", object_act.cpu().detach().numpy())
            np.save(parameters["save_path"] + parameters["tag"] + "_probe" + ".npy", probe.cpu().detach().numpy())

        if rank == 0:
            writer.add_scalar("train/accumulated_loss", global_accum_loss, step)

            if step % show_every == 0:
                # Save image previews
                amp = torch.abs(object_act)
                phase = torch.angle(object_act)
                probe_amp_img = torch.abs(probe)
                probe_phase_img = torch.angle(probe)

                plt.imsave(os.path.join(parameters["save_path"], parameters["tag"] + "_object_amp.jpg"),
                           amp.cpu().detach().numpy(), cmap="grey")
                plt.imsave(os.path.join(parameters["save_path"], parameters["tag"] + "_object_phase.jpg"),
                           phase.cpu().detach().numpy(), cmap="magma_r")
                plt.imsave(os.path.join(parameters["save_path"], parameters["tag"] + "_probe_amp.jpg"),
                           probe_amp_img.cpu().detach().numpy(), cmap="grey")
                plt.imsave(os.path.join(parameters["save_path"], parameters["tag"] + "_probe_phase.jpg"),
                           probe_phase_img.cpu().detach().numpy(), cmap="magma")

                # TensorBoard images
                amp = _downsample_for_tb(amp)
                phase = _downsample_for_tb(phase)
                probe_amp_img = _downsample_for_tb(probe_amp_img)
                probe_phase_img = _downsample_for_tb(probe_phase_img)

                amp_tb = _to_tb_image_single_channel(amp).detach().float().cpu()
                phase01 = (phase + torch.pi) / (2 * torch.pi)
                phase_tb = _to_tb_image_single_channel(phase01).detach().float().cpu()

                writer.add_image("object/amp", amp_tb, step, dataformats="CHW")
                writer.add_image("object/phase", phase_tb, step, dataformats="CHW")

                probe_amp_tb = _to_tb_image_single_channel(probe_amp_img).detach().float().cpu()
                probe_phase01 = (probe_phase_img + torch.pi) / (2 * torch.pi)
                probe_phase_tb = _to_tb_image_single_channel(probe_phase01).detach().float().cpu()
                writer.add_image("probe/amp", probe_amp_tb, step, dataformats="CHW")
                writer.add_image("probe/phase", probe_phase_tb, step, dataformats="CHW")

                print(f"[GPU {rank}] Step {step}, global accumulated loss: {global_accum_loss * 1e5:.6f}", flush=True)

    end_time = time.time()
    elapsed_time = end_time - start_time

    if rank == 0:
        print(f"\n Training completed in {elapsed_time:.2f} seconds "
              f"({elapsed_time / 60:.2f} minutes).")

    dist.destroy_process_group()


def train_model(parameters):
    # Launcher: spawn one process per GPU
    print("Using stocastic gradient descent(SGD)!\n")
    print("Please use tensor board to see the intermediate images.")
    print("The tensor board files are stored in the folder: ", os.path.join(parameters["save_path"], "tb"))
    print("The images of latest reconstructions can be found at : ", parameters["save_path"], "\n")
    world_size = torch.cuda.device_count()

    mp.spawn(train_Pty_INR_SGD,
             args=(world_size, parameters),
             nprocs=world_size,
             join=True)