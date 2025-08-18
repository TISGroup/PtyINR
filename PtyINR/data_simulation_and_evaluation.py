import scipy
from numpy import zeros, newaxis
from numpy import random as np_random

import torch
import numpy as np
import torch
from torch import nn
import numpy as np
import skimage
from PIL import Image, ImageChops
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Grayscale, Pad
import matplotlib.pyplot as plt
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_msssim import ms_ssim, ssim
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
import h5py
from PtyINR.forward import *
import io
import time
from math import log10, sqrt
from PtyINR.tools import *


def get_step_size(overlap_ratio, probe_size):
    move_ratio = 1 - overlap_ratio  # fraction of probe size to move each step
    step_size = round(probe_size * move_ratio)  # integer step size in pixels
    return step_size


def diffraction_pattern_generate(amplitude_gt, phase_gt, overlap_ratio, probe, parameters
                                 , noise="clean", seed=1):
    # set all RNG seeds for reproducibility
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")  # use GPU 0 if available
    step_size = get_step_size(overlap_ratio, probe.shape[0])
    print("step size: ", step_size)

    # build complex object from amplitude and phase
    actual_object = torch.complex(amplitude_gt * torch.cos(phase_gt), amplitude_gt * torch.sin(phase_gt))
    actual_object = actual_object.to(device)
    probe = probe.to(device)
    resolution = parameters["resolution"]  # pixel size (m) per axis
    scanpts = int((amplitude_gt.shape[0] - probe.shape[0]) / step_size + 1)  # number of scan positions per axis
    detection_size = int(probe.shape[0])  # detector patch size equals probe size
    actual_amp = torch.tensor(
        np.zeros((scanpts * scanpts, detection_size, detection_size)))  # stack of diffraction amplitudes
    points = torch.tensor(np.zeros((2, scanpts * scanpts)))  # scan coordinates in micrometers

    pre_fft, post_fft = get_pre_post_fft()  # cached FFT scaling/twiddle factors
    pre_fft = pre_fft.to(device)
    post_fft = post_fft.to(device)

    for i in range(scanpts):  # vertical
        for j in range(scanpts):  # horizontal
            # simulate diffraction amplitude for a probe-sized crop at (j,i)
            actual_amp[i * scanpts + j] = forward(actual_object[j * step_size:j * step_size + probe.shape[0]
                                                  , i * step_size:i * step_size + probe.shape[1]], probe, pre_fft,
                                                  post_fft)
            # compute real-space scan coordinates in micrometers
            st = step_size * resolution[-1] * 1e6 * (scanpts - 1) / 2
            points[0][i * scanpts + j] = -st + step_size * resolution[-2] * 1e6 * j  # x j  um
            points[1][i * scanpts + j] = -st + step_size * resolution[-1] * 1e6 * i  # y i

    # move simulated amplitudes to CPU NumPy for saving/noise injection
    if device != "cpu":
        actual_amp = actual_amp.detach().cpu().numpy()
    else:
        actual_amp = actual_amp.numpy()

    # prepare noise terms (Gaussian and Poisson) in intensity domain
    gaussian_high = np.random.normal(0, 100, actual_amp.shape)
    diff_amp = actual_amp
    poisson_high = np_random.poisson(lam=((diff_amp / diff_amp.max()) ** 2) * 10, size=diff_amp.shape)
    poisson_high = poisson_high / 10 * (diff_amp.max()) ** 2

    # apply selected noise model; outputs amplitude (sqrt of intensity)
    if noise == "clean":
        actual_amp = actual_amp
    elif noise == "gaussian":
        actual_amp = actual_amp ** 2 + gaussian_high
        actual_amp = actual_amp.clip(min=0)
        actual_amp = np.sqrt(actual_amp)
    elif noise == "poisson":
        actual_amp = np.sqrt(poisson_high)
    elif noise == "combined":
        actual_amp = poisson_high + gaussian_high
        actual_amp = actual_amp.clip(min=0)
        actual_amp = np.sqrt(actual_amp)
    else:
        print("no such noise level!")

    print("the diffraction pattern shape is ", actual_amp.shape)
    # save simulated dataset for training/inference
    with h5py.File("data/generated_diffraction_pattern.h5", 'w') as f1:
        angle = f1.create_dataset("angle", data=parameters["angle"])
        ccd_pixel_um = f1.create_dataset("ccd_pixel_um", data=parameters["ccd_pixel_um"])
        diffamp = f1.create_dataset("diffamp", data=actual_amp)
        lambda_nm = f1.create_dataset("lambda_nm", data=parameters["lambda_nm"])
        points = f1.create_dataset("points", data=points)
        z_m = f1.create_dataset("z_m", data=parameters["z_m"])

        f1.close()

    # return _


def find_global_phase_shift_loop(A, B, num_steps=100000):
    # brute-force search for global phase that best aligns phases of A to reference B
    best_phi = 0
    min_error = float('inf')

    phase_range = np.linspace(-np.pi, np.pi, num_steps)

    for phi in phase_range:
        error = np.mean((np.angle(A * np.exp(phi * 1j)) - B) ** 2)  # least-squares error in phase

        if error < min_error:
            min_error = error
            best_phi = phi

    return best_phi


def psnr_amp(original, compressed):
    # amplitude PSNR assuming values to be within [0,1]
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return [psnr, ssim(original, compressed, data_range=1)]


def psnr_phase(original, compressed):
    # phase PSNR assuming range [-pi, pi]
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = np.pi
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return [psnr, ssim(original, compressed, data_range=2 * np.pi)]


def recon_obj_evaluation(recon, ground_truth):
    # crop/align ground truth and reconstruction to evaluation window
    ground_truth = ground_truth[-241:, -241:]
    ground_truth = ground_truth[30:-30, 30:-30]
    true_amp = np.abs(ground_truth)
    true_phase = np.angle(ground_truth)

    recon = recon[:241, :241]
    amp = np.abs(recon)[30:-30, 30:-30]
    amp = amp * np.median(true_amp) / np.median(amp)  # median scaling for amplitude
    amp = amp.clip(max=1)
    phase_shift = find_global_phase_shift_loop(recon[30:-30, 30:-30], true_phase)
    phase = np.angle(recon * np.exp(phase_shift * 1j))[30:-30, 30:-30]

    # PSNR/SSIM metrics
    psnr_amp_result = psnr_amp(amp, true_amp)
    psnr_phase_result = psnr_phase(phase, true_phase)

    psnr_amp_1 = round(psnr_amp_result[0], 2)
    psnr_amp_2 = round(psnr_amp_result[1], 2)
    psnr_phase_1 = round(psnr_phase_result[0], 2)
    psnr_phase_2 = round(psnr_phase_result[1], 2)
    print(
        f"Reconstructed object amplitude and phase in PSNR/SSIM: {psnr_amp_1}/{psnr_amp_2}  {psnr_phase_1}/{psnr_phase_2}")


def recon_probe_evaluation(recon, ground_truth):
    # normalize probe amplitude and align global phase
    true_amp = np.abs(ground_truth)
    true_amp = true_amp / true_amp.max()
    true_phase = np.angle(ground_truth)

    amp = np.abs(recon)
    amp = amp / amp.max()
    amp = amp * np.median(true_amp) / np.median(amp)
    phase_shift = find_global_phase_shift_loop(recon, true_phase)
    phase = np.angle(recon * np.exp(phase_shift * 1j))

    # PSNR/SSIM metrics
    psnr_amp_result = psnr_amp(amp, true_amp)
    psnr_phase_result = psnr_phase(phase, true_phase)

    psnr_amp_1 = round(psnr_amp_result[0], 2)
    psnr_amp_2 = round(psnr_amp_result[1], 2)
    psnr_phase_1 = round(psnr_phase_result[0], 2)
    psnr_phase_2 = round(psnr_phase_result[1], 2)
    print(
        f"Reconstructed probe amplitude and phase in PSNR/SSIM: {psnr_amp_1}/{psnr_amp_2}  {psnr_phase_1}/{psnr_phase_2}")