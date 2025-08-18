import scipy
from numpy import zeros, newaxis
import random

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

import time

from parameters import *
from PtyINR.tools import *


def _confine(A):
    # Wrap coordinates to centered FFT-like range [-N/2, N/2) per axis
    sh = np.asarray(A.shape)[1:]
    A = A.astype(float)
    m = np.reshape(sh, (len(sh),) + len(sh) * (1,))
    return (A + m // 2.0) % m - m // 2.0


def _translate_to_pix(sh, center):
    # Convert center specifier to pixel coordinates
    sh = np.array(sh)
    if not isinstance(center, str):
        cen = np.asarray(center) % sh
    elif center == 'fftshift':
        cen = sh // 2.0
    elif center == 'geometric':
        cen = sh / 2.0 - 0.5
    elif center == 'fft':
        cen = sh * 0.0
    else:
        raise TypeError('Input %s not understood for center' % str(center))

    return cen


def grids(sh, psize=None, center='geometric', FFTlike=True):
    # Generate coordinate grids with optional physical pixel sizes and centering
    sh = np.asarray(sh)

    cen = _translate_to_pix(sh, center)

    grid = np.indices(sh).astype(float) - np.reshape(cen, (len(sh),) + len(sh) * (1,))

    if FFTlike:
        grid = _confine(grid)  # wrap to FFT convention

    if psize is None:
        return grid
    else:
        psize = np.asarray(psize)
        if psize.size == 1:
            psize = psize * np.ones((len(sh),))
        psize = np.asarray(psize).reshape((len(sh),) + len(sh) * (1,))
        return grid * psize  # scale by physical pixel size


def get_pre_post_fft():
    # Precompute quadratic phase factors for far-field propagation (pre/post FFT multipliers)
    sh = parameters["sh"]
    resolution = parameters["resolution"]
    c_sam = parameters["c_sam"]
    psize = parameters["psize"]
    c_det = parameters["c_det"]
    lz = parameters["lz"]

    [X, Y] = grids(sh, resolution, c_sam)  # sample-plane coordinates
    [V, W] = grids(sh, psize, c_det)  # detector-plane coordinates

    pre_curve = np.exp(
        1j * np.pi * (X ** 2 + Y ** 2) / lz)  # quadratic phase at sample

    pre_fft = pre_curve * np.exp(
        -2.0 * np.pi * 1j * ((X - X[0, 0]) * V[0, 0] +
                             (Y - Y[0, 0]) * W[0, 0]) / lz
    )  # shift term to align sampling between planes

    post_curve = np.exp(
        1j * np.pi * (V ** 2 + W ** 2) / lz)  # quadratic phase at detector

    post_fft = post_curve * np.exp(
        -2.0 * np.pi * 1j * (X[0, 0] * V + Y[0, 0] * W) / lz
    )  # position-dependent phase

    pre_fft = torch.tensor(pre_fft)
    post_fft = torch.tensor(post_fft)
    return pre_fft, post_fft


def forward(object_p, probe, pre_fft, post_fft):
    # Single-view forward model: exit wave -> far-field propagation -> diffraction amplitude
    exit_wave2 = torch.mul(object_p, probe)  # exit wave at sample
    w = post_fft * 1 / parameters["shape_size"] * torch.fft.fft2(
        pre_fft * exit_wave2)  # scaled FFT with pre/post factors
    return torch.abs(w)  # return amplitude at detector