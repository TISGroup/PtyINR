import scipy
from numpy import zeros, newaxis
import random

import torch
import numpy as np
import torch
from torch import nn
import numpy as np
import skimage
from PIL import Image,ImageChops
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


def _confine(A):
    """\
    Doc TODO.
    """
    sh = np.asarray(A.shape)[1:]
    A = A.astype(float)
    m = np.reshape(sh, (len(sh),) + len(sh) * (1,))
    return (A + m // 2.0) % m - m // 2.0


def _translate_to_pix(sh, center):
    """\
    Take arbitrary input and translate it to a pixel position with respect to sh.
    """
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
    """\
    ``q0,q1,... = grids(sh)``
    returns centered coordinates for a N-dimensional array of shape sh (pixel units)

    ``q0,q1,... = grids(sh,psize)``
    gives the coordinates scaled according to the given pixel size psize.

    ``q0,q1,... = grids(sh,center='fftshift')``
    gives the coordinates shifted according to fftshift convention for the origin

    ``q0,q1,... = grids(sh,psize,center=(c0,c1,c2,...))``
    gives the coordinates according scaled with psize having the origin at (c0,c1,..)


    Parameters
    ----------
    sh : tuple of int
        The shape of the N-dimensional array

    psize : float or tuple of float
        Pixel size in each dimensions

    center : tupel of int
        Tuple of pixel, or use ``center='fftshift'`` for fftshift-like grid
        and ``center='geometric'`` for the matrix center as grid origin

    FFTlike : bool
        If False, grids ar not bound by the interval [-sh//2:sh//2[

    Returns
    -------
    ndarray
        The coordinate grids
    """
    sh = np.asarray(sh)

    cen = _translate_to_pix(sh, center)

    grid = np.indices(sh).astype(float) - np.reshape(cen, (len(sh),) + len(sh) * (1,))

    if FFTlike:
        grid = _confine(grid)

    if psize is None:
        return grid
    else:
        psize = np.asarray(psize)
        if psize.size == 1:
            psize = psize * np.ones((len(sh),))
        psize = np.asarray(psize).reshape((len(sh),) + len(sh) * (1,))
        return grid * psize
    

sh=parameters["sh"]
resolution=parameters["resolution"]
c_sam=parameters["c_sam"]
psize=parameters["psize"]
c_det=parameters["c_det"]
lz=parameters["lz"]
device=parameters["device"]
    
[X, Y] = grids(sh, resolution, c_sam)
[V, W] = grids(sh, psize, c_det)

pre_curve = np.exp(
    1j * np.pi * (X**2 + Y**2) / lz)

pre_fft = pre_curve * np.exp(
    -2.0 * np.pi * 1j * ((X-X[0, 0]) * V[0, 0] +
                         (Y-Y[0, 0]) * W[0, 0]) / lz
)


post_curve = np.exp(
    1j * np.pi * (V**2 + W**2) / lz)


post_fft = post_curve * np.exp(
    -2.0 * np.pi * 1j * (X[0, 0]*V + Y[0, 0]*W) / lz
)

pre_fft=torch.tensor(pre_fft).to(device)
post_fft=torch.tensor(post_fft).to(device)




def forward(object_p,probe):
    exit_wave2=torch.mul(object_p,probe)
    w = post_fft * 1/parameters["shape_size"] * torch.fft.fft2(pre_fft * exit_wave2)
    return torch.abs(w)
