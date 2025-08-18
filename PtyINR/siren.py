import scipy
from numpy import zeros, newaxis
import random
import os
import numpy as np
import h5py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

import numpy as np


class SineLayer(nn.Module):
    # A linear layer followed by a sinusoidal activation: sin(omega_0 * (Wx + b))

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0  # frequency scaling factor for sine
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()  # custom initialization as in SIREN

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform init in [-1/in, 1/in]
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                # Hidden/output layers: scaled by 1/omega_0 to keep activations well-conditioned
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))  # sine activation with frequency omega_0


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):  #outermost_linear means whether adding a linear layer at outermost layer
        super().__init__()

        self.net2 = []  # list to hold layers before wrapping in Sequential
        #         #############################################################################
        self.net2.append(SineLayer(in_features, hidden_features,
                                   is_first=True, omega_0=first_omega_0))  # first layer with special init

        for i in range(hidden_layers):
            self.net2.append(SineLayer(hidden_features, hidden_features,
                                       is_first=False, omega_0=hidden_omega_0))  # hidden sine layers

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                # SIREN-recommended init for final linear when no sine activation at the end
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net2.append(final_linear)  # linear output head
        else:
            self.net2.append(SineLayer(hidden_features, out_features,
                                       is_first=False, omega_0=hidden_omega_0))  # sine-activated output

        self.net2 = nn.Sequential(*self.net2)  # build the module chain

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        model_output = self.net2(coords)

        return model_output