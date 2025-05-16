# PtyINR: Ptychographic Implicit Neural Representation

**PtyINR** is a deep learning framework for joint reconstruction of complex-valued **objects** and **illumination probes** in X-ray ptychography. It leverages **implicit neural representations** and **physics-informed optimization** to enable high-quality reconstructions under challenging conditions such as **limited scan overlap** and **short exposure time**. This repository provides the main scripts, utilities, and interactive notebooks to run PtyINR on both **simulated** and **real experimental datasets**.

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/TISGroup/PtyINR.git
   cd PtyINR
2. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
3. **Install the tiny-cuda-nn**:

   ```bash
   cd tiny-cuda-nn/bindings/torch
   pip install .

## Usage
1. **Input data preparation**
2. **Ptychographic reconstruction**
3. **Results evaluation**

## Notes for tuning hyper-parameters
1. **Object neural networks**
2. **Probe neural netowkrs**
3. **Loss function**

## Dependencies
numpy  
scipy  
torchmetrics  
h5py  
scikit-image  
matplotlib  
pytorch_msssim  
scikit-learn  
torchsummary  
commentjson  
gdown  
tiny-cuda-nn  

## Repository Structure

   ```bash

   PtyINR/  
   ├── data/                          # Input datasets  
   ├── result/                        # Output reconstructions  
   ├── tiny-cuda-nn/                  # Our probe neural network backbone, modified to use float precision  
   ├── utils/                         # Utility functions  
   │   ├── Deep_Models.py             # Defines MLP-based architectures for neural representations  
   │   ├── Forward.py                 # Implements the forward ptychographic propagation model  
   │   ├── Simulate_Data_Process.py  # Tools for simulating ptychographic measurements  
   │   ├── Training_models.py        # Training loop and optimization utilities  
   ├── Main.py                        # Main script for training and evaluation  
   ├── Parameters.py                  # Parameter configurations  
   ├── config_hash.json               # JSON config file for probe neural networks  
   ├── interactive_rec_real_data.ipynb        # Notebook for experimental data reconstructions  
   ├── interactive_rec_simulated_data.ipynb   # Notebook for simulated data reconstructions  
   ├── Metrics4Simulated_object.ipynb         # Object metrics  
   ├── Metrics4Simulated_probe.ipynb          # Probe metrics  
   ├── requirements.txt               # Python dependencies  
   └── README.md                      

```
## Acknowledgements

We gratefully acknowledge the contributions of the following open-source projects, which have significantly inspired and supported the development of this work:

- [**Instant Neural Graphics Primitives with a Multiresolution Hash Encoding**](https://github.com/NVlabs/instant-ngp)

- [**Implicit Neural Representations with Periodic Activation Functions (SIREN)**](https://vsitzmann.github.io/siren/)

- [**Ptypy**](https://github.com/ptycho/ptypy/tree/master)

Their contributions to the deep learning and computational imaging communities have been invaluable in the development of PtyINR.
