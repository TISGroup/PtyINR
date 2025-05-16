# PtyINR: Ptychographic Implicit Neural Representation

**PtyINR** is a deep learning framework for joint reconstruction of complex-valued **objects** and **illumination probes** in X-ray ptychography. It leverages **implicit neural representations** and **physics-informed optimization** to enable high-quality reconstructions under challenging conditions such as **low-dose measurements**, **noisy data**, and **limited overlap**. This repository provides the main scripts, utilities, and interactive notebooks to run PtyINR on both **simulated** and **real experimental datasets**.

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/PtyINR.git
   cd PtyINR
2. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
3. **Install the tiny-cuda-nn**:

   ```bash
   cd tiny-cuda-nn/bindings/torch
   pip install .

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
PtyINR/
├── data/                          # Input datasets
├── result/                        # Output reconstructions
├── tiny-cuda-nn/                  # Neural field backend
├── utils/                         # Utility functions
├── Main.py                        # Main training/testing script
├── Parameters.py                  # Parameter configurations
├── config_hash.json               # JSON config file
├── interactive_rec_real_data.ipynb        # Notebook for real data
├── interactive_rec_simulated_data.ipynb   # Notebook for simulated data
├── Metrics4Simulated_object.ipynb         # Object metrics
├── Metrics4Simulated_probe.ipynb          # Probe metrics
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
