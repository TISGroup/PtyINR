# PtyINR: Ptychographic Implicit Neural Representation

**PtyINR** is a deep learning framework for joint reconstruction of complex-valued **objects** and **illumination probes** in X-ray ptychography. It leverages **implicit neural representations** and **physics-informed optimization** to enable high-quality reconstructions under challenging conditions such as **low-dose measurements**, **noisy data**, and **limited overlap**. This repository provides the main scripts, utilities, and interactive notebooks to run PtyINR on both **simulated** and **real experimental datasets**.

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
├── tiny-cuda-nn/                  # Our probe neural network backbone  
├── utils/                         # Utility functions  
├── Main.py                        # Scripts for training PtyINR  
├── Parameters.py                  # Parameter configurations 
├── config_hash.json               # JSON config file for probe nueral networks  
├── interactive_rec_real_data.ipynb        # Notebook for experimental data reconstructions  
├── interactive_rec_simulated_data.ipynb   # Notebook for simulated data reconstructions  
├── Metrics4Simulated_object.ipynb         # Object metrics  
├── Metrics4Simulated_probe.ipynb          # Probe metrics  
├── requirements.txt               # Python dependencies  
├── README.md                      # Project documentation  
