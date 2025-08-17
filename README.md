# PtyINR: Ptychographic Implicit Neural Representation 
[![Explore PtyINR in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IYwsGLXPDQR_djQwJ1Xe5aCt1zaGsY5M?usp=sharing)<br>

**PtyINR** is a deep learning framework for joint reconstruction of complex-valued **objects** and **illumination probes** in X-ray ptychography. It leverages **implicit neural representations** and **physics-informed optimization** to enable high-quality reconstructions under challenging conditions such as **limited scan overlap** and **short exposure time**. This repository provides the main scripts, utilities, and interactive notebooks to run PtyINR on both **simulated** and **real experimental datasets**.

<img src="doc/illustrate.jpg" width="100%"/>

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
   cd utils/tiny-cuda-nn/bindings/torch
   pip install .

## Usage

### 1. Input Data Preparation

Your input should be in an `.h5` file format containing the following datasets:

| Header         | Description |
|----------------|-------------|
| **`diffamp`** | 3D array of diffraction amplitudes, **without FFT shift**, with shape: `(N, H, W)`<br>— `N`: number of diffraction patterns<br>— `H`, `W`: height and width of each pattern |
| **`points`**   | 2D array of scan positions with shape: `(2, N)`<br>— First row: x-coordinates<br>— Second row: y-coordinates<br>— Units: micrometers (µm) |
| **`lambda_nm`**| Wavelength of the incoming X-ray (in nanometers) |
| **`ccd_pixel_um`** | Pixel size of the detector (in micrometers) |
| **`z_m`**      | Distance from the sample to the detector (in meters) |

> Ensure that the data is collected (or simulated) in **transmission mode** of X-ray.  
> For simulated data generation, refer to: `utils/data_simulation_and_evaluation.py`

---

### 2. Running Ptychographic Reconstruction

PtyINR supports two workflows:

#### a. Jupyter Notebooks (Recommended for Exploration & Visualization)

- Configure parameters in `parameters.py`
- **`interactive_rec_simulated_data.ipynb`** — For simulated experiments  
- **`interactive_rec_real_data.ipynb`** — For real experimental data

#### b. Python Script (Recommended for Batch or Headless Runs)

- Configure parameters in `parameters.py`
- Run the script from the terminal:

```bash
python main.py
```

> For guidance on how to tune PtyINR's hyperparameters (e.g., learning rates, loss weights), refer to: **`notes_for_parameters.md`**

---

## Repository Structure

   ```bash

   PtyINR/  
   ├── data/                          # Input datasets
   ├── demo/
   │   ├── interactive_rec_real_data.ipynb    # Notebook for reconstructing real data(user data)
   │   ├── interactive_rec_simulated_data.ipynb  # Notebook for reconstructing simulated data
   ├── doc/                           # Documentations on PtyINR  
   │   ├── notes_for_parameters.md    # Guidance on how to tune the hyperparameters in PtyINR  
   ├── PtyINR/                         # Utility functions  
   │   ├── siren.py             # Defines MLP-based architectures for object neural representations  
   │   ├── forward.py                 # Implements the forward ptychographic propagation model  
   │   ├── data_simulation_and_evaluation.py   # Tools for simulating ptychographic measurements and result evaluation
   │   ├── train.py        # Training loop and optimization utilities  
   │   ├── tiny-cuda-nn/             # Our probe neural network backbone, modified to use float precision
   ├── main.py                        # Main script for training  
   ├── parameters.py                  # Parameter configurations  
   ├── requirements.txt               # Python dependencies  
   └── README.md                      

```
## Acknowledgements

We gratefully acknowledge the contributions of the following open-source projects, which have significantly inspired and supported the development of this work:

- [**Instant Neural Graphics Primitives with a Multiresolution Hash Encoding**](https://github.com/NVlabs/instant-ngp)

- [**Implicit Neural Representations with Periodic Activation Functions (SIREN)**](https://github.com/vsitzmann/siren)

- [**PtychoNN: Deep learning of ptychographic imaging**](https://github.com/mcherukara/PtychoNN/tree/master)

- [**PtyPy - Ptychography Reconstruction for Python**](https://github.com/ptycho/ptypy/tree/master)

Their contributions to the deep learning and computational imaging communities have been invaluable in the development of PtyINR.
