# Ptychography Using Implicit Neural Representation (INR)

This project implements **ptychographic reconstruction** using **Implicit Neural Representations (INRs)**, with two neural networks representing the object and probe functions. It reconstructs the object's amplitude and phase from diffraction patterns.

## **Project Structure**

├── data/ # Input diffraction patterns

├── ptypy-master/ # PtyPy framework for ptychographic simulations

├── tiny-cuda-nn/ # CUDA-accelerated neural network library

├── utils/ # Utility scripts

├── Main.py # Main script to run the model

├── Parameters.py # Configuration for object network and training

├── config_hash.json # Additional configurations for probe netwrok

├── requirements.txt # Dependencies

## **Installation**
1. Clone the repository:
   git clone https://github.com/TISGroup/PtyINR.git
   cd PtyINR
2. Set up a virtual environment and install dependencies
   pip install -r requirements.txt

## **Usage**
1. Place your diffraction patterns in the data/ folder.
2. Configure training parameters in Parameters.py.
3. Run the reconstruction:   Python Main.py

