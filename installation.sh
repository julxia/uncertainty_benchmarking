#!/bin/bash
# Uncertainty Benchmarking Installation Script

set -e

echo "=== Starting Uncertainty Benchmarking installation ==="

# Create and activate Python virtual environment
echo "Setting up Python virtual environment 'mlc'..."
python3 -m venv mlc

# Setup MLC scripts
echo "Setting up MLC and MLPerf Benchmarking..."
pip install mlc-scripts
mlcr install,python-venv --name=mlperf

export MLC_SCRIPT_EXTRA_CMD="--adr.python.name=mlperf"

# Install CUDA and cuDNN dependencies
echo "Installing NVIDIA CUDA and cuDNN dependencies for Ubuntu 22.04..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cudnn cudnn-cuda-12 unzip

echo "Configuring CUDA paths..."
export CUDNN_PATH=/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Hugging Face Benchmarking
echo "Installing packages for Hugging Face Benchmarking..."
pip install torch transformers datasets diffusers

echo "=== Setup complete! ==="
echo "To activate this environment, run: source mlc/bin/activate"
