# Code was modified/commented by Copilot

"""
INSTALLATION/LIBS
NOTE: designed for Google Colab environment
"""

# Force remove everything (to clear out broken .so files)
!pip uninstall -y torch torchvision torchaudio torch-scatter torch-sparse torcheeg

import subprocess
import sys

# Check if CUDA is available, gets version
try:
    nvcc_output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
    if "release 12.1" in nvcc_output or "release 12." in nvcc_output:
        cuda_version = "cu121"
    elif "release 11.8" in nvcc_output or "release 11." in nvcc_output:
        cuda_version = "cu118"
    else:
        cuda_version = "cu118"  # Default to cu118
    print(f"GPU detected, using CUDA version {cuda_version}")
except:
    cuda_version = "cpu"

# Install PyTorch w/ detected version
!pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/{cuda_version}
!pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+{cuda_version}.html
!pip install torcheeg --no-deps  # no-deps to avoid Scipy version conflict

# Other runtime dependencies
!pip install mne pandas xlrd tqdm scikit-learn

# Verify installation
import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")