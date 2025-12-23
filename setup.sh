#!/bin/bash
set -e

# Setup script for Open-dLLM
# Creates a virtual environment and installs all dependencies

VENV_NAME=".venv_open_dllm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Open-dLLM Setup Script ==="
echo "Working directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"

# Set CUDA environment (required for flash-attn build)
# Auto-detect CUDA installation
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
elif [ -d "/usr/local/cuda-12.9" ]; then
    export CUDA_HOME=/usr/local/cuda-12.9
elif [ -d "/usr/local/cuda-12.1" ]; then
    export CUDA_HOME=/usr/local/cuda-12.1
else
    echo "ERROR: No CUDA installation found in /usr/local/"
    exit 1
fi
export PATH=$CUDA_HOME/bin:$PATH

echo ""
echo "=== Checking CUDA ==="
echo "CUDA_HOME: $CUDA_HOME"
if command -v nvcc &> /dev/null; then
    nvcc -V
else
    echo "WARNING: nvcc not found. flash-attn may fail to build."
fi

echo ""
echo "=== Creating virtual environment: $VENV_NAME ==="
python3 -m venv "$VENV_NAME"

echo ""
echo "=== Activating virtual environment ==="
source "$VENV_NAME/bin/activate"

echo ""
echo "=== Upgrading pip ==="
pip install --upgrade pip

echo ""
echo "=== Installing PyTorch (cu121) ==="
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== Installing build dependencies ==="
pip install wheel packaging ninja setuptools numpy psutil

echo ""
echo "=== Installing flash-attn (with --no-build-isolation) ==="
pip install flash-attn --no-build-isolation

echo ""
echo "=== Installing remaining requirements ==="
pip install -r requirements.txt

echo ""
echo "=== Installation complete! ==="
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); import flash_attn; print(\"flash-attn: OK\")'"


echo ""
echo "=== Installing current package with pip install -e . ==="
pip install -e .

