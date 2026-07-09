#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing Python packages for this project..."
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt

if [ -f /etc/nv_tegra_release ]; then
  echo "Jetson device detected. For true GPU acceleration, use a JetPack-compatible PyTorch runtime instead of the generic pip wheel."
  echo "Recommended path: install the NVIDIA Jetson container stack and run the PyTorch container:"
  echo "  git clone https://github.com/dusty-nv/jetson-containers"
  echo "  bash jetson-containers/install.sh"
  echo "  jetson-containers run \$(autotag l4t-pytorch)"
else
  echo "For GPU acceleration on a desktop or server, install a CUDA-enabled PyTorch build matching your GPU and driver stack."
fi

echo "Required packages installed successfully."
