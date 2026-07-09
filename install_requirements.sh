#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing Python packages for this project..."
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt

echo "If you are running on Jetson and want GPU-accelerated OCR, install PyTorch with:"
echo "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132"
echo "Required packages installed successfully."
