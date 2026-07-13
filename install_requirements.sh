#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing Python packages for this project..."

python3 - <<'PY'
import cv2
import torch
import torchvision

print(f"Using OpenCV {cv2.__version__}")
print(f"Using PyTorch {torch.__version__}")
print(f"Using torchvision {torchvision.__version__}")
PY

python3 -m pip install -r requirements.txt

# EasyOCR declares generic torch and opencv-python-headless dependencies. Install
# the application itself without those dependencies so pip cannot replace the
# JetPack-compatible CUDA/GStreamer builds already provided by the container.
python3 -m pip install --no-deps easyocr==1.7.2

if [ -f /etc/nv_tegra_release ]; then
  echo "Jetson device detected."
else
  echo "Note: /etc/nv_tegra_release was not found; confirm this is the intended Jetson container."
fi

python3 - <<'PY'
import cv2
import easyocr
import torch

build_info = cv2.getBuildInformation()
gstreamer_lines = [line.strip() for line in build_info.splitlines() if "GStreamer" in line]
cuda_available = torch.cuda.is_available()
print(f"EasyOCR {easyocr.__version__} installed successfully")
print(f"CUDA available: {cuda_available}")
print(gstreamer_lines[0] if gstreamer_lines else "GStreamer status was not reported by OpenCV")
if not cuda_available:
    raise SystemExit("ERROR: CUDA is unavailable. Use a JetPack-compatible PyTorch container before running the scanner.")
PY

echo "Required packages installed successfully without replacing JetPack OpenCV or PyTorch."
