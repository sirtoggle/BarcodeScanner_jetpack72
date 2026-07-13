#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing Python packages for this project..."

python3 - <<'PY'
import platform
import sys

if sys.version_info < (3, 8):
    raise SystemExit(
        f"ERROR: Python {platform.python_version()} is too old. "
        "Use a Jetson PyTorch container with Python 3.8 or newer."
    )

profile = "Jetson Python 3.8 compatibility" if sys.version_info < (3, 9) else "modern Python"
print(f"Selected dependency profile: {profile} ({platform.machine()})")
PY

install_system_opencv() {
  local use_sudo=0

  if python3 -c "import cv2" >/dev/null 2>&1; then
    return
  fi

  echo "OpenCV is not present in this container; installing Ubuntu's GStreamer-enabled Python package..."

  if [ "$(id -u)" -eq 0 ]; then
    apt-get update
    apt-get install -y python3-opencv
  elif command -v sudo >/dev/null 2>&1; then
    use_sudo=1
    sudo apt-get update
    sudo apt-get install -y python3-opencv
  else
    echo "ERROR: OpenCV is missing and this user cannot install python3-opencv." >&2
    echo "Run this script as root, or build a jetson-containers image containing pytorch and opencv:cuda." >&2
    exit 1
  fi

  # Some /usr/local Python container builds omit Ubuntu's dist-packages path.
  # Add it through a .pth file only when apt installed cv2 but Python cannot see it.
  if ! python3 -c "import cv2" >/dev/null 2>&1 && [ -d /usr/lib/python3/dist-packages ]; then
    if [ "$use_sudo" -eq 1 ]; then
      sudo python3 - <<'PY'
import site
from pathlib import Path

site_packages = Path(site.getsitepackages()[0])
site_packages.mkdir(parents=True, exist_ok=True)
(site_packages / "ubuntu-dist-packages.pth").write_text(
    "/usr/lib/python3/dist-packages\n",
    encoding="utf-8",
)
PY
    else
      python3 - <<'PY'
import site
from pathlib import Path

site_packages = Path(site.getsitepackages()[0])
site_packages.mkdir(parents=True, exist_ok=True)
(site_packages / "ubuntu-dist-packages.pth").write_text(
    "/usr/lib/python3/dist-packages\n",
    encoding="utf-8",
)
PY
    fi
  fi

  if ! python3 -c "import cv2" >/dev/null 2>&1; then
    echo "ERROR: python3-opencv was installed, but this Python runtime still cannot import cv2." >&2
    echo "Use a jetson-containers image built with both pytorch and opencv:cuda." >&2
    exit 1
  fi
}

install_system_opencv

python3 - <<'PY'
import cv2
import numpy
import sys
import torch
import torchvision

print(f"Using Python {sys.version.split()[0]}")
print(f"Using OpenCV {cv2.__version__}")
print(f"Using NumPy {numpy.__version__}")
print(f"Using PyTorch {torch.__version__}")
print(f"Using torchvision {torchvision.__version__}")
PY

# Jetson Containers can set PIP_INDEX_URL to its CUDA wheel cache. These support
# packages are normal architecture wheels and should come from public PyPI; torch
# and torchvision were already supplied and verified by the GPU container.
python3 -m pip install --index-url https://pypi.org/simple -r requirements.txt

# EasyOCR declares generic torch and opencv-python-headless dependencies. Install
# the application itself without those dependencies so pip cannot replace the
# JetPack-compatible CUDA/GStreamer builds already provided by the container.
python3 -m pip install --index-url https://pypi.org/simple --no-deps easyocr==1.7.2

if [ -f /etc/nv_tegra_release ]; then
  echo "Jetson device detected."
else
  echo "Note: /etc/nv_tegra_release was not found; confirm this is the intended Jetson container."
fi

python3 - <<'PY'
import cv2
import easyocr
import numpy
import os
import torch

build_info = cv2.getBuildInformation()
gstreamer_lines = [line.strip() for line in build_info.splitlines() if "GStreamer" in line]
cuda_available = torch.cuda.is_available()
skip_gpu_check = os.getenv("ID_SCANNER_SKIP_GPU_CHECK") == "1"
gstreamer_enabled = any("YES" in line for line in gstreamer_lines)
print(f"EasyOCR {easyocr.__version__} installed successfully")
print(f"NumPy {numpy.__version__} imports successfully with OpenCV {cv2.__version__}")
print(f"CUDA available: {cuda_available}")
print(gstreamer_lines[0] if gstreamer_lines else "GStreamer status was not reported by OpenCV")
if not cuda_available and not skip_gpu_check:
    raise SystemExit("ERROR: CUDA is unavailable. Use a JetPack-compatible PyTorch container before running the scanner.")
if skip_gpu_check:
    print("CUDA runtime check deferred until the built image is started on the Jetson.")
if not gstreamer_enabled:
    print("WARNING: OpenCV does not report GStreamer support. USB cameras can still work, but CSI cameras require a GStreamer-enabled build.")
PY

echo "Required packages installed successfully without replacing JetPack OpenCV or PyTorch."
