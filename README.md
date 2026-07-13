# ID Scanner Setup Guide

This guide is for running the scanner with true GPU acceleration on a Jetson device using a JetPack-compatible PyTorch runtime.

## Requirements
- Jetson device such as Jetson Orin Nano
- Working camera
- Internet access
- Terminal access

## 1. Clone the project
```bash
git clone https://github.com/sirtoggle/BarcodeScanner_jetpack72
cd BarcodeScanner_jetpack72
```

## 2. Run the Jetson GPU environment
Use NVIDIA's Jetson container stack so PyTorch runs with the correct JetPack/L4T support:

```bash
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker $USER
sudo systemctl enable docker
sudo systemctl start docker

git clone https://github.com/dusty-nv/jetson-containers
sudo mkdir -p /usr/local/bin
bash jetson-containers/install.sh
jetson-containers run $(autotag l4t-pytorch)
```

## 3. Install the project dependencies
After the container starts on the Jetson device, run:

```bash
cd /workspace/BarcodeScanner_jetpack72
bash install_requirements.sh
```

The installer deliberately keeps the container's JetPack-compatible OpenCV,
PyTorch, and torchvision builds. Do not install `opencv-python` or
`opencv-python-headless` into this container; those generic packages can replace
the accelerated camera stack.

## 4. Start the scanner
```bash
python3 test9.py
```

On first run, the OCR model may be downloaded automatically.

If you want output files written directly to a USB drive, set the output folder first:

```bash
export ID_SCANNER_OUTPUT_DIR="/media/<your-user>/<your-usb-name>"
python3 test9.py
```

The output mount is rechecked before every confirmed scan. You can safely unmount
the old stick and insert a blank replacement while the scanner remains running;
the next scan will use the newly mounted stick even if its label and mount path
changed. Avoid presenting a card during the brief interval when no stick is
mounted, because the scanner will use its local fallback folder to prevent data
loss.

For the most reliable ID selection, configure the exact number of digits printed
on your cards. For example, for an eight-digit ID:

```bash
export ID_EXPECTED_LENGTH=8
python3 test9.py
```

If IDs also have a known prefix or format, you can require it with a full regular
expression. This example accepts eight-digit IDs beginning with `12`:

```bash
export ID_PATTERN='12[0-9]{6}'
```

Useful optional tuning settings:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `DETECTION_MAX_WIDTH` | `960` | Lower values reduce CPU load; raise it if small cards are missed. |
| `OCR_INTERVAL_SECONDS` | `0.18` | Time between OCR attempts while a card is visible. |
| `OCR_MIN_CONFIDENCE` | `0.40` | Rejects uncertain OCR readings. |
| `CONFIRMATION_MATCHES` | `3` | Matching recent readings required before saving. |
| `ID_SCANNER_SAVE_IMAGES` | `true` | Set to `false` if card images should not be retained. |

## 5. Use the scanner
- Place a card or ID in front of the camera.
- The app will try to detect and read it.
- Press q in the camera window to quit.

## 6. Optional: GPIO support
If you need external hardware control, install Jetson GPIO inside the container:

```bash
sudo apt update
sudo apt install -y python3-dev python3-pip
sudo apt install -y libgpiod-dev
python3 -m pip install Jetson.GPIO
```

## 7. Troubleshooting
If the app does not start on the Jetson device:
- Check that the camera is connected.
- Confirm you are running inside the Jetson GPU container on that device.
- Verify that PyTorch can see the GPU:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

If that prints False, the container is not using a JetPack-compatible PyTorch runtime for your device.

