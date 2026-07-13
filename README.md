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
python3 -m pip install -r requirements.txt
```

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

