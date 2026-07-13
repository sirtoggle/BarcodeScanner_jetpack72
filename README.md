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
```

Log out of the Jetson desktop completely and log back in (or reboot) so the
`docker` group membership takes effect. Then verify Docker works without `sudo`:

```bash
docker info
```

Do not continue until that command succeeds without a socket permission error.
Then install and start Jetson Containers as your normal user:

```bash
git clone https://github.com/dusty-nv/jetson-containers
sudo mkdir -p /usr/local/bin
bash jetson-containers/install.sh
cd /home/ebh_admin/BarcodeScanner_jetpack72
jetson-containers run -v "$PWD:/workspace" -v "/media:/media" $(autotag l4t-pytorch)
```

## 3. Install the project dependencies
After the container starts on the Jetson device, run:

```bash
cd /workspace
bash install_requirements.sh
```

The launch command explicitly mounts the project at `/workspace` and the Jetson's
removable-media directory at `/media`, so the scanner can access both its code
and the daily USB stick.

The installer keeps the container's JetPack-compatible PyTorch and torchvision
builds. If the container does not include `cv2`, it installs Ubuntu's
GStreamer-enabled `python3-opencv` package. Do not install `opencv-python` or
`opencv-python-headless` with pip; those generic packages can replace the camera
stack. OpenCV performs the lightweight, downscaled card detection on the CPU,
while EasyOCR runs its inference on the Jetson GPU.

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

If `jetson-containers run` reports permission denied for `/var/run/docker.sock`,
the current login has not picked up its `docker` group membership. Log out and
back in or reboot, run `docker info` without `sudo`, and only then retry Jetson
Containers. Avoid running the helper with `sudo`; messages such as `groups:
cannot find name for group ID ...` come from host group IDs that have no matching
name inside the container and are not a fix for Docker socket access.

