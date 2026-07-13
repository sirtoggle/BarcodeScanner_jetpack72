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
Then install Jetson Containers as your normal user:

```bash
git clone https://github.com/dusty-nv/jetson-containers
sudo mkdir -p /usr/local/bin
bash jetson-containers/install.sh
```

## 3. Build and start the scanner
Run this from the normal Jetson host terminal, not from inside a container:

```bash
cd /home/ebh_admin/BarcodeScanner_jetpack72
bash run_jetson.sh
```

The first launch builds a reusable local image containing all Python and OpenCV
dependencies. Later launches reuse Docker's cached image and do not require
`bash install_requirements.sh`. The script mounts the current project at
`/workspace`, mounts removable media at `/media`, preserves the downloaded OCR
model on the host, and starts `test9.py` automatically.

The image keeps the base container's JetPack-compatible PyTorch and torchvision
builds and installs Ubuntu's GStreamer-enabled OpenCV when needed. Do not install
`opencv-python` or `opencv-python-headless`; those generic packages can replace
the camera stack.

Both the legacy Python 3.8 `l4t-pytorch` image and newer Python images are
supported. The installer automatically selects compatible NumPy, SciPy,
scikit-image, and python-bidi releases for the Python version in the container.

On first run, the OCR model may be downloaded automatically. It is stored under
the Jetson user's cache directory so subsequent temporary containers reuse it.

### Installing updates

Press `q` to stop the scanner. If you are at a container prompt, run `exit` first.
Then update and restart from the normal Jetson host terminal:

```bash
cd /home/ebh_admin/BarcodeScanner_jetpack72
git pull --ff-only
bash run_jetson.sh
```

Do not run `git pull` from `/workspace` inside the container. `/workspace` is the
host checkout mounted into the temporary container, while Git credentials and
repository ownership belong to the normal host user.

To open a diagnostic shell instead of immediately starting the scanner:

```bash
bash run_jetson.sh bash
```

If you want output files written directly to a USB drive, set the output folder first:

```bash
export ID_SCANNER_OUTPUT_DIR="/media/<your-user>/<your-usb-name>"
bash run_jetson.sh
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
bash run_jetson.sh
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
| `DISPLAY_MAX_WIDTH` | `960` | Safe fallback width; fullscreen geometry is chosen by the desktop. |
| `OCR_INTERVAL_SECONDS` | `0.18` | Time between OCR attempts while a card is visible. |
| `OCR_MIN_CONFIDENCE` | `0.40` | Rejects uncertain OCR readings. |
| `CONFIRMATION_MATCHES` | `3` | Matching recent readings required before saving. |
| `ID_SCANNER_SAVE_IMAGES` | `true` | Set to `false` if card images should not be retained. |
| `ID_SCANNER_FULLSCREEN` | `true` | Opens the live video as a borderless fullscreen window. |

## 4. Use the scanner
- Place a card or ID in front of the camera.
- The app will try to detect and read it.
- Press q in the camera window to quit.

## 5. Optional: GPIO support
If you need external hardware control, install Jetson GPIO inside the container:

```bash
sudo apt update
sudo apt install -y python3-dev python3-pip
sudo apt install -y libgpiod-dev
python3 -m pip install Jetson.GPIO
```

## 6. Troubleshooting
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

