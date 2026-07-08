# ID Scanner Setup Guide

This guide walks you through setting up the ID scanner project on a Jetson device.

## What you need
- A Jetson device such as a Jetson Orin Nano
- A working camera connected to the device
- Internet access
- A terminal window

## 1. Open the terminal
Open a terminal on the Jetson.

## 2. Get the project files
If the project is not already on the Jetson, download it with one of these commands:

If you have GitHub CLI installed:

```bash
gh repo clone sirtoggle/BarcodeScanner_jetpack72
cd BarcodeScanner_jetpack72
```

If you do not have GitHub CLI, use:

```bash
git clone https://github.com/sirtoggle/BarcodeScanner_jetpack72
cd BarcodeScanner_jetpack72
```

If the project is already on the Jetson, go to the folder instead:

```bash
cd /path/to/BarcodeScanner_jetpack72
```

## 3. Create a Python environment (recommended)
It is a good idea to use a virtual environment so the packages stay isolated.

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv
python3 -m venv ~/idscanner-env
source ~/idscanner-env/bin/activate
python -m pip install --upgrade pip
```

## 4. Install the required packages
Run the setup script:

```bash
chmod +x install_requirements.sh
./install_requirements.sh
```

If the script is not available, install the packages directly:

```bash
python -m pip install -r requirements.txt
```

## 5. Start the ID scanner
Run the program:

```bash
python test9.py
```

On first run it will download the model.

If you want the files to go directly to a USB drive, set the output folder before running:

```bash
export ID_SCANNER_OUTPUT_DIR="/media/<your-user>/<your-usb-name>"
python test9.py
```

You can also set it permanently in your shell profile if you want.

## 6. Use the ID scanner
- Place a card or ID in front of the camera.
- The program will try to detect and read it.
- Press `q` in the camera window to quit.

## 7. If something goes wrong
If the program does not start:
- Check that the camera is connected.
- Make sure the virtual environment is active.
- Make sure the packages installed successfully.
- Try installing them again:

```bash
python -m pip install -r requirements.txt
```

## 8. Notes
- This setup is intended for a Jetson environment.
- If PyTorch does not install correctly for your JetPack version, you may need to install it from NVIDIA’s Jetson packages separately.

## 9. Helpful commands
```bash
ls
pwd
python --version
source ~/idscanner-env/bin/activate
```
