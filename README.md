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

## 3. Install the required packages
Run the setup script:

```bash
chmod +x install_requirements.sh
./install_requirements.sh
```

This installs the packages listed in `requirements.txt`.

## 4. Start the ID scanner
Run the program:

```bash
python3 test9.py
```

## 5. Use the ID scanner
- Place a card or ID in front of the camera.
- The program will try to detect and read it.
- Press `q` in the camera window to quit.

## 6. If something goes wrong
If the program does not start:
- Check that the camera is connected.
- Make sure the packages installed successfully.
- Try installing them again:

```bash
python3 -m pip install -r requirements.txt
```

## 7. Notes
- This setup is intended for a Jetson environment.
- If PyTorch does not install correctly for your JetPack version, you may need to install it from NVIDIA’s Jetson packages separately.

## 8. Helpful commands
```bash
ls
pwd
python3 --version
```
