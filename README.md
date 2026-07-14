# ID Scanner Setup Guide

This is the complete new-unit setup for the GPU-accelerated ID scanner on an
NVIDIA Jetson Orin Nano. The commands use the current user's home directory, so
the installation is not tied to a particular username.

## Before starting

You need:

- A Jetson Orin Nano with JetPack and the Ubuntu desktop installed
- A working camera and internet connection
- A normal desktop user account with terminal access
- Automatic desktop login enabled if the scanner must start unattended

Use the normal desktop user for this setup. Use `sudo` only on commands where it
is shown. Do not run Jetson Containers, `run_jetson.sh`, or the boot-service
installer with `sudo`.

## 1. Install and enable Docker

Run once on a new Jetson:

```bash
sudo apt update
sudo apt install -y docker.io xinput
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
sudo reboot
```

After the reboot, open a terminal and verify Docker works without `sudo`:

```bash
docker info
```

Do not continue until that command succeeds without a socket permission error.

## 2. Install Jetson Containers

Run as the normal desktop user:

```bash
cd "$HOME"
git clone https://github.com/dusty-nv/jetson-containers
sudo mkdir -p /usr/local/bin
bash "$HOME/jetson-containers/install.sh"
```

## 3. Install and test the scanner

```bash
cd "$HOME"
git clone https://github.com/sirtoggle/BarcodeScanner_jetpack72
cd "$HOME/BarcodeScanner_jetpack72"
cp -n scanner.env.example scanner.env
bash run_jetson.sh
```

The first launch builds a reusable local image and downloads the OCR model. It
can take several minutes. Later launches reuse Docker's cached image and OCR
model; do not run `install_requirements.sh` manually.

The launcher mounts the project at `/workspace` and removable media at `/media`.
The removable-media mount uses live propagation so USB sticks inserted after
the scanner starts are visible without recreating the container. The launcher
also supplies the camera, display, and JetPack-compatible CUDA runtime and
starts `test9.py` automatically.

Confirm that:

- The camera opens and fits the screen
- `SCANNER READY` appears
- A test card produces the correct ID, card date, and full name
- A CSV file is created on the USB stick

Press `q` to stop this initial manual test.

## 4. Start automatically after boot

The fullscreen camera needs a logged-in graphical desktop. Enable automatic
desktop login for the scanner's normal user in Ubuntu's user settings. Then run:

```bash
cd "$HOME/BarcodeScanner_jetpack72"
bash install_boot_service.sh
sudo reboot
```

Run `install_boot_service.sh` without `sudo`. Five seconds after automatic
desktop login, it imports the display session and starts the scanner. The kiosk
service retries if Docker, the desktop, or the camera is temporarily unavailable
and restarts the scanner ten seconds after it exits.

Useful service commands:

```bash
systemctl --user status barcode-scanner.service
systemctl --user stop barcode-scanner.service
bash "$HOME/BarcodeScanner_jetpack72/start_scanner_on_login.sh"
journalctl --user -u barcode-scanner.service -f
```

Because the boot service treats the scanner as a kiosk, pressing `q` causes it
to restart. Use `systemctl --user stop barcode-scanner.service` for maintenance.
To remove automatic startup:

```bash
cd "$HOME/BarcodeScanner_jetpack72"
bash install_boot_service.sh --remove
```

## 5. Persistent settings

Settings that must survive reboot belong in:

```text
$HOME/BarcodeScanner_jetpack72/scanner.env
```

The file is created from `scanner.env.example` and is intentionally excluded
from Git, so updates do not overwrite unit-specific settings. Edit it with:

```bash
nano "$HOME/BarcodeScanner_jetpack72/scanner.env"
```

Example:

```bash
ID_EXPECTED_LENGTH=8
ID_PATTERN='12[0-9]{6}'
ID_SCANNER_FULLSCREEN=true
ID_SCANNER_DISABLE_TOUCH=true
```

Logo phrases that must not be mistaken for a person's name are kept in:

```text
$HOME/BarcodeScanner_jetpack72/logo_words.txt
```

Add one logo phrase per line. Blank lines and lines beginning with `#` are
ignored. The included file starts with `Wynn Rewards` and
`Encore Boston Harbor`. Because the file is tracked by Git, you can update it
once, commit it, and pull the same list onto every scanner.

Useful optional settings:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `DETECTION_MAX_WIDTH` | `960` | Lower values reduce CPU load; raise it if small cards are missed. |
| `DISPLAY_MAX_WIDTH` | `960` | Safe fallback width; fullscreen geometry is chosen by the desktop. |
| `OCR_INTERVAL_SECONDS` | `0.18` | Time between ID OCR attempts. |
| `OCR_MIN_CONFIDENCE` | `0.40` | Rejects uncertain ID and date readings. |
| `NAME_MIN_CONFIDENCE` | `0.45` | Rejects uncertain full-name readings. |
| `CONFIRMATION_MATCHES` | `3` | Matching ID readings required before saving. |
| `ID_SCANNER_SAVE_IMAGES` | `true` | Set to `false` to disable card-image retention. |
| `ID_SCANNER_FULLSCREEN` | `true` | Opens the live video as a borderless fullscreen window. |
| `ID_SCANNER_DISABLE_TOUCH` | `true` | Disables matching touchscreens when the scanner starts. Set to `false` to retain touch input. |
| `ID_SCANNER_TOUCH_MATCH` | common touchscreen names | Optional case-insensitive device-name pattern for an unusual monitor. |

After changing `scanner.env` or `logo_words.txt`, restart the service:

```bash
systemctl --user restart barcode-scanner.service
```

Touch input is disabled on the Jetson host at each scanner start. USB mice and
keyboards remain usable. Existing installations only need to install `xinput`
once; setting `ID_SCANNER_DISABLE_TOUCH=false` in `scanner.env` restores touch:

```bash
sudo apt install -y xinput
```

If the service log says that no touchscreen matched, list the device names:

```bash
xinput list --name-only
```

Then put a distinctive part of the touchscreen name in `scanner.env`, for
example `ID_SCANNER_TOUCH_MATCH='ILITEK|Multi-Touch'`, and restart the service.

## 6. Daily USB-stick operation

The scanner automatically locates mounted removable media and rechecks it before
every confirmed scan. Docker mount propagation keeps those changes visible to
the running scanner. The old USB stick can be safely ejected and replaced by a
blank one while the scanner remains open; no application or Jetson restart is
required. Wait for Ubuntu to finish mounting the new stick before presenting
the first card. If a card is presented during the brief interval when no stick
is mounted, the scanner uses a local fallback folder to prevent data loss.

If operators cannot use Ubuntu's eject command, use this physical-swap routine:

1. Remove the card from the camera area.
2. Wait until the display says `SCANNER READY`.
3. Pull out the old USB stick.
4. Insert the blank replacement and wait a few seconds before scanning.

Before returning to `SCANNER READY`, the scanner closes and explicitly flushes
both the CSV and card image, including the USB directory entry. This minimizes
the risk from physical removal. Never pull the stick during a scan, countdown,
or save operation; no software can make removal during an active write safe.

Each CSV row contains four columns in this order:

1. Detected ID
2. Scanner timestamp
3. Printed card date
4. Full name

The date or name is blank when it was not detected confidently. Validated dates
are excluded from ID selection even when OCR reads `/` as `7`. Name OCR runs only
after ID confirmation to preserve live performance. It ignores common labels,
organization terms, single-word logos, and the configured logo names.

## 7. Install updates

Do not run `git pull` from `/workspace` inside the temporary container. Stop the
service and update the host checkout as the normal user:

```bash
systemctl --user stop barcode-scanner.service
cd "$HOME/BarcodeScanner_jetpack72"
git pull --ff-only
bash install_boot_service.sh
bash start_scanner_on_login.sh
```

Docker rebuilds only when dependency files change. Normal Python-code updates
start quickly using the existing image.

## 8. Troubleshooting

### Docker permission denied

The login session has not picked up membership in the `docker` group. Reboot,
then run `docker info` without `sudo`. Do not run the launcher with `sudo`.
Messages such as `groups: cannot find name for group ID ...` inside the container
are not a Docker permission fix.

### View boot-service errors

```bash
journalctl --user -u barcode-scanner.service -n 100 --no-pager
```

### Open a diagnostic container shell

Stop the boot service first, then open the prepared container:

```bash
systemctl --user stop barcode-scanner.service
cd "$HOME/BarcodeScanner_jetpack72"
bash run_jetson.sh bash
```

Inside the container, verify CUDA:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

It must print `True`. Run `exit` to return to the Jetson host.

### Scanner does not open

- Confirm the camera is connected.
- Confirm `docker info` works without `sudo`.
- Check the service log shown above.
- Confirm automatic desktop login completed and the Ubuntu desktop is visible.
