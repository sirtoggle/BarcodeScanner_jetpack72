#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
AUTOSTART_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
SERVICE_FILE="$USER_UNIT_DIR/barcode-scanner.service"
AUTOSTART_FILE="$AUTOSTART_DIR/barcode-scanner.desktop"

if [ "$(id -u)" -eq 0 ]; then
  echo "ERROR: Run this installer as the normal desktop user, not with sudo." >&2
  exit 1
fi

if [ "${1:-}" = "--remove" ]; then
  systemctl --user stop barcode-scanner.service >/dev/null 2>&1 || true
  rm -f "$SERVICE_FILE" "$AUTOSTART_FILE"
  systemctl --user daemon-reload
  echo "Barcode Scanner boot startup removed."
  exit 0
fi

for command_name in bash docker script systemctl; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: Required command '$command_name' was not found." >&2
    exit 1
  fi
done

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not available to this user." >&2
  echo "Confirm 'docker info' works without sudo before installing startup." >&2
  exit 1
fi

mkdir -p "$USER_UNIT_DIR" "$AUTOSTART_DIR"
if [ ! -f "$SCRIPT_DIR/scanner.env" ]; then
  cp "$SCRIPT_DIR/scanner.env.example" "$SCRIPT_DIR/scanner.env"
  echo "Created $SCRIPT_DIR/scanner.env for persistent scanner settings."
fi

BASH_BIN="$(command -v bash)"
DOCKER_BIN="$(command -v docker)"
SCRIPT_BIN="$(command -v script)"

temporary_service="$(mktemp)"
temporary_autostart="$(mktemp)"
trap 'rm -f "$temporary_service" "$temporary_autostart"' EXIT

cat >"$temporary_service" <<EOF
[Unit]
Description=Barcode Scanner kiosk

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR
Environment=ID_SCANNER_CONTAINER_NAME=barcode-scanner-runtime
Environment=PYTHONUNBUFFERED=1
Environment="PATH=$PATH"
ExecStartPre=-$DOCKER_BIN rm -f barcode-scanner-runtime
ExecStart=$SCRIPT_BIN -q -e -f -c "$BASH_BIN '$SCRIPT_DIR/run_jetson.sh'" /dev/null
ExecStop=-$DOCKER_BIN stop --time 10 barcode-scanner-runtime
Restart=always
RestartSec=10
TimeoutStopSec=20
KillMode=control-group
StandardOutput=journal
StandardError=journal
EOF

cat >"$temporary_autostart" <<EOF
[Desktop Entry]
Type=Application
Name=Barcode Scanner
Comment=Start the Barcode Scanner kiosk after graphical login
Exec=$BASH_BIN "$SCRIPT_DIR/start_scanner_on_login.sh"
Terminal=false
NoDisplay=true
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=5
EOF

install -m 0644 "$temporary_service" "$SERVICE_FILE"
install -m 0644 "$temporary_autostart" "$AUTOSTART_FILE"
systemctl --user daemon-reload

echo "Barcode Scanner boot startup installed."
echo "Enable automatic desktop login for user '$USER', then reboot to test it."
echo "Persistent settings: $SCRIPT_DIR/scanner.env"
echo "Logs: journalctl --user -u barcode-scanner.service -f"
