#!/bin/bash
set -euo pipefail

# The graphical session owns these values. Import them into the user service so
# jetson-containers can safely forward the active desktop into the container.
environment_names=()
for name in DISPLAY XAUTHORITY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS WAYLAND_DISPLAY; do
  if [ -n "${!name:-}" ]; then
    environment_names+=("$name")
  fi
done

if [ "${#environment_names[@]}" -gt 0 ]; then
  systemctl --user import-environment "${environment_names[@]}"
fi

systemctl --user daemon-reload
exec systemctl --user restart barcode-scanner.service
