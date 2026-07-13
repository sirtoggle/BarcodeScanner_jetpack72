#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="barcode-scanner:jetson"
MODEL_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/barcode-scanner/easyocr"
CONFIG_FILE="${ID_SCANNER_CONFIG_FILE:-$SCRIPT_DIR/scanner.env}"

if [ -f "$CONFIG_FILE" ]; then
  echo "Loading scanner settings from $CONFIG_FILE"
  set -a
  # This local file is owned and maintained by the Jetson operator.
  source "$CONFIG_FILE"
  set +a
fi

container_name_args=()
if [ -n "${ID_SCANNER_CONTAINER_NAME:-}" ]; then
  if [[ ! "$ID_SCANNER_CONTAINER_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: ID_SCANNER_CONTAINER_NAME contains invalid characters." >&2
    exit 1
  fi
  container_name_args=(--name "$ID_SCANNER_CONTAINER_NAME")
fi

for command_name in docker autotag jetson-containers; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: Required command '$command_name' was not found." >&2
    exit 1
  fi
done

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not available to this user." >&2
  echo "Log out and back in after joining the docker group, then run 'docker info'." >&2
  exit 1
fi

cd "$SCRIPT_DIR"
mkdir -p "$MODEL_CACHE"

echo "Preparing the reusable scanner image..."
BASE_IMAGE="$(autotag l4t-pytorch)"
if [ -z "$BASE_IMAGE" ]; then
  echo "ERROR: autotag did not return a JetPack-compatible PyTorch image." >&2
  exit 1
fi

# Docker reuses the completed dependency layer unless the Dockerfile,
# requirements, installer, or base image changed.
docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --file Dockerfile.jetson \
  --tag "$IMAGE_NAME" \
  .

if [ "$#" -eq 0 ]; then
  set -- python3 /workspace/test9.py
fi

echo "Starting the scanner..."
exec jetson-containers run \
  "${container_name_args[@]}" \
  -v "$SCRIPT_DIR:/workspace" \
  -v "/media:/media" \
  -v "$MODEL_CACHE:/root/.EasyOCR" \
  -e CAMERA_SOURCE \
  -e CAMERA_INDEX \
  -e CSI_SENSOR_ID \
  -e CAMERA_WIDTH \
  -e CAMERA_HEIGHT \
  -e CAMERA_FPS \
  -e CAMERA_FOURCC \
  -e CAMERA_FLIP_METHOD \
  -e DETECTION_MAX_WIDTH \
  -e DISPLAY_MAX_WIDTH \
  -e OCR_INTERVAL_SECONDS \
  -e OCR_CANVAS_SIZE \
  -e OCR_MIN_CONFIDENCE \
  -e NAME_MIN_CONFIDENCE \
  -e ID_MIN_LENGTH \
  -e ID_MAX_LENGTH \
  -e ID_EXPECTED_LENGTH \
  -e ID_PATTERN \
  -e CONFIRMATION_MATCHES \
  -e CONFIRMATION_WINDOW \
  -e ID_SCANNER_OUTPUT_DIR \
  -e ID_SCANNER_SAVE_IMAGES \
  -e ID_SCANNER_DISABLE_BLANKING \
  -e ID_SCANNER_FULLSCREEN \
  -e ID_SCANNER_LOGO_WORDS \
  "$IMAGE_NAME" "$@"
