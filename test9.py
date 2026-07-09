from __future__ import annotations

import os
import subprocess
import sys
import importlib.util
from pathlib import Path

# OpenCV's Qt-based GUI helpers can emit a font warning on Linux/Jetson if the
# expected font directory is missing. Create a compatible fallback before cv2 is imported.
def configure_opencv_qt_font_path() -> None:
    if os.name != "posix":
        return

    for candidate in ("/usr/share/fonts", "/usr/share/fonts/truetype", "/usr/share/fonts/opentype"):
        if os.path.isdir(candidate):
            os.environ.setdefault("QT_QPA_FONTDIR", candidate)
            break

    spec = importlib.util.find_spec("cv2")
    if spec is None or not spec.origin:
        return

    cv2_package_dir = Path(spec.origin).resolve().parent
    qt_font_dir = cv2_package_dir / "qt" / "fonts"
    qt_font_dir.mkdir(parents=True, exist_ok=True)


configure_opencv_qt_font_path()

import cv2
import numpy as np
import easyocr
import csv
import time
import re
from datetime import datetime
from typing import Any, List, Optional, Tuple

try:
    import torch
except Exception:
    torch = None

def is_gpu_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


if is_gpu_available():
    # Let cuDNN pick the fastest kernels for fixed input sizes.
    torch.backends.cudnn.benchmark = True


cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, os.cpu_count() or 1))


reader = easyocr.Reader(['en'], gpu=is_gpu_available())

# Camera defaults tuned for Jetson Orin Nano with USB 4K cameras.
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "usb").strip().lower()
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CSI_SENSOR_ID = int(os.getenv("CSI_SENSOR_ID", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1920"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "1080"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))
CAMERA_FOURCC = os.getenv("CAMERA_FOURCC", "MJPG").strip().upper()
CAMERA_FLIP_METHOD = int(os.getenv("CAMERA_FLIP_METHOD", "0"))

# OCR cadence can be adjusted without editing code.
OCR_INTERVAL_SECONDS = float(os.getenv("OCR_INTERVAL_SECONDS", "0.2"))

# Set this to your mounted USB folder if you want files written there directly.
# Example on Jetson: /media/admin/USB_DRIVE
OUTPUT_DIR = os.getenv("ID_SCANNER_OUTPUT_DIR", "").strip()

# -----------------------------
# CARD DETECTION (ROBUST)
# -----------------------------
def find_card(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    frame_area = frame.shape[0] * frame.shape[1]

    def evaluate_contours(contours, min_area_ratio=0.04):
        best_box = None
        best_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < frame_area * min_area_ratio:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) not in (4, 5, 6):
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            if aspect < 0.5 or aspect > 2.8:
                continue

            if area > best_area:
                best_area = area
                best_box = (x, y, w, h)

        return best_box

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE helps with uneven lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge-based detection for cards with strong borders.
    edges = cv2.Canny(blur, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = evaluate_contours(contours)

    if best_box is not None:
        return best_box

    # Threshold-based detection for darker cards such as black cards.
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = evaluate_contours(contours, min_area_ratio=0.03)

    if best_box is not None:
        return best_box

    # Color-based detection for red cards.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 60, 40), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 60, 40), (179, 255, 255))
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.medianBlur(red_mask, 5)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return evaluate_contours(contours, min_area_ratio=0.02)


# -----------------------------
# OCR ID EXTRACTION
# -----------------------------
def extract_id(results: List[Tuple[Any, str, Any]]) -> Optional[str]:
    text = " ".join([r[1] for r in results])

    numbers = re.findall(r"\d+", text)

    if not numbers:
        return None

    # If OCR finds several numbers, use the longest one.
    # This often gives the card number instead of small extra digits.
    candidate = max(numbers, key=len)

    # Ignore very short numbers because they are usually OCR mistakes.
    if len(candidate) < 4:
        return None

    return candidate


# -----------------------------
# SAVE FUNCTIONS
# -----------------------------
def save_scan(id_number: str, timestamp: str) -> None:
    date_str = datetime.now().strftime("%m-%d-%Y")
    path = find_usb_path()
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"scans{date_str}.csv")

    try:
        with open(filename, "a", newline="") as f:
            csv.writer(f).writerow([id_number, timestamp])
        print(f"✔ SAVED → {id_number} | {timestamp}")
    except PermissionError as exc:
        fallback_dir = os.path.join(os.getcwd(), "id_scanner_output")
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_file = os.path.join(fallback_dir, f"scans{date_str}.csv")
        with open(fallback_file, "a", newline="") as f:
            csv.writer(f).writerow([id_number, timestamp])
        print(f"⚠ Permission denied for {path}; used fallback path {fallback_dir}: {exc}")


def save_image(roi: np.ndarray, timestamp: str) -> None:
    path = find_usb_path()
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{timestamp}.jpg")

    try:
        cv2.imwrite(filename, roi)
        print(f"📸 SAVED IMAGE → {filename}")
    except Exception as exc:
        fallback_dir = os.path.join(os.getcwd(), "id_scanner_output")
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_file = os.path.join(fallback_dir, f"{timestamp}.jpg")
        cv2.imwrite(fallback_file, roi)
        print(f"⚠ Image save fallback used: {fallback_file} ({exc})")

def find_usb_path() -> str:
    if OUTPUT_DIR:
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            test_file = os.path.join(OUTPUT_DIR, ".write_test")
            with open(test_file, "a"):
                pass
            os.remove(test_file)
            print(f"Using configured output directory: {OUTPUT_DIR}")
            return OUTPUT_DIR
        except Exception as exc:
            print(f"Configured output directory is not writable: {OUTPUT_DIR} ({exc})")

    def is_writable_dir(path: str) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, ".write_test")
            with open(test_file, "a"):
                pass
            os.remove(test_file)
            return True
        except Exception:
            return False

    def looks_like_usb_path(path: str) -> bool:
        name = os.path.basename(os.path.normpath(path)).lower()
        return any(token in name for token in ("usb", "drive", "flash", "sd", "removable", "disk"))

    excluded_paths = {
        os.path.realpath(os.path.expanduser("~")),
        os.path.realpath(os.getcwd()),
    }

    candidates: List[str] = []

    for base in ["/media", "/run/media", "/mnt"]:
        if not os.path.exists(base):
            continue

        for root, dirs, _ in os.walk(base):
            real_root = os.path.realpath(root)
            if real_root in excluded_paths:
                dirs[:] = []
                continue

            if real_root != os.path.realpath(base) and is_writable_dir(real_root):
                if os.path.ismount(real_root) or looks_like_usb_path(real_root):
                    candidates.append(real_root)

            dirs[:] = [
                d for d in dirs
                if os.path.realpath(os.path.join(root, d)) not in excluded_paths
            ]

    if candidates:
        selected = sorted(
            set(candidates),
            key=lambda path: (
                0 if os.path.ismount(path) else 1,
                0 if looks_like_usb_path(path) else 1,
                len(path.split(os.sep)),
                path.lower(),
            ),
        )[0]
        print(f"Auto-selected USB output directory: {selected}")
        return selected

    fallback_dir = os.path.join(os.getcwd(), "id_scanner_output")
    os.makedirs(fallback_dir, exist_ok=True)
    return fallback_dir


def draw_centered_overlay(
    frame: np.ndarray,
    text: str,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 1.6,
    thickness: int = 4,
    y_ratio: float = 0.5,
) -> None:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(20, (w - text_size[0]) // 2)
    y = max(text_size[1] + 20, int(h * y_ratio))
    pad = 20
    cv2.rectangle(frame, (x - pad, y - text_size[1] - pad), (x + text_size[0] + pad, y + pad), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def configure_display_window() -> None:
    try:
        cv2.namedWindow("ID Scanner", cv2.WINDOW_NORMAL)
    except Exception:
        pass


def apply_fullscreen_window() -> None:
    # Preferred method: ask OpenCV to set the window fullscreen.
    try:
        cv2.setWindowProperty("ID Scanner", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return
    except Exception:
        pass

    # Fallback: resize the window to the display size (works on Windows/Linux)
    try:
        # Try to get the native screen size via tkinter (usually available).
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()

        cv2.resizeWindow("ID Scanner", int(screen_w), int(screen_h))
        cv2.moveWindow("ID Scanner", 0, 0)
        return
    except Exception:
        pass

    # Last resort: try common X11/window utilities for Linux.
    for command in (
        ["wmctrl", "-r", "ID Scanner", "-b", "add,fullscreen"],
        ["wmctrl", "-r", "ID Scanner", "-e", "0,0,0,1024,600"],
        ["xdotool", "search", "--name", "ID Scanner", "windowactivate", "windowmove", "0", "0", "windowsize", "100%", "100%"],
    ):
        try:
            subprocess.run(command, check=False, capture_output=True, text=True, timeout=3)
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue


def launch_kiosk_mode() -> None:
    try:
        if os.environ.get("DISPLAY"):
            subprocess.Popen(["xset", "s", "off"])
            subprocess.Popen(["xset", "-dpms"])
            subprocess.Popen(["xset", "s", "noblank"])
            subprocess.Popen(["gsettings", "set", "org.gnome.desktop.session", "idle-delay", "0"])
    except Exception:
        pass


def create_camera_capture(camera_index: int = 0) -> cv2.VideoCapture:
    # On Jetson/Linux, V4L2 is the first thing to try for normal USB cameras.
    if CAMERA_SOURCE in ("auto", "usb"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if cap.isOpened():
            if len(CAMERA_FOURCC) == 4:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()

    # Jetson boards often use a CSI camera, so we try a GStreamer pipeline too.
    if CAMERA_SOURCE in ("auto", "csi"):
        gst_pipeline = (
            f"nvarguscamerasrc sensor-id={CSI_SENSOR_ID} ! "
            f"video/x-raw(memory:NVMM), width={CAMERA_WIDTH}, height={CAMERA_HEIGHT}, "
            f"format=NV12, framerate={CAMERA_FPS}/1 ! "
            f"nvvidconv flip-method={CAMERA_FLIP_METHOD} ! "
            "video/x-raw, format=BGRx ! videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(camera_index)
                        
def main() -> None:
    cap = create_camera_capture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Unable to open camera. Check camera index and backend support.")

    launch_kiosk_mode()
    configure_display_window()
    time.sleep(0.2)
    apply_fullscreen_window()

    # CPU-side card detection and overlay rendering run here.
    # OCR uses the Jetson GPU path when PyTorch/CUDA is available.

    # Remember when each card ID was last saved so the same card is not scanned twice too quickly.
    scanned_ids = {}

    # Wait this long before allowing the same ID to be saved again.
    cooldown_seconds = 10

    recent_ids = []
    buffer_size = 8

    stable_id = None
    stable_start_time = None

    last_ocr_time = 0
    ocr_interval = max(0.05, OCR_INTERVAL_SECONDS)
    
    # Countdown tracking variables
    displayed_countdown = None
    last_confirmed_count = 0

    paused = False
    pause_until = 0
    ready_popup_until = time.time() + 2.0
    
    # Count how many frames in a row the card is missing.
    card_lost_frames = 0
    # Only clear the scan buffer if the card has been missing for several frames.
    lost_threshold = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        if display.shape[1] > 1280:
            scale = 1280 / display.shape[1]
            display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Pause briefly after a successful scan so the user can remove the card.

        if not paused and not getattr(main, "fullscreen_applied", False):
            apply_fullscreen_window()
            main.fullscreen_applied = True

        if paused:
            draw_centered_overlay(
                display,
                "REMOVE CARD",
                color=(0, 0, 255),
                bg_color=(0, 0, 0),
                font_scale=2.6,
                thickness=7,
                y_ratio=0.5,
            )

            if time.time() > pause_until:
                paused = False
                ready_popup_until = time.time() + 2.0

            cv2.imshow("ID Scanner", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Look for the card in the current video frame.

        card = find_card(frame)

        if card:
            card_lost_frames = 0  # Reset counter when card is detected
            
            x, y, w, h = card
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi = frame[y:y+h, x:x+w]

            # Run OCR only every short interval so we do not process every single frame.

            if time.time() - last_ocr_time > ocr_interval:

                # EasyOCR runs here; on Jetson this can use the GPU when the runtime is configured for it.
                results = reader.readtext(roi)
                id_number = extract_id(results)

                print("Detected:", id_number)

                if id_number:
                    recent_ids.append(id_number)

                    if len(recent_ids) > buffer_size:
                        recent_ids.pop(0)

                    # Count how often each number appears in the recent frames.

                    counts = {}
                    for i in recent_ids:
                        counts[i] = counts.get(i, 0) + 1

                    candidate = max(counts, key=counts.get)
                    count = counts[candidate]

                    # Update countdown display: count down from 3 to 1
                    # Only update when count increases to show new number
                    if count > last_confirmed_count and count <= 3:
                        displayed_countdown = 4 - count  # Convert: 1→3, 2→2, 3→1
                        last_confirmed_count = count

                    # The ID must be seen several times before we trust it.

                    if len(candidate) >= 6 and count >= 3:

                        if stable_id != candidate:
                            stable_id = candidate
                            stable_start_time = time.time()

                        # Make sure the same ID stays visible for a short moment.
                        if time.time() - stable_start_time > 0.3:

                            now = time.time()
                            last_seen = scanned_ids.get(candidate)

                            # Use a cooldown so the same card is not saved again immediately.

                            if last_seen is None or (now - last_seen > cooldown_seconds):

                                scanned_ids[candidate] = now

                                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                                save_scan(candidate, timestamp)
                                save_image(roi, timestamp)

                                paused = True
                                pause_until = time.time() + 2.5
                                ready_popup_until = time.time() + 0.5

                            # Clear the temporary tracking data after we save or skip the card.

                            recent_ids.clear()
                            stable_id = None
                            stable_start_time = None
                            displayed_countdown = None
                            last_confirmed_count = 0

                last_ocr_time = time.time()

        else:
            # No card was found in this frame.
            card_lost_frames += 1
            
            # Wait a few frames before clearing the buffer.
            # This helps avoid false resets when detection briefly fails.
            if card_lost_frames >= lost_threshold:
                recent_ids.clear()
                stable_id = None
                stable_start_time = None
                displayed_countdown = None
                last_confirmed_count = 0

        if not paused and time.time() < ready_popup_until:
            draw_centered_overlay(
                display,
                "SCANNER READY",
                color=(0, 255, 0),
                bg_color=(0, 0, 0),
                font_scale=1.6,
                thickness=5,
                y_ratio=0.2,
            )
        
        # Display countdown if one is active
        if not paused and displayed_countdown is not None and card_lost_frames == 0:
            draw_centered_overlay(
                display,
                str(displayed_countdown),
                color=(0, 255, 0),
                bg_color=(0, 0, 0),
                font_scale=3.0,
                thickness=8,
                y_ratio=0.5,
            )

        cv2.imshow("ID Scanner", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()