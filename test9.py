from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from typing import Any, Optional, Tuple

from scanner_core import (
    ConsensusTracker,
    compile_id_pattern,
    extract_card_date,
    extract_full_name,
    extract_id,
    is_gpu_available,
    path_is_on_mount,
    scale_box,
)
from scanner_storage import fallback_output_dir, write_image_file, write_scan_row

# OpenCV's Qt-based GUI helpers can emit a font warning on Linux/Jetson if the
# expected font directory is missing. Point Qt at system fonts before cv2 is imported.
def configure_opencv_qt_font_path() -> None:
    if os.name != "posix":
        return

    for candidate in ("/usr/share/fonts", "/usr/share/fonts/truetype", "/usr/share/fonts/opentype"):
        if os.path.isdir(candidate):
            os.environ.setdefault("QT_QPA_FONTDIR", candidate)
            break


configure_opencv_qt_font_path()

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

GPU_AVAILABLE = is_gpu_available(torch)

if GPU_AVAILABLE:
    # Let cuDNN pick the fastest kernels for fixed input sizes.
    torch.backends.cudnn.benchmark = True


cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, min(4, os.cpu_count() or 1)))

if GPU_AVAILABLE:
    print("INFO: PyTorch GPU acceleration is available.")
elif torch is not None:
    print("WARNING: PyTorch CUDA is unavailable. For true GPU acceleration on Jetson Orin, use a JetPack-compatible PyTorch runtime such as the NVIDIA jetson-containers image.")
else:
    print("WARNING: torch is not installed. EasyOCR will run in CPU mode.")


def env_int(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    raw_value = os.getenv(name)
    try:
        value = int(raw_value) if raw_value is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def env_float(name: str, default: float, *, minimum: Optional[float] = None) -> float:
    raw_value = os.getenv(name)
    try:
        value = float(raw_value) if raw_value is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw_value!r}")

# Camera defaults tuned for Jetson Orin Nano with USB 4K cameras.
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "usb").strip().lower()
CAMERA_INDEX = env_int("CAMERA_INDEX", 0, minimum=0)
CSI_SENSOR_ID = env_int("CSI_SENSOR_ID", 0, minimum=0)
CAMERA_WIDTH = env_int("CAMERA_WIDTH", 1920, minimum=320)
CAMERA_HEIGHT = env_int("CAMERA_HEIGHT", 1080, minimum=240)
CAMERA_FPS = env_int("CAMERA_FPS", 30, minimum=1)
CAMERA_FOURCC = os.getenv("CAMERA_FOURCC", "MJPG").strip().upper()
CAMERA_FLIP_METHOD = env_int("CAMERA_FLIP_METHOD", 0, minimum=0)

# Card detection runs on a smaller CPU-side frame; OCR still receives the
# original-resolution card region.
DETECTION_MAX_WIDTH = env_int("DETECTION_MAX_WIDTH", 960, minimum=320)
DISPLAY_MAX_WIDTH = env_int("DISPLAY_MAX_WIDTH", 960, minimum=320)

# OCR cadence can be adjusted without editing code.
OCR_INTERVAL_SECONDS = env_float("OCR_INTERVAL_SECONDS", 0.18, minimum=0.05)
OCR_CANVAS_SIZE = env_int("OCR_CANVAS_SIZE", 1280, minimum=320)
OCR_MIN_CONFIDENCE = env_float("OCR_MIN_CONFIDENCE", 0.40, minimum=0.0)
NAME_MIN_CONFIDENCE = env_float("NAME_MIN_CONFIDENCE", 0.45, minimum=0.0)
DEFAULT_LOGO_WORDS = ("Wynn Rewards", "Encore Boston Harbor")
CUSTOM_LOGO_WORDS = tuple(
    value.strip()
    for value in os.getenv("ID_SCANNER_LOGO_WORDS", "").replace(";", ",").split(",")
    if value.strip()
)
LOGO_WORDS = DEFAULT_LOGO_WORDS + CUSTOM_LOGO_WORDS
ID_MIN_LENGTH = env_int("ID_MIN_LENGTH", 6, minimum=1)
ID_MAX_LENGTH = env_int("ID_MAX_LENGTH", 20, minimum=ID_MIN_LENGTH)
ID_EXPECTED_LENGTH = env_int("ID_EXPECTED_LENGTH", 0, minimum=0) or None
ID_PATTERN = compile_id_pattern(os.getenv("ID_PATTERN", ""))
CONFIRMATION_MATCHES = env_int("CONFIRMATION_MATCHES", 3, minimum=1)
CONFIRMATION_WINDOW = env_int(
    "CONFIRMATION_WINDOW",
    max(5, CONFIRMATION_MATCHES),
    minimum=CONFIRMATION_MATCHES,
)

# Set this to your mounted USB folder if you want files written there directly.
# Example on Jetson: /media/admin/USB_DRIVE
OUTPUT_DIR = os.getenv("ID_SCANNER_OUTPUT_DIR", "").strip()
SAVE_IMAGES = env_bool("ID_SCANNER_SAVE_IMAGES", True)
DISABLE_DISPLAY_BLANKING = env_bool("ID_SCANNER_DISABLE_BLANKING", True)
FULLSCREEN_DISPLAY = env_bool("ID_SCANNER_FULLSCREEN", True)
WINDOW_NAME = "ID Scanner"

if ID_EXPECTED_LENGTH is not None and not ID_MIN_LENGTH <= ID_EXPECTED_LENGTH <= ID_MAX_LENGTH:
    raise ValueError("ID_EXPECTED_LENGTH must be between ID_MIN_LENGTH and ID_MAX_LENGTH")


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_KERNEL_5 = np.ones((5, 5), np.uint8)
_KERNEL_7 = np.ones((7, 7), np.uint8)

# -----------------------------
# CARD DETECTION (ROBUST)
# -----------------------------
def find_card(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    frame_area = frame.shape[0] * frame.shape[1]

    def evaluate_contours(contours, min_area_ratio=0.04, max_area_ratio=0.90):
        best_box = None
        best_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < frame_area * min_area_ratio or area > frame_area * max_area_ratio:
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
    gray = _CLAHE.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge-based detection for cards with strong borders.
    edges = cv2.Canny(blur, 30, 100)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _KERNEL_5)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = evaluate_contours(contours)

    if best_box is not None:
        return best_box

    # Threshold-based detection for darker cards such as black cards.
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, _KERNEL_7)
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
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, _KERNEL_7)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return evaluate_contours(contours, min_area_ratio=0.02)


# -----------------------------
# SAVE FUNCTIONS
# -----------------------------
def save_scan(
    id_number: str,
    timestamp: str,
    card_date: str,
    full_name: str,
    output_dir: str,
) -> str:
    try:
        write_scan_row(output_dir, id_number, timestamp, card_date, full_name)
        print(
            f"SAVED: {id_number} | {timestamp} | "
            f"card date: {card_date or 'not detected'} | "
            f"name: {full_name or 'not detected'}"
        )
        return output_dir
    except OSError as exc:
        fallback_dir = fallback_output_dir(os.getcwd())
        if os.path.realpath(output_dir) == os.path.realpath(fallback_dir):
            raise RuntimeError(f"Unable to save scan data to {output_dir}: {exc}") from exc

        filename = write_scan_row(fallback_dir, id_number, timestamp, card_date, full_name)
        print(f"WARNING: Output failed at {output_dir}; saved CSV to {filename}: {exc}")
        return fallback_dir


def save_image(roi: np.ndarray, timestamp: str, output_dir: str) -> str:
    if not SAVE_IMAGES:
        return output_dir

    try:
        filename = write_image_file(output_dir, roi, timestamp, cv2.imwrite)
        print(f"SAVED IMAGE: {filename}")
        return output_dir
    except OSError as exc:
        fallback_dir = fallback_output_dir(os.getcwd())
        if os.path.realpath(output_dir) == os.path.realpath(fallback_dir):
            raise RuntimeError(f"Unable to save image to {output_dir}: {exc}") from exc

        fallback_file = write_image_file(fallback_dir, roi, timestamp, cv2.imwrite)
        print(f"WARNING: Image output failed at {output_dir}; saved to {fallback_file}: {exc}")
        return fallback_dir


def verify_writable_dir(path: str) -> bool:
    # Never create a missing removable-media path: doing so after an unplug could
    # silently redirect scans onto the internal filesystem.
    if not os.path.isdir(path):
        return False

    test_file = os.path.join(path, f".id_scanner_write_test_{os.getpid()}")
    try:
        with open(test_file, "w", encoding="utf-8"):
            pass
        os.remove(test_file)
        return True
    except OSError:
        try:
            if os.path.exists(test_file):
                os.remove(test_file)
        except OSError:
            pass
        return False


def mounted_output_candidates(*, include_all_mnt: bool = False) -> list[str]:
    candidates: set[str] = set()

    def is_plausible_removable_mount(path: str, base: str) -> bool:
        if base != "/mnt":
            return True
        if include_all_mnt:
            return True
        name = os.path.basename(os.path.normpath(path)).lower()
        return any(token in name for token in ("usb", "drive", "flash", "sd", "removable", "disk"))

    for base in ("/media", "/run/media", "/mnt"):
        if not os.path.isdir(base):
            continue

        try:
            first_level = [entry.path for entry in os.scandir(base) if entry.is_dir(follow_symlinks=False)]
        except OSError:
            continue

        for path in first_level:
            if os.path.ismount(path) and is_plausible_removable_mount(path, base):
                candidates.add(os.path.realpath(path))

            # Desktop Linux normally mounts removable media at /media/user/drive.
            try:
                second_level = [
                    entry.path
                    for entry in os.scandir(path)
                    if entry.is_dir(follow_symlinks=False)
                ]
            except OSError:
                continue

            for nested_path in second_level:
                if os.path.ismount(nested_path) and is_plausible_removable_mount(nested_path, base):
                    candidates.add(os.path.realpath(nested_path))

    return sorted(candidates, key=lambda path: (len(path.split(os.sep)), path.lower()))


def find_usb_path() -> str:
    candidates = mounted_output_candidates()

    if OUTPUT_DIR:
        all_mounts = mounted_output_candidates(include_all_mnt=True)
        if path_is_on_mount(OUTPUT_DIR, all_mounts) and verify_writable_dir(OUTPUT_DIR):
            print(f"Using configured output directory: {OUTPUT_DIR}")
            return OUTPUT_DIR
        print(f"Configured USB directory is not currently mounted and writable: {OUTPUT_DIR}")

    for candidate in candidates:
        if verify_writable_dir(candidate):
            print(f"Auto-selected mounted output directory: {candidate}")
            return candidate

    fallback_dir = fallback_output_dir(os.getcwd())
    print(f"No writable removable mount found; using fallback: {fallback_dir}")
    return fallback_dir


def refresh_output_dir(current_output_dir: str) -> str:
    fallback_dir = os.path.join(os.getcwd(), "id_scanner_output")
    mounts = mounted_output_candidates(include_all_mnt=True)

    if (
        os.path.realpath(current_output_dir) != os.path.realpath(fallback_dir)
        and path_is_on_mount(current_output_dir, mounts)
        and verify_writable_dir(current_output_dir)
    ):
        return current_output_dir

    refreshed_output_dir = find_usb_path()
    if os.path.realpath(refreshed_output_dir) != os.path.realpath(current_output_dir):
        print(f"Output directory changed to: {refreshed_output_dir}")
    return refreshed_output_dir


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


def resize_to_max_width(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    width = frame.shape[1]
    if width <= max_width:
        return frame, 1.0

    scale = max_width / width
    resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def create_ocr_reader() -> Any:
    # Import lazily so tests and utility imports never initialize or download a model.
    import easyocr

    return easyocr.Reader(["en"], gpu=GPU_AVAILABLE)


def configure_display_window() -> None:
    try:
        flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_GUI_NORMAL"):
            flags |= cv2.WINDOW_GUI_NORMAL
        cv2.namedWindow(WINDOW_NAME, flags)
    except Exception:
        pass


def apply_fullscreen_window() -> bool:
    if not FULLSCREEN_DISPLAY:
        return True

    # OpenCV/Qt on Jetson only honors fullscreen after imshow() has mapped the
    # window.  Process pending GUI events before asking the window manager.
    try:
        cv2.waitKey(1)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(1)
        fullscreen_state = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        if fullscreen_state >= 0.5:
            return True
    except Exception:
        pass

    # Ask the desktop compositor to choose the monitor geometry. Avoid resizing
    # to a reported pixel size here: display scaling on Jetson can make that
    # value larger than the usable desktop and push the window off-screen.
    for command in (
        ["wmctrl", "-r", WINDOW_NAME, "-b", "add,fullscreen"],
        ["wmctrl", "-r", WINDOW_NAME, "-b", "add,maximized_vert,maximized_horz"],
    ):
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

    return False


def show_display_frame(frame: np.ndarray, frame_number: int) -> int:
    cv2.imshow(WINDOW_NAME, frame)

    # This becomes the safe fallback size if the GUI backend cannot enter true
    # fullscreen. The default 960px-wide frame fits common small Jetson displays.
    if frame_number == 1:
        try:
            cv2.resizeWindow(WINDOW_NAME, frame.shape[1], frame.shape[0])
        except Exception:
            pass

    # Retry while the Jetson desktop compositor finishes mapping the window.
    # These attempts happen only during startup and do not affect steady-state
    # scanner performance.
    if FULLSCREEN_DISPLAY and frame_number in (1, 5, 20):
        cv2.waitKey(30)
        apply_fullscreen_window()

    return cv2.waitKey(1) & 0xFF


def set_display_blanking(disabled: bool) -> None:
    if not DISABLE_DISPLAY_BLANKING or not os.environ.get("DISPLAY"):
        return

    commands = (
        (["xset", "s", "off"], ["xset", "s", "on"]),
        (["xset", "-dpms"], ["xset", "+dpms"]),
        (["xset", "s", "noblank"], ["xset", "s", "blank"]),
    )
    for disable_command, restore_command in commands:
        command = disable_command if disabled else restore_command
        try:
            subprocess.run(command, check=False, capture_output=True, timeout=2)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue


def configure_usb_capture(cap: cv2.VideoCapture) -> None:
    if len(CAMERA_FOURCC) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def create_camera_capture(camera_index: int = 0) -> cv2.VideoCapture:
    # On Jetson/Linux, V4L2 is the first thing to try for normal USB cameras.
    if CAMERA_SOURCE in ("auto", "usb"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if cap.isOpened():
            configure_usb_capture(cap)
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
            "video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()

    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        configure_usb_capture(cap)
    return cap
                        
def main() -> None:
    print("Loading OCR model...")
    reader = create_ocr_reader()
    output_dir = find_usb_path()
    cap = create_camera_capture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Unable to open camera. Check camera index and backend support.")

    tracker = ConsensusTracker(CONFIRMATION_MATCHES, CONFIRMATION_WINDOW)
    date_tracker = ConsensusTracker(1, CONFIRMATION_WINDOW)
    scanned_ids: dict[str, float] = {}
    cooldown_seconds = 10.0
    last_ocr_time = 0.0
    displayed_countdown: Optional[int] = None
    paused = False
    pause_until = 0.0
    card_lost_frames = 0
    lost_threshold = 5
    camera_read_failures = 0
    displayed_frames = 0

    set_display_blanking(True)
    configure_display_window()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                camera_read_failures += 1
                if camera_read_failures >= 10:
                    raise RuntimeError("Camera stopped returning frames after 10 retries.")
                time.sleep(0.05)
                continue
            camera_read_failures = 0

            display, display_scale = resize_to_max_width(frame, DISPLAY_MAX_WIDTH)
            display = display.copy()
            now = time.monotonic()

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

                if now >= pause_until:
                    paused = False
                    displayed_countdown = None
                    tracker.reset()
                    date_tracker.reset()

                displayed_frames += 1
                if show_display_frame(display, displayed_frames) == ord("q"):
                    break
                continue

            # Detection is CPU-side and runs on a smaller frame. The selected box
            # is mapped back to the original frame so OCR and saved images retain detail.
            detection_frame, detection_scale = resize_to_max_width(frame, DETECTION_MAX_WIDTH)
            detected_box = find_card(detection_frame)

            if detected_box is not None:
                card_lost_frames = 0
                x, y, width, height = scale_box(detected_box, detection_scale, frame.shape)
                display_box = (
                    round(x * display_scale),
                    round(y * display_scale),
                    max(1, round(width * display_scale)),
                    max(1, round(height * display_scale)),
                )
                display_x, display_y, display_width, display_height = display_box
                cv2.rectangle(
                    display,
                    (display_x, display_y),
                    (display_x + display_width, display_y + display_height),
                    (0, 255, 0),
                    2,
                )

                roi = frame[y : y + height, x : x + width]

                if now - last_ocr_time >= OCR_INTERVAL_SECONDS:
                    last_ocr_time = now
                    try:
                        results = reader.readtext(
                            roi,
                            decoder="greedy",
                            beamWidth=1,
                            batch_size=1,
                            workers=0,
                            allowlist="0123456789/-",
                            detail=1,
                            paragraph=False,
                            canvas_size=OCR_CANVAS_SIZE,
                            mag_ratio=1.0,
                        )
                        id_number = extract_id(
                            results,
                            min_length=ID_MIN_LENGTH,
                            max_length=ID_MAX_LENGTH,
                            min_confidence=OCR_MIN_CONFIDENCE,
                            expected_length=ID_EXPECTED_LENGTH,
                            pattern=ID_PATTERN,
                        )
                        card_date = extract_card_date(
                            results,
                            min_confidence=OCR_MIN_CONFIDENCE,
                        )
                    except Exception as exc:
                        print(f"WARNING: OCR failed for this frame: {exc}")
                        id_number = None
                        card_date = None

                    confirmed_id, match_count = tracker.observe(id_number)
                    date_tracker.observe(card_date)
                    print(
                        f"Detected ID: {id_number} ({match_count}/{CONFIRMATION_MATCHES}); "
                        f"card date: {card_date}"
                    )

                    if id_number is not None and confirmed_id is None:
                        displayed_countdown = max(1, CONFIRMATION_MATCHES - match_count)
                    elif id_number is None:
                        displayed_countdown = None

                    if confirmed_id is not None:
                        confirmation_time = time.monotonic()
                        # Bound long-running kiosk memory use by pruning expired IDs.
                        scanned_ids = {
                            saved_id: saved_at
                            for saved_id, saved_at in scanned_ids.items()
                            if confirmation_time - saved_at <= cooldown_seconds
                        }
                        last_saved = scanned_ids.get(confirmed_id)

                        if last_saved is None:
                            try:
                                name_results = reader.readtext(
                                    roi,
                                    decoder="greedy",
                                    beamWidth=1,
                                    batch_size=1,
                                    workers=0,
                                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-' ",
                                    detail=1,
                                    paragraph=False,
                                    canvas_size=OCR_CANVAS_SIZE,
                                    mag_ratio=1.0,
                                )
                                full_name = extract_full_name(
                                    name_results,
                                    min_confidence=NAME_MIN_CONFIDENCE,
                                    logo_words=LOGO_WORDS,
                                ) or ""
                            except Exception as exc:
                                print(f"WARNING: Name OCR failed for this card: {exc}")
                                full_name = ""

                            output_dir = refresh_output_dir(output_dir)
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            recent_dates = [value for value in date_tracker.readings if value]
                            saved_card_date = (
                                max(recent_dates, key=recent_dates.count) if recent_dates else ""
                            )
                            output_dir = save_scan(
                                confirmed_id,
                                timestamp,
                                saved_card_date,
                                full_name,
                                output_dir,
                            )
                            output_dir = save_image(roi, timestamp, output_dir)
                            scanned_ids[confirmed_id] = confirmation_time

                        tracker.reset()
                        date_tracker.reset()
                        displayed_countdown = None
                        paused = True
                        pause_until = time.monotonic() + 2.5
            else:
                card_lost_frames += 1
                if card_lost_frames >= lost_threshold:
                    tracker.reset()
                    date_tracker.reset()
                    displayed_countdown = None

            if displayed_countdown is None or card_lost_frames > 0:
                draw_centered_overlay(
                    display,
                    "SCANNER READY",
                    color=(0, 255, 0),
                    bg_color=(0, 0, 0),
                    font_scale=1.6,
                    thickness=5,
                    y_ratio=0.2,
                )

            if displayed_countdown is not None and card_lost_frames == 0:
                draw_centered_overlay(
                    display,
                    str(displayed_countdown),
                    color=(0, 255, 0),
                    bg_color=(0, 0, 0),
                    font_scale=3.0,
                    thickness=8,
                    y_ratio=0.5,
                )

            displayed_frames += 1
            if show_display_frame(display, displayed_frames) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        set_display_blanking(False)


if __name__ == "__main__":
    main()
