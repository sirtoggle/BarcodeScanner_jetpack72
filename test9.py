import cv2
import numpy as np
import easyocr
import csv
import time
import re
import os
import importlib
from datetime import datetime

try:
    import torch
except Exception:
    torch = None

try:
    GPIO = importlib.import_module("Jetson.GPIO")
except Exception:
    GPIO = None


def is_gpu_available():
    return bool(torch is not None and torch.cuda.is_available())


if is_gpu_available():
    # Let cuDNN pick the fastest kernels for fixed input sizes.
    torch.backends.cudnn.benchmark = True


cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, os.cpu_count() or 1))


reader = easyocr.Reader(['en'], gpu=is_gpu_available())

GREEN_LED_PIN = int(os.getenv("GREEN_LED_PIN", "12"))
LED_BLINK_COUNT = int(os.getenv("LED_BLINK_COUNT", "3"))
LED_ON_SECONDS = float(os.getenv("LED_ON_SECONDS", "0.15"))
LED_OFF_SECONDS = float(os.getenv("LED_OFF_SECONDS", "0.15"))

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

GPIO_READY = False


def setup_led():
    global GPIO_READY

    if GPIO is None:
        return

    try:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(GREEN_LED_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO_READY = True
    except Exception as exc:
        print(f"GPIO setup failed: {exc}")
        GPIO_READY = False


# -----------------------------
# CARD DETECTION (ROBUST)
# -----------------------------
def find_card(frame):
    frame_area = frame.shape[0] * frame.shape[1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # CLAHE for better light card detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 30, 100)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)

        if area < frame_area * 0.07:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

        x, y, w, h = cv2.boundingRect(c)

        aspect = w / float(h)
        if aspect < 0.5 or aspect > 2.8:
            continue

        if area > best_area:
            best_area = area
            best_box = (x, y, w, h)

    return best_box


# -----------------------------
# OCR ID EXTRACTION
# -----------------------------
def extract_id(results):
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
def save_scan(id_number, timestamp):
    date_str = datetime.now().strftime("%m-%d-%Y")
    path = find_usb_path()
    with open(os.path.join(path, f"scans{date_str}.csv"), "a", newline="") as f:
        csv.writer(f).writerow([id_number, timestamp])

    print(f"✔ SAVED → {id_number} | {timestamp}")

def save_image(roi, timestamp):
    path = find_usb_path()
    filename = os.path.join(path, f"{timestamp}.jpg")
    cv2.imwrite(filename, roi)
    print(f"📸 SAVED IMAGE → {filename}")


def blink_green_led(times=LED_BLINK_COUNT, on_time=LED_ON_SECONDS, off_time=LED_OFF_SECONDS):
    if not GPIO_READY:
        # If the board cannot control the LED, show a message instead.
        print("LED blink requested (GPIO unavailable)")
        return

    for _ in range(max(1, times)):
        GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
        time.sleep(max(0.01, on_time))
        GPIO.output(GREEN_LED_PIN, GPIO.LOW)
        time.sleep(max(0.01, off_time))

def find_usb_path():

    for base in ["/media", "/mnt", "/run/media"]:
        if not os.path.exists(base):
            continue

        for root, dirs, _ in os.walk(base):
            if dirs:
                candidate = os.path.join(root, dirs[0])
                if os.path.isdir(candidate):
                    return candidate

    fallback_dir = os.path.join(os.getcwd(), "id_scanner_output")
    os.makedirs(fallback_dir, exist_ok=True)
    return fallback_dir


def create_camera_capture(camera_index=0):
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
                        
def main():
    cap = create_camera_capture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Unable to open camera. Check camera index and backend support.")

    setup_led()

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

    paused = False
    pause_until = 0
    
    # Count how many frames in a row the card is missing.
    card_lost_frames = 0
    # Only clear the scan buffer if the card has been missing for several frames.
    lost_threshold = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Pause briefly after a successful scan so the user can remove the card.

        if paused:
            cv2.putText(display, "REMOVE CARD", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            if time.time() > pause_until:
                paused = False

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

                                blink_green_led()

                                paused = True
                                pause_until = time.time() + 2.5

                            # Clear the temporary tracking data after we save or skip the card.

                            recent_ids.clear()
                            stable_id = None
                            stable_start_time = None

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

        # Show the current buffer on the video window so debugging is easier.

        cv2.putText(display, f"Buffer: {recent_ids}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("ID Scanner", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if GPIO_READY:
        GPIO.cleanup()


if __name__ == "__main__":
    main()