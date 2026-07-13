from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Callable


ImageWriter = Callable[[str, Any], bool]


def fallback_output_dir(working_dir: str) -> str:
    path = os.path.join(working_dir, "id_scanner_output")
    os.makedirs(path, exist_ok=True)
    return path


def set_private_permissions(filename: str) -> None:
    if os.name == "posix":
        try:
            os.chmod(filename, 0o600)
        except OSError:
            # FAT/exFAT removable drives do not implement POSIX permissions.
            pass


def write_scan_row(
    output_dir: str,
    id_number: str,
    timestamp: str,
    card_date: str = "",
) -> str:
    date_str = datetime.now().strftime("%m-%d-%Y")
    filename = os.path.join(output_dir, f"scans{date_str}.csv")

    with open(filename, "a", newline="", encoding="utf-8") as output_file:
        # Preserve the original ID and scan-time column positions. Card date is
        # blank when OCR did not confidently detect a valid printed date.
        csv.writer(output_file).writerow([id_number, timestamp, card_date])
        output_file.flush()
        os.fsync(output_file.fileno())
    set_private_permissions(filename)
    return filename


def write_image_file(
    output_dir: str,
    image: Any,
    timestamp: str,
    image_writer: ImageWriter,
) -> str:
    filename = os.path.join(output_dir, f"{timestamp}.jpg")
    temporary_file = os.path.join(output_dir, f".{timestamp}.{os.getpid()}.tmp.jpg")

    try:
        if not image_writer(temporary_file, image):
            raise OSError(f"Image encoder could not write {temporary_file}")
        os.replace(temporary_file, filename)
        set_private_permissions(filename)
        return filename
    finally:
        try:
            if os.path.exists(temporary_file):
                os.remove(temporary_file)
        except OSError:
            pass
