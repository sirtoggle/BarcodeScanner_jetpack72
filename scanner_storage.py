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


def flush_file(filename: str) -> None:
    """Wait for a completed file's contents to reach its storage device."""
    # Windows requires a writable descriptor for fsync; Linux accepts this too.
    with open(filename, "rb+") as saved_file:
        os.fsync(saved_file.fileno())


def flush_directory(path: str) -> None:
    """Persist directory entries when the filesystem supports directory fsync."""
    def flush_all_linux_storage() -> None:
        # FAT/exFAT drivers can reject directory fsync. Linux sync is the safe
        # fallback for removable media; it is only used when needed.
        if os.name == "posix" and hasattr(os, "sync"):
            os.sync()

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(path, flags)
    except OSError:
        flush_all_linux_storage()
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        flush_all_linux_storage()
    finally:
        os.close(directory_fd)


def write_scan_row(
    output_dir: str,
    id_number: str,
    timestamp: str,
    card_date: str = "",
    full_name: str = "",
) -> str:
    date_str = datetime.now().strftime("%m-%d-%Y")
    filename = os.path.join(output_dir, f"scans{date_str}.csv")

    with open(filename, "a", newline="", encoding="utf-8") as output_file:
        # Preserve the original ID and scan-time column positions. Card date is
        # blank when OCR did not confidently detect a valid printed date.
        csv.writer(output_file).writerow([id_number, timestamp, card_date, full_name])
        output_file.flush()
        set_private_permissions(filename)
        os.fsync(output_file.fileno())
    flush_directory(output_dir)
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
        set_private_permissions(temporary_file)
        flush_file(temporary_file)
        os.replace(temporary_file, filename)
        flush_directory(output_dir)
        return filename
    finally:
        try:
            if os.path.exists(temporary_file):
                os.remove(temporary_file)
        except OSError:
            pass
