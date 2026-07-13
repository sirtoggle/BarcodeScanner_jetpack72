from __future__ import annotations

import os
import re
import warnings
from collections import deque
from datetime import datetime
from typing import Any, Iterable, Optional, Pattern, Sequence, Tuple


OcrResult = Tuple[Any, str, Any]


def _normalize_card_date(month_text: str, day_text: str, year_text: str) -> Optional[str]:
    try:
        month = int(month_text)
        day = int(day_text)
        year = int(year_text)
    except ValueError:
        return None

    if len(year_text) == 2:
        year += 2000 if year <= 69 else 1900
    if not 1900 <= year <= 2100:
        return None

    try:
        parsed = datetime(year, month, day)
    except ValueError:
        return None
    return parsed.strftime("%m/%d/%Y")


def _extract_date_from_text(text: str) -> Optional[str]:
    compact_text = re.sub(r"\s+", "", text)

    # Normal US dates such as 07/13/2026 or 07-13-2026. Also accept the
    # unambiguous ISO order 2026/07/13.
    for match in re.finditer(
        r"(?<!\d)(\d{1,4})[./-](\d{1,2})[./-](\d{2,4})(?!\d)",
        compact_text,
    ):
        first, second, third = match.groups()
        if len(first) == 4:
            normalized = _normalize_card_date(second, third, first)
        else:
            normalized = _normalize_card_date(first, second, third)
        if normalized is not None:
            return normalized

    # EasyOCR sometimes recognizes one or both slashes as the digit 7. Accept
    # that substitution only at the two fixed separator positions and only when
    # the remaining value is a valid calendar date.
    for match in re.finditer(
        r"(?<!\d)(\d{2})([./-]|7)(\d{2})([./-]|7)(\d{4})(?!\d)",
        compact_text,
    ):
        month, first_separator, day, second_separator, year = match.groups()
        if "7" not in (first_separator, second_separator):
            continue
        normalized = _normalize_card_date(month, day, year)
        if normalized is not None:
            return normalized

    for match in re.finditer(
        r"(?<!\d)(\d{4})([./-]|7)(\d{2})([./-]|7)(\d{2})(?!\d)",
        compact_text,
    ):
        year, first_separator, month, second_separator, day = match.groups()
        if "7" not in (first_separator, second_separator):
            continue
        normalized = _normalize_card_date(month, day, year)
        if normalized is not None:
            return normalized

    return None


def extract_card_date(
    results: Iterable[OcrResult],
    *,
    min_confidence: float = 0.40,
) -> Optional[str]:
    """Return a normalized card date without treating arbitrary digits as dates."""
    candidates: list[tuple[float, str]] = []
    for result in results:
        if len(result) < 3:
            continue
        try:
            confidence = float(result[2])
        except (TypeError, ValueError):
            continue
        if confidence < min_confidence:
            continue

        candidate = _extract_date_from_text(str(result[1]))
        if candidate is not None:
            candidates.append((confidence, candidate))

    return max(candidates)[1] if candidates else None


def path_is_on_mount(path: str, mounts: Sequence[str]) -> bool:
    real_path = os.path.realpath(path)
    for mount in mounts:
        real_mount = os.path.realpath(mount)
        try:
            if os.path.commonpath((real_path, real_mount)) == real_mount:
                return True
        except ValueError:
            continue
    return False


def scale_box(
    box: Tuple[int, int, int, int],
    scale: float,
    frame_shape: tuple[int, ...],
) -> Tuple[int, int, int, int]:
    """Map a box from a resized detection frame to the original frame."""
    if scale <= 0:
        raise ValueError("scale must be positive")

    x, y, width, height = box
    frame_height, frame_width = frame_shape[:2]
    scaled_x = max(0, min(frame_width - 1, round(x / scale)))
    scaled_y = max(0, min(frame_height - 1, round(y / scale)))
    scaled_width = max(1, min(frame_width - scaled_x, round(width / scale)))
    scaled_height = max(1, min(frame_height - scaled_y, round(height / scale)))
    return scaled_x, scaled_y, scaled_width, scaled_height


def is_gpu_available(torch_module: Optional[Any]) -> bool:
    """Return CUDA availability without leaking the noisy Jetson capability warning."""
    if torch_module is None:
        return False

    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None:
        return False

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Found CPU0 Orin.*compute capability.*",
                category=UserWarning,
            )
            return bool(cuda_module.is_available())
    except Exception:
        return False


def extract_id(
    results: Iterable[OcrResult],
    *,
    min_length: int = 6,
    max_length: int = 20,
    min_confidence: float = 0.40,
    expected_length: Optional[int] = None,
    pattern: Optional[Pattern[str]] = None,
) -> Optional[str]:
    """Choose the most confident valid numeric identifier from OCR results.

    Candidates are kept within a single OCR text region so unrelated fields on a
    card are never concatenated. Spaces and hyphens are removed because OCR often
    inserts them into long card numbers.
    """
    candidates: list[tuple[float, int, str]] = []

    for result in results:
        if len(result) < 3:
            continue

        text = str(result[1])
        try:
            confidence = float(result[2])
        except (TypeError, ValueError):
            continue

        if confidence < min_confidence:
            continue

        # A validated date is a separate card field and must never compete with
        # the identifier, including when OCR replaced both slashes with 7.
        if _extract_date_from_text(text) is not None:
            continue

        compact_text = re.sub(r"[\s-]+", "", text)
        for candidate in re.findall(r"\d+", compact_text):
            if expected_length is not None and len(candidate) != expected_length:
                continue
            if not min_length <= len(candidate) <= max_length:
                continue
            if pattern is not None and pattern.fullmatch(candidate) is None:
                continue
            candidates.append((confidence, len(candidate), candidate))

    if not candidates:
        return None

    return max(candidates)[2]


def compile_id_pattern(value: str) -> Optional[Pattern[str]]:
    value = value.strip()
    return re.compile(value) if value else None


class ConsensusTracker:
    """Require repeated recent OCR readings while tolerating occasional misses."""

    def __init__(self, required_matches: int = 3, window_size: int = 5) -> None:
        if required_matches < 1:
            raise ValueError("required_matches must be at least 1")
        if window_size < required_matches:
            raise ValueError("window_size must be at least required_matches")

        self.required_matches = required_matches
        self._readings: deque[Optional[str]] = deque(maxlen=window_size)

    def observe(self, value: Optional[str]) -> tuple[Optional[str], int]:
        self._readings.append(value)
        if value is None:
            return None, 0

        matches = sum(reading == value for reading in self._readings)
        confirmed = value if matches >= self.required_matches else None
        return confirmed, matches

    def reset(self) -> None:
        self._readings.clear()

    @property
    def readings(self) -> Sequence[Optional[str]]:
        return tuple(self._readings)
