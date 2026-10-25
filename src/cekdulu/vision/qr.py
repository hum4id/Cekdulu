"""Decode QR dari foto/screenshot dengan OpenCV (tanpa dependensi sistem tambahan)."""

from __future__ import annotations

import cv2
import numpy as np


def _decode(img: np.ndarray, detector: cv2.QRCodeDetector) -> list[str]:
    ok, decoded, _, _ = detector.detectAndDecodeMulti(img)
    if ok and decoded is not None:
        found = [d for d in decoded if d]
        if found:
            return found
    text, _, _ = detector.detectAndDecode(img)
    return [text] if text else []


def decode_qr_codes(bgr: np.ndarray) -> list[str]:
    """Coba beberapa varian (asli, grayscale, upscale, threshold) karena foto stiker QR
    sering buram, kecil, atau kontras rendah."""
    detector = cv2.QRCodeDetector()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    variants = [bgr, gray]
    h, w = gray.shape
    if max(h, w) < 1000:
        variants.append(cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    variants.append(
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    )
    for variant in variants:
        try:
            found = _decode(variant, detector)
        except cv2.error:
            continue
        if found:
            return list(dict.fromkeys(found))
    return []
