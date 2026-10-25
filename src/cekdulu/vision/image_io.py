"""Validasi dan decoding gambar unggahan secara aman."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_PIXELS = 25_000_000  # cegah decompression bomb

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class InvalidImageError(ValueError):
    pass


def load_image(data: bytes, max_bytes: int) -> Image.Image:
    if not data:
        raise InvalidImageError("Berkas kosong.")
    if len(data) > max_bytes:
        raise InvalidImageError(f"Ukuran gambar melebihi {max_bytes // (1024 * 1024)} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise InvalidImageError("Berkas bukan gambar yang valid.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise InvalidImageError(f"Format {fmt} tidak didukung. Gunakan JPEG/PNG/WEBP.")
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def to_bgr_array(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img)
    return rgb[:, :, ::-1].copy()


def to_png_bytes(img: Image.Image, max_side: int = 1600) -> bytes:
    """Re-encode (membuang metadata EXIF/GPS) dan membatasi resolusi sebelum dikirim ke LLM."""
    copy = img.copy()
    copy.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    copy.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
