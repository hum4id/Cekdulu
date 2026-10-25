"""Penanganan PII: masking untuk tampilan/log dan HMAC untuk korelasi tanpa menyimpan plaintext.

Prinsip (selaras UU No. 27/2022 tentang Pelindungan Data Pribadi): minimisasi data,
pembatasan tujuan, dan retensi terbatas. Nomor rekening & telepon tidak pernah dicatat
dalam bentuk utuh di log maupun tabel analisis.
"""

from __future__ import annotations

import hashlib
import hmac
import re

_DIGITS = re.compile(r"\D")


def digits_only(value: str) -> str:
    return _DIGITS.sub("", value)


def mask_number(value: str, keep_last: int = 4) -> str:
    """``1234567890`` -> ``******7890``. Non-digit diabaikan."""
    digits = digits_only(value)
    if not digits:
        return ""
    keep = min(keep_last, max(len(digits) - 2, 1))
    return "*" * (len(digits) - keep) + digits[-keep:]


def hmac_digest(value: str, secret: str) -> str:
    """HMAC-SHA256 hex dari nilai yang sudah dinormalisasi (lowercase, trim)."""
    normalized = value.strip().lower().encode("utf-8")
    return hmac.new(secret.encode("utf-8"), normalized, hashlib.sha256).hexdigest()


_PHONE = re.compile(r"(?:\+?62|0)8\d[\d\s-]{6,13}\d")
_LONG_DIGITS = re.compile(r"\b\d{8,20}\b")


def redact_text(text: str) -> str:
    """Masking nomor telepon dan deret digit panjang (kandidat rekening/NIK) di teks bebas."""
    text = _PHONE.sub(lambda m: mask_number(m.group()), text)
    return _LONG_DIGITS.sub(lambda m: mask_number(m.group()), text)
