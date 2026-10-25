"""Normalisasi teks yang dipakai IDENTIK saat training dan inference.

Placeholder (``__url__``, ``__phone__``, dst.) membuat model belajar *pola* (ada tautan,
ada nominal) alih-alih menghafal nomor/URL spesifik di dataset, sehingga lebih tahan
terhadap pergeseran data. Fungsi ini harus bersifat deterministik dan murni.
"""

from __future__ import annotations

import re
import unicodedata

URL_RE = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|id|co\.id|net|org|me|ly|io|xyz|site|online|top|info|link|app|vip|shop|cc|club|live)(?:/\S*)?"
)
PHONE_RE = re.compile(r"(?:\+?62|\b0)8\d[\d\s-]{6,13}\d")
MONEY_RE = re.compile(
    r"(?i)\brp\.?\s?\d[\d.,]*(?:\s?(?:rb|ribu|k|jt|juta|m|miliar)\b)?|\b\d[\d.,]*\s?(?:rb|ribu|jt|juta)\b"
)
USSD_RE = re.compile(r"\*\d{2,4}[*#\d]*#")
NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
REPEAT_RE = re.compile(r"(.)\1{2,}")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = URL_RE.sub(" __url__ ", text)
    text = USSD_RE.sub(" __ussd__ ", text)
    text = PHONE_RE.sub(" __phone__ ", text)
    text = MONEY_RE.sub(" __money__ ", text)
    text = NUM_RE.sub(" __num__ ", text)
    text = REPEAT_RE.sub(r"\1\1", text)  # "bangeeettt" -> "bangeett"
    return SPACE_RE.sub(" ", text).strip()
