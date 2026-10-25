"""Deteksi taktik manipulasi (rule-based, explainable) dan ekstraksi entitas.

Modul ini TIDAK menentukan label; ia memberi sinyal yang dapat diaudit untuk:
1) menjelaskan *mengapa* pesan berisiko, dan
2) eskalasi oleh ``RiskPolicy`` untuk pola yang hampir pasti berbahaya (minta OTP, kirim APK).

Pola disusun dari modus yang dipublikasikan OJK/IASC dan pemberitaan 2025-2026
(belanja daring, fake call/impersonation, investasi, lowongan kerja, APK WhatsApp, QRIS).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from cekdulu.domain.schemas import Entities, Severity, TacticHit
from cekdulu.ml.preprocessing import MONEY_RE, PHONE_RE, URL_RE
from cekdulu.security.pii import mask_number

_NEGATION = re.compile(
    r"(?i)\b(jangan|jgn|tidak pernah|tdk pernah|tak pernah|bukan)\b[^.!?\n]{0,40}$"
)


@dataclass(frozen=True)
class TacticRule:
    code: str
    label: str
    severity: Severity
    pattern: re.Pattern[str]
    negatable: bool = False


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


RULES: tuple[TacticRule, ...] = (
    TacticRule(
        "CREDENTIAL_REQUEST",
        "Meminta kode OTP/PIN/password",
        Severity.CRITICAL,
        _rx(
            r"\b(kirim(?:kan)?|sebut(?:kan)?|berikan|kasih|bagikan|infokan|balas|ketik|masukkan|"
            r"input|share|forward|teruskan)\b[^.!?\n]{0,40}\b(otp|kode\s*(?:verifikasi|otp|aktivasi|"
            r"keamanan|unik)|pin|password|kata\s*sandi|cvv|m-?pin)\b"
            r"|\b(kode\s*otp|otp|pin)\b[^.!?\n]{0,30}\b(kirim(?:kan)?|berikan|sebutkan|bagikan)\b"
            r"[^.!?\n]{0,20}\b(ke\s*(?:sini|kami|petugas|saya|nomor)|kepada\s*(?:kami|petugas|saya))"
        ),
        negatable=True,
    ),
    TacticRule(
        "APP_INSTALL",
        "Meminta memasang aplikasi/berkas APK",
        Severity.CRITICAL,
        _rx(
            r"\.apk\b|\b(unduh|download|instal|install|pasang)\b[^.!?\n]{0,40}\b(aplikasi|apk|"
            r"file|berkas)\b[^.!?\n]{0,40}\b(berikut|ini|terlampir|di\s*bawah|link)\b"
        ),
    ),
    TacticRule(
        "QR_RECEIVE_FUNDS",
        "Meminta scan QR untuk 'menerima' uang",
        Severity.CRITICAL,
        _rx(
            r"\b(scan|pindai)\b[^.!?\n]{0,40}\b(qr|qris|barcode)\b[^.!?\n]{0,60}\b(terima|menerima|"
            r"cairkan|pencairan|refund|pengembalian|klaim)\b"
        ),
    ),
    TacticRule(
        "ADVANCE_FEE",
        "Meminta transfer/biaya di muka",
        Severity.HIGH,
        _rx(
            r"\b(biaya\s*(?:admin|administrasi|pendaftaran|pengiriman|asuransi|pajak|aktivasi|"
            r"pencairan|materai|registrasi)|pajak\s*hadiah|deposit(?:o)?\s*(?:awal|dulu|terlebih)|"
            r"(?:tf|transfer|trf)\s*(?:dulu|terlebih\s*dahulu|dahulu)|uang\s*jaminan|top\s*up\s*dulu)\b"
        ),
    ),
    TacticRule(
        "WRONG_TRANSFER",
        "Klaim salah transfer & minta dikembalikan",
        Severity.HIGH,
        _rx(r"\b(salah\s*(?:transfer|tf|kirim)|kesalahan\s*transfer)\b"),
    ),
    TacticRule(
        "TOO_GOOD_TO_BE_TRUE",
        "Iming-iming hadiah/keuntungan tidak wajar",
        Severity.MEDIUM,
        _rx(
            r"\b(selamat[^.!?\n]{0,30}(?:terpilih|menang|mendapatkan)|pemenang|undian|hadiah\s*"
            r"(?:utama|langsung|tunai|mobil|motor)|untung\s*\d+\s*%|profit\s*(?:harian|pasti|"
            r"\d+)|pasti\s*untung|tanpa\s*risiko|modal\s*kecil\s*untung\s*besar|\d+\s*x\s*lipat|"
            r"cuan\s*(?:harian|pasti|besar)|penghasilan\s*(?:harian|jutaan))"
        ),
    ),
    TacticRule(
        "TASK_JOB_SCAM",
        "Lowongan kerja/tugas online berbayar",
        Severity.HIGH,
        _rx(
            r"\b(kerja\s*(?:paruh\s*waktu|sampingan|online|dari\s*rumah)|freelance\s*harian|"
            r"like\s*(?:dan|&)\s*(?:follow|subscribe)|komisi\s*per\s*(?:tugas|task)|"
            r"tugas\s*(?:like|review|rating)|gaji\s*harian)\b"
        ),
    ),
    TacticRule(
        "INVESTMENT_SCHEME",
        "Skema investasi/robot trading",
        Severity.HIGH,
        _rx(
            r"\b(robot\s*trading|investasi\s*(?:aman|terjamin|harian)|titip\s*dana|"
            r"member\s*vip|copy\s*trade|sinyal\s*trading|skema\s*bagi\s*hasil)\b"
        ),
    ),
    TacticRule(
        "ACCOUNT_THREAT",
        "Ancaman pemblokiran/penangguhan akun",
        Severity.HIGH,
        _rx(
            r"\b(akun|rekening|kartu|atm|nomor)\b[^.!?\n]{0,40}\b(diblokir|terblokir|dinonaktifkan|"
            r"ditangguhkan|dibekukan|hangus|kadaluarsa|kedaluwarsa)\b"
        ),
    ),
    TacticRule(
        "AUTHORITY_CLAIM",
        "Mengaku dari institusi/aparat",
        Severity.MEDIUM,
        _rx(
            r"\b(ojk|bank\s*indonesia|kepolisian|polisi|polda|polres|kejaksaan|ditjen\s*pajak|"
            r"djp|bpjs|e-?tilang|surat\s*tilang|kurir|petugas\s*bank|customer\s*service|"
            r"cs\s*(?:bank|resmi)|komdigi|kominfo|bea\s*cukai)\b"
        ),
    ),
    TacticRule(
        "URGENCY",
        "Tekanan waktu",
        Severity.LOW,
        _rx(
            r"\b(segera|secepatnya|sekarang\s*juga|hari\s*ini\s*juga|dalam\s*\d+\s*(?:jam|menit)|"
            r"batas\s*waktu|kesempatan\s*terakhir|darurat|mendesak|buruan)\b"
        ),
    ),
    TacticRule(
        "SECRECY",
        "Meminta merahasiakan",
        Severity.MEDIUM,
        _rx(r"\b(jangan\s*(?:bilang|beritahu|kasih\s*tahu)\s*(?:siapa|orang)|rahasiakan)\b"),
    ),
)

SHORTENERS = {
    "bit.ly",
    "s.id",
    "tinyurl.com",
    "t.ly",
    "cutt.ly",
    "shorturl.at",
    "rb.gy",
    "is.gd",
    "ow.ly",
    "tiny.cc",
    "goo.su",
    "linktr.ee",
    "wa.me",
    "t.me",
}
RISKY_TLDS = {"xyz", "top", "site", "online", "click", "buzz", "vip", "shop", "live", "info", "icu"}
BRANDS = (
    "bca",
    "bri",
    "bni",
    "mandiri",
    "bsi",
    "cimb",
    "btn",
    "permata",
    "danamon",
    "jago",
    "seabank",
    "ovo",
    "gopay",
    "shopeepay",
    "linkaja",
    "shopee",
    "tokopedia",
    "lazada",
    "jne",
    "j&t",
    "sicepat",
    "pos indonesia",
    "pln",
    "bpjs",
    "ojk",
    "telkomsel",
    "indosat",
)
_BRAND_RE = re.compile(r"(?i)\b(" + "|".join(re.escape(b) for b in BRANDS) + r")\b")
_ACCOUNT_CTX = re.compile(
    r"(?i)(?:no\.?\s*rek(?:ening)?|norek|rekening|rek\.?|va|virtual\s*account)\D{0,15}(\d[\d\s-]{6,22}\d)"
)
_IP_HOST = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def _is_negated(text: str, start: int) -> bool:
    return bool(_NEGATION.search(text[max(0, start - 60) : start]))


def detect_tactics(text: str) -> list[TacticHit]:
    hits: list[TacticHit] = []
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            if rule.negatable and _is_negated(text, match.start()):
                continue
            hits.append(
                TacticHit(
                    code=rule.code,
                    label=rule.label,
                    severity=rule.severity,
                    evidence=match.group(0).strip()[:120],
                )
            )
            break  # satu bukti per aturan sudah cukup untuk penjelasan
    hits.extend(_url_tactics(extract_urls(text)))
    return hits


def extract_urls(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(0).rstrip(".,);]") for m in URL_RE.finditer(text)))


def _host(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    return (parsed.hostname or "").lower()


def assess_url(url: str) -> list[str]:
    """Kembalikan daftar alasan URL mencurigakan (kosong = tidak ada sinyal)."""
    reasons: list[str] = []
    host = _host(url)
    if not host:
        return reasons
    path = urlparse(url if "://" in url else f"http://{url}").path.lower()
    if host in SHORTENERS or host.removeprefix("www.") in SHORTENERS:
        reasons.append(f"tautan pemendek ({host}) menyembunyikan tujuan asli")
    if _IP_HOST.match(host):
        reasons.append("tautan memakai alamat IP, bukan domain resmi")
    if "xn--" in host:
        reasons.append("domain punycode (bisa meniru huruf domain resmi)")
    if path.endswith(".apk"):
        reasons.append("tautan mengunduh berkas APK")
    tld = host.rsplit(".", 1)[-1]
    brand = _BRAND_RE.search(host.replace("-", " ").replace(".", " "))
    if brand and (tld in RISKY_TLDS or "-" in host):
        reasons.append(f"domain mencatut nama '{brand.group(1)}' pada domain tidak resmi")
    elif tld in RISKY_TLDS:
        reasons.append(f"domain ber-TLD .{tld} yang sering dipakai situs palsu")
    return reasons


def _url_tactics(urls: list[str]) -> list[TacticHit]:
    for url in urls:
        reasons = assess_url(url)
        if reasons:
            critical = any("APK" in r for r in reasons)
            return [
                TacticHit(
                    code="SUSPICIOUS_LINK",
                    label="Tautan mencurigakan: " + "; ".join(reasons),
                    severity=Severity.CRITICAL if critical else Severity.HIGH,
                    evidence=url[:120],
                )
            ]
    return []


_UNIT = {
    "rb": 1_000,
    "ribu": 1_000,
    "k": 1_000,
    "jt": 1_000_000,
    "juta": 1_000_000,
    "m": 1_000_000_000,
    "miliar": 1_000_000_000,
}


def parse_amount_idr(raw: str) -> int | None:
    s = raw.lower().replace("rp", "").replace(" ", "").strip(". ")
    unit = 1
    for suffix, mult in sorted(_UNIT.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(suffix):
            unit, s = mult, s[: -len(suffix)]
            break
    if not s:
        return None
    if unit > 1 and "," in s and "." not in s:  # 1,5jt
        s = s.replace(",", ".")
        try:
            return int(float(s) * unit)
        except ValueError:
            return None
    s = re.sub(r"[.,](\d{3})", r"\1", s)  # pemisah ribuan
    s = s.replace(",", ".")
    try:
        return int(float(s) * unit)
    except ValueError:
        return None


def extract_entities(text: str) -> Entities:
    urls = extract_urls(text)
    phones = [mask_number(m.group()) for m in PHONE_RE.finditer(text)]
    accounts = [mask_number(m.group(1)) for m in _ACCOUNT_CTX.finditer(text)]
    amounts = [a for a in (parse_amount_idr(m.group()) for m in MONEY_RE.finditer(text)) if a]
    brands = [m.group(1).upper() for m in _BRAND_RE.finditer(text)]
    return Entities(
        urls=urls[:10],
        phone_numbers=list(dict.fromkeys(phones))[:10],
        account_numbers=list(dict.fromkeys(accounts))[:10],
        amounts_idr=list(dict.fromkeys(amounts))[:10],
        impersonated=list(dict.fromkeys(brands))[:10],
    )
