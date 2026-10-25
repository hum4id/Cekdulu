"""Penilaian risiko hasil decode QR: QRIS, URL, atau teks lain."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from cekdulu.domain.schemas import (
    QrisAssessment,
    QrisFinding,
    QrisInfo,
    RiskLevel,
    Severity,
)
from cekdulu.ml.tactics import RULES, assess_url
from cekdulu.qris.parser import QrisParseError, parse_qris

_QR_RECEIVE = next(r for r in RULES if r.code == "QR_RECEIVE_FUNDS").pattern
_RECEIVE_WORDS = re.compile(
    r"(?i)\b(terima(?!\s*kasih)|menerima|refund|pengembalian\s*dana|cairkan|pencairan|klaim)\b"
)
_SEV_TO_RISK = {
    Severity.LOW: RiskLevel.LOW,
    Severity.MEDIUM: RiskLevel.MEDIUM,
    Severity.HIGH: RiskLevel.HIGH,
    Severity.CRITICAL: RiskLevel.CRITICAL,
}
_ORDER = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]


def _max_risk(findings: list[QrisFinding]) -> RiskLevel:
    if not findings:
        return RiskLevel.LOW
    return max((_SEV_TO_RISK[f.severity] for f in findings), key=_ORDER.index)


def _similar(a: str, b: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def assess_qr_payload(
    payload: str,
    claimed_merchant: str | None = None,
    claimed_amount: int | None = None,
    context_text: str | None = None,
) -> QrisAssessment:
    payload = payload.strip()
    findings: list[QrisFinding] = []

    if context_text and (_QR_RECEIVE.search(context_text) or _RECEIVE_WORDS.search(context_text)):
        findings.append(
            QrisFinding(
                code="QR_FOR_RECEIVING",
                severity=Severity.CRITICAL,
                message="QRIS dipakai untuk MEMBAYAR. Siapa pun yang menyuruh scan QR untuk "
                "'menerima' dana/refund hampir pasti penipu.",
            )
        )

    if payload.lower().startswith(("http://", "https://", "www.")):
        for reason in assess_url(payload):
            sev = Severity.CRITICAL if "APK" in reason else Severity.HIGH
            findings.append(QrisFinding(code="QR_URL_SUSPICIOUS", severity=sev, message=reason))
        findings.append(
            QrisFinding(
                code="QR_IS_URL",
                severity=Severity.MEDIUM,
                message="QR ini berisi tautan, bukan QRIS pembayaran. Jangan masukkan data login/OTP.",
            )
        )
        return QrisAssessment(
            raw_type="url", url=payload, risk_level=_max_risk(findings), findings=findings
        )

    try:
        qris = parse_qris(payload)
    except QrisParseError as exc:
        if payload.startswith("000201"):
            findings.append(
                QrisFinding(
                    code="QRIS_MALFORMED",
                    severity=Severity.HIGH,
                    message=f"Struktur QRIS rusak: {exc}",
                )
            )
            return QrisAssessment(
                raw_type="invalid", risk_level=_max_risk(findings), findings=findings
            )
        return QrisAssessment(raw_type="text", risk_level=_max_risk(findings), findings=findings)

    f = qris.fields
    info = QrisInfo(
        is_qris=any(a.gui and "QRIS" in a.gui.upper() for a in qris.accounts)
        or f.get("58") == "ID",
        crc_valid=qris.crc_valid,
        initiation=qris.initiation,
        merchant_name=f.get("59"),
        merchant_city=f.get("60"),
        nmid=qris.nmid,
        acquirer_gui=[a.gui for a in qris.accounts if a.gui],
        merchant_criteria=next((a.criteria for a in qris.accounts if a.criteria), None),
        mcc=f.get("52"),
        currency=f.get("53"),
        country=f.get("58"),
        amount=f.get("54"),
        tip_indicator=f.get("55"),
    )

    if not qris.crc_valid:
        findings.append(
            QrisFinding(
                code="CRC_INVALID",
                severity=Severity.CRITICAL,
                message="Checksum (CRC) QR tidak cocok: payload telah diubah/dimanipulasi.",
            )
        )
    if info.country != "ID" or info.currency != "360":
        findings.append(
            QrisFinding(
                code="NOT_IDR",
                severity=Severity.HIGH,
                message=f"QR bukan untuk Rupiah/Indonesia (negara={info.country}, mata uang={info.currency}).",
            )
        )
    if not info.nmid:
        findings.append(
            QrisFinding(
                code="NO_NMID",
                severity=Severity.MEDIUM,
                message="Tidak ditemukan NMID QRIS (tag 51). Verifikasi merchant di aplikasi pembayaran.",
            )
        )
    if info.tip_indicator == "02":
        findings.append(
            QrisFinding(
                code="FIXED_FEE",
                severity=Severity.MEDIUM,
                message="QR menambahkan biaya/tip tetap otomatis. Periksa total sebelum membayar.",
            )
        )
    if claimed_merchant and info.merchant_name:
        ratio = _similar(claimed_merchant, info.merchant_name)
        if ratio < 0.6:
            findings.append(
                QrisFinding(
                    code="MERCHANT_MISMATCH",
                    severity=Severity.HIGH,
                    message=f"Nama merchant di QR ('{info.merchant_name}') berbeda dari yang diklaim "
                    f"('{claimed_merchant}'). Waspada stiker QR palsu (quishing).",
                )
            )
    if claimed_amount is not None and info.amount:
        try:
            qr_amount = float(info.amount)
        except ValueError:
            qr_amount = -1
        if abs(qr_amount - claimed_amount) > 0.5:
            findings.append(
                QrisFinding(
                    code="AMOUNT_MISMATCH",
                    severity=Severity.HIGH,
                    message=f"Nominal di QR ({info.amount}) berbeda dari yang disepakati ({claimed_amount}).",
                )
            )

    if not findings:
        findings.append(
            QrisFinding(
                code="STRUCTURE_OK",
                severity=Severity.LOW,
                message="Struktur QRIS valid. Tetap cocokkan nama merchant yang muncul di aplikasi "
                "pembayaran sebelum konfirmasi.",
            )
        )
    return QrisAssessment(
        raw_type="qris", info=info, risk_level=_max_risk(findings), findings=findings
    )
