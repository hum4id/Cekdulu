"""Kebijakan risiko: menggabungkan probabilitas model dan sinyal taktik secara transparan.

Desain:
- Skor dasar = P(penipuan) dari classifier.
- Threshold keputusan diambil dari metadata model (dituning di validation set untuk
  mengejar recall kelas penipuan), bukan argmax 0.5.
- Eskalasi deterministik untuk taktik CRITICAL (minta OTP, APK, scan-QR-terima-dana):
  pola ini hampir tidak pernah muncul di pesan sah, dan false negative-nya mahal.
- Tidak ada aturan yang MENURUNKAN skor model (aturan hanya menambah kewaspadaan).
"""

from __future__ import annotations

from dataclasses import dataclass

from cekdulu.domain.schemas import RiskLevel, Severity, TacticHit

CRITICAL_FLOOR = 0.90
MULTI_HIGH_FLOOR = 0.70


@dataclass(frozen=True)
class RiskDecision:
    score: float
    level: RiskLevel
    escalated_by: list[str]


class RiskPolicy:
    def __init__(self, fraud_threshold: float, low_cutoff: float = 0.25) -> None:
        if not 0 < low_cutoff < fraud_threshold < 1:
            raise ValueError("Harus 0 < low_cutoff < fraud_threshold < 1")
        self.fraud_threshold = fraud_threshold
        self.low_cutoff = low_cutoff

    def decide(self, p_fraud: float, tactics: list[TacticHit]) -> RiskDecision:
        score = float(p_fraud)
        escalated: list[str] = []

        critical = [t.code for t in tactics if t.severity == Severity.CRITICAL]
        high = [t.code for t in tactics if t.severity == Severity.HIGH]
        if critical:
            score = max(score, CRITICAL_FLOOR)
            escalated.extend(critical)
        elif len(high) >= 2:
            score = max(score, MULTI_HIGH_FLOOR)
            escalated.extend(high)

        if score >= CRITICAL_FLOOR:
            level = RiskLevel.CRITICAL
        elif score >= self.fraud_threshold:
            level = RiskLevel.HIGH
        elif score >= self.low_cutoff or high:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW
        return RiskDecision(score=round(score, 4), level=level, escalated_by=escalated)

    @staticmethod
    def is_fraud(level: RiskLevel) -> bool:
        """Definisi 'terdeteksi penipuan' untuk evaluasi end-to-end."""
        return level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
