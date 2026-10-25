"""Orkestrasi analisis: teks, gambar (QR + OCR), QRIS, dan insiden golden hour.

Service ini murni (tanpa FastAPI) sehingga dipakai bersama oleh API, bot polling, dan
demo Gradio. Semua dependensi diinjeksi lewat konstruktor (mudah di-test/mock).
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta

from cekdulu.config import Settings
from cekdulu.domain.schemas import (
    RISK_LABEL_ID,
    ImageAnalysisResult,
    IncidentCreate,
    IncidentResult,
    Label,
    QrisAssessment,
    RiskLevel,
    TacticHit,
    TextAnalysisResult,
)
from cekdulu.llm.base import LLMClient, LLMError
from cekdulu.llm.prompts import EXPLAIN_SCHEMA, EXPLAIN_SYSTEM, build_explain_user_prompt
from cekdulu.ml.classifier import TextClassifier
from cekdulu.ml.risk import RiskPolicy
from cekdulu.ml.tactics import detect_tactics, extract_entities
from cekdulu.observability import ANALYSES_TOTAL, LLM_FALLBACK_TOTAL
from cekdulu.qris.assess import assess_qr_payload
from cekdulu.security.pii import digits_only, hmac_digest, mask_number, redact_text
from cekdulu.services.golden_hour import REQUIRED_FIELDS, build_steps, minutes_since, urgency_label
from cekdulu.services.report_pdf import render_incident_pdf
from cekdulu.storage.db import AnalysisRecord, IncidentRecord, ReportRecord, Repository, new_id
from cekdulu.vision.image_io import load_image, to_bgr_array
from cekdulu.vision.ocr import OcrEngine
from cekdulu.vision.qr import decode_qr_codes

logger = logging.getLogger(__name__)
_ORDER = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

ACTIONS: dict[RiskLevel, list[str]] = {
    RiskLevel.CRITICAL: [
        "JANGAN transfer, klik tautan, memasang aplikasi, atau memberikan OTP/PIN.",
        "Blokir dan laporkan nomor/akun pengirim di aplikasi chat.",
        "Verifikasi lewat kanal resmi (aplikasi/situs resmi atau call center di balik kartu).",
        "Jika sudah terlanjur transfer, ketik /lapor untuk panduan golden hour.",
    ],
    RiskLevel.HIGH: [
        "Jangan transfer atau klik tautan sebelum verifikasi independen.",
        "Hubungi pihak yang diklaim lewat nomor/aplikasi resmi, bukan kontak dari pesan.",
        "Terapkan prinsip 2L OJK: Legal (cek izin) dan Logis (imbal hasil masuk akal?).",
    ],
    RiskLevel.MEDIUM: [
        "Ada ciri yang patut diwaspadai. Verifikasi pengirim lewat kanal resmi.",
        "Jangan membagikan data pribadi atau kode OTP kepada siapa pun.",
    ],
    RiskLevel.LOW: [
        "Tidak ditemukan ciri penipuan yang kuat, tetapi tetap jangan bagikan OTP/PIN.",
    ],
}


def _template_explanation(
    level: RiskLevel, label: Label, tactics: list[TacticHit], p_fraud: float
) -> str:
    if tactics:
        names = ", ".join(t.label.lower() for t in tactics[:3])
        core = f"Pesan ini menunjukkan ciri: {names}."
    else:
        core = "Tidak ada taktik manipulasi spesifik yang terdeteksi oleh aturan."
    model = f"Model menilai kemiripan dengan pola penipuan sebesar {p_fraud:.0%}"
    if label == Label.PROMO and level in (RiskLevel.LOW, RiskLevel.MEDIUM):
        model += " dan pesan lebih mirip promosi"
    return f"{RISK_LABEL_ID[level]}. {core} {model}."


class AnalysisService:
    def __init__(
        self,
        settings: Settings,
        classifier: TextClassifier,
        llm: LLMClient,
        repo: Repository,
        ocr: OcrEngine,
    ) -> None:
        self.settings = settings
        self.classifier = classifier
        self.llm = llm
        self.repo = repo
        self.ocr = ocr
        threshold = settings.fraud_threshold or classifier.metadata.fraud_threshold
        self.policy = RiskPolicy(threshold, low_cutoff=min(0.25, threshold / 2))
        self._secret = settings.hash_secret.get_secret_value()
        self._fraud_idx = classifier.metadata.labels.index(Label.PENIPUAN)

    # ------------------------------------------------------------------ text
    def analyze_text(
        self, text: str, channel: str = "api", persist: bool = True, explain_with_llm: bool = True
    ) -> TextAnalysisResult:
        start = time.perf_counter()
        text = text.strip()[: self.settings.max_text_chars]
        probs = self.classifier.predict_proba([text])[0]
        labels = self.classifier.metadata.labels
        p_fraud = float(probs[self._fraud_idx])
        tactics = detect_tactics(text)
        decision = self.policy.decide(p_fraud, tactics)
        label = labels[int(probs.argmax())]
        if RiskPolicy.is_fraud(decision.level):
            label = Label.PENIPUAN

        explanation, source = (
            _template_explanation(decision.level, label, tactics, p_fraud),
            "template",
        )
        if explain_with_llm and self.llm.enabled:
            try:
                data = self.llm.generate_json(
                    system=EXPLAIN_SYSTEM,
                    user_text=build_explain_user_prompt(
                        redact_text(text), RISK_LABEL_ID[decision.level], [t.label for t in tactics]
                    ),
                    schema=EXPLAIN_SCHEMA,
                )
                candidate = str(data.get("explanation", "")).strip()
                if 20 <= len(candidate) <= 800:
                    explanation, source = candidate, "llm"
            except LLMError as exc:
                LLM_FALLBACK_TOTAL.inc()
                logger.warning("llm_explain_fallback", extra={"error": str(exc)})

        analysis_id = new_id()
        latency = round((time.perf_counter() - start) * 1000, 1)
        result = TextAnalysisResult(
            analysis_id=analysis_id,
            label=label,
            probabilities={lbl: round(float(p), 4) for lbl, p in zip(labels, probs, strict=True)},
            risk_score=decision.score,
            risk_level=decision.level,
            risk_label=RISK_LABEL_ID[decision.level],
            tactics=tactics,
            entities=extract_entities(text),
            explanation=explanation,
            recommended_actions=ACTIONS[decision.level],
            model_version=self.classifier.metadata.model_version,
            explanation_source=source,
            latency_ms=latency,
        )
        ANALYSES_TOTAL.labels(kind="text", risk_level=decision.level.value).inc()
        if persist:
            self.repo.add(
                AnalysisRecord(
                    id=analysis_id,
                    kind="text",
                    channel=channel,
                    label=label.value,
                    risk_level=decision.level.value,
                    risk_score=decision.score,
                    model_version=result.model_version,
                    text_hmac=hmac_digest(text, self._secret),
                    text_len=len(text),
                    raw_text=text if self.settings.store_raw_text else None,
                    tactics=[t.code for t in tactics],
                    latency_ms=latency,
                )
            )
        logger.info(
            "text_analyzed",
            extra={
                "analysis_id": analysis_id,
                "risk": decision.level.value,
                "p_fraud": round(p_fraud, 4),
                "latency_ms": latency,
            },
        )
        return result

    # ------------------------------------------------------------------ qris
    def analyze_qr_payload(
        self,
        payload: str,
        claimed_merchant: str | None = None,
        claimed_amount: int | None = None,
        context_text: str | None = None,
        channel: str = "api",
    ) -> QrisAssessment:
        result = assess_qr_payload(payload, claimed_merchant, claimed_amount, context_text)
        ANALYSES_TOTAL.labels(kind="qris", risk_level=result.risk_level.value).inc()
        self.repo.add(
            AnalysisRecord(
                kind="qris",
                channel=channel,
                risk_level=result.risk_level.value,
                tactics=[f.code for f in result.findings],
            )
        )
        return result

    # ------------------------------------------------------------------ image
    def analyze_image(
        self, data: bytes, caption: str | None = None, channel: str = "api"
    ) -> ImageAnalysisResult:
        img = load_image(data, max_bytes=int(self.settings.max_upload_mb * 1024 * 1024))
        warnings: list[str] = []
        qr_payloads = decode_qr_codes(to_bgr_array(img))
        qr_results = [assess_qr_payload(p, context_text=caption) for p in qr_payloads[:5]]

        ocr_text, ocr_source = (None, None)
        if not qr_payloads or caption is None:
            ocr_text, ocr_source = self.ocr.extract(img)
            if ocr_text is None and not qr_payloads:
                warnings.append(
                    "Teks pada gambar tidak dapat dibaca (OCR tidak tersedia/tidak berhasil). "
                    "Coba kirim teks pesannya langsung."
                )

        combined = "\n".join(t for t in (caption, ocr_text) if t)
        text_result = (
            self.analyze_text(combined, channel=channel, persist=False)
            if combined.strip()
            else None
        )
        warnings.append(
            "Gambar tidak dapat membuktikan uang sudah masuk. Untuk bukti transfer, selalu cek "
            "mutasi rekening di aplikasi resmi."
        )

        levels = [r.risk_level for r in qr_results]
        if text_result:
            levels.append(text_result.risk_level)
        overall = max(levels, key=_ORDER.index) if levels else RiskLevel.LOW
        analysis_id = new_id()
        ANALYSES_TOTAL.labels(kind="image", risk_level=overall.value).inc()
        self.repo.add(
            AnalysisRecord(
                id=analysis_id,
                kind="image",
                channel=channel,
                risk_level=overall.value,
                risk_score=text_result.risk_score if text_result else None,
                model_version=self.classifier.metadata.model_version,
                tactics=[t.code for t in text_result.tactics] if text_result else [],
            )
        )
        return ImageAnalysisResult(
            analysis_id=analysis_id,
            qr_codes=qr_results,
            ocr_text=ocr_text,
            ocr_source=ocr_source,
            text_analysis=text_result,
            overall_risk_level=overall,
            warnings=warnings,
        )

    # -------------------------------------------------------------- incident
    def create_incident(self, data: IncidentCreate) -> tuple[IncidentResult, bytes]:
        now = datetime.now(UTC)
        transfer_time = data.transfer_time or now
        if transfer_time.tzinfo is None:  # asumsikan WIB bila tanpa zona waktu
            transfer_time = transfer_time.replace(tzinfo=UTC) - timedelta(hours=7)
        if transfer_time > now + timedelta(minutes=5):
            raise ValueError("Waktu transfer tidak boleh di masa depan.")
        minutes = minutes_since(transfer_time, now)
        steps = build_steps(data.source_bank)
        incident_id, report_id = new_id(), new_id()
        pdf = render_incident_pdf(incident_id, data, transfer_time, steps, generated_at=now)
        expires = now + timedelta(hours=self.settings.report_ttl_hours)

        account_hmac = None
        if data.destination_account and digits_only(data.destination_account):
            account_hmac = hmac_digest(digits_only(data.destination_account), self._secret)
        self.repo.add(
            IncidentRecord(
                id=incident_id,
                transfer_time=transfer_time.replace(tzinfo=None),
                amount_idr=data.amount_idr,
                destination_bank=data.destination_bank,
                account_masked=mask_number(data.destination_account or "") or None,
                account_hmac=account_hmac,
                channel=data.channel,
            )
        )
        self.repo.add(
            ReportRecord(
                id=report_id,
                incident_id=incident_id,
                expires_at=expires.replace(tzinfo=None),
                pdf=pdf,
            )
        )
        missing = [f for f in REQUIRED_FIELDS if getattr(data, f) in (None, "")]
        ANALYSES_TOTAL.labels(kind="incident", risk_level="NA").inc()
        logger.info("incident_created", extra={"incident_id": incident_id, "minutes": minutes})
        return IncidentResult(
            incident_id=incident_id,
            report_id=report_id,
            report_expires_at=expires,
            minutes_since_transfer=minutes,
            urgency=urgency_label(minutes),
            steps=steps,
            missing_fields=missing,
        ), pdf

    def account_report_count(self, account: str) -> int:
        digits = digits_only(account)
        return self.repo.count_account_reports(hmac_digest(digits, self._secret)) if digits else 0
