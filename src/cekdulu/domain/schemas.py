"""Kontrak data (Pydantic) yang dipakai API, service, dan demo."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Label(StrEnum):
    NORMAL = "normal"
    PENIPUAN = "penipuan"
    PROMO = "promo"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


RISK_LABEL_ID: dict[RiskLevel, str] = {
    RiskLevel.LOW: "Risiko rendah",
    RiskLevel.MEDIUM: "Perlu waspada",
    RiskLevel.HIGH: "Risiko tinggi",
    RiskLevel.CRITICAL: "Sangat berbahaya",
}


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TacticHit(BaseModel):
    code: str
    label: str
    severity: Severity
    evidence: str = Field(description="Potongan teks yang memicu deteksi.")


class Entities(BaseModel):
    urls: list[str] = Field(default_factory=list)
    phone_numbers: list[str] = Field(default_factory=list, description="Sudah dimasking.")
    account_numbers: list[str] = Field(default_factory=list, description="Sudah dimasking.")
    amounts_idr: list[int] = Field(default_factory=list)
    impersonated: list[str] = Field(default_factory=list)


class TextAnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    channel: str = Field(default="api", max_length=32)


class TextAnalysisResult(BaseModel):
    analysis_id: str
    label: Label
    probabilities: dict[Label, float]
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    risk_label: str
    tactics: list[TacticHit]
    entities: Entities
    explanation: str
    recommended_actions: list[str]
    model_version: str
    explanation_source: str = Field(description="'llm' atau 'template'.")
    latency_ms: float


class QrisFinding(BaseModel):
    code: str
    severity: Severity
    message: str


class QrisInfo(BaseModel):
    is_qris: bool
    crc_valid: bool | None = None
    initiation: str | None = Field(default=None, description="static/dynamic")
    merchant_name: str | None = None
    merchant_city: str | None = None
    nmid: str | None = None
    acquirer_gui: list[str] = Field(default_factory=list)
    merchant_criteria: str | None = None
    mcc: str | None = None
    currency: str | None = None
    country: str | None = None
    amount: str | None = None
    tip_indicator: str | None = None


class QrisAssessment(BaseModel):
    raw_type: str = Field(description="qris | url | text | invalid")
    info: QrisInfo | None = None
    url: str | None = None
    risk_level: RiskLevel
    findings: list[QrisFinding]


class QrisParseRequest(BaseModel):
    payload: str = Field(min_length=1, max_length=1024)
    claimed_merchant: str | None = Field(default=None, max_length=64)
    claimed_amount: int | None = Field(default=None, ge=0)
    context_text: str | None = Field(default=None, max_length=2000)


class ImageAnalysisResult(BaseModel):
    analysis_id: str
    qr_codes: list[QrisAssessment]
    ocr_text: str | None
    ocr_source: str | None
    text_analysis: TextAnalysisResult | None
    overall_risk_level: RiskLevel
    warnings: list[str]


class IncidentCreate(BaseModel):
    transfer_time: datetime | None = Field(
        default=None, description="Waktu transfer (ISO 8601). Default: sekarang."
    )
    amount_idr: int | None = Field(default=None, ge=0)
    source_bank: str | None = Field(default=None, max_length=64)
    destination_bank: str | None = Field(default=None, max_length=64)
    destination_account: str | None = Field(default=None, max_length=40)
    destination_name: str | None = Field(default=None, max_length=100)
    scammer_contact: str | None = Field(default=None, max_length=100)
    channel: str | None = Field(default=None, max_length=50, description="WhatsApp, Instagram, ...")
    chronology: str | None = Field(default=None, max_length=3000)


class GoldenHourStep(BaseModel):
    order: int
    title: str
    detail: str


class IncidentResult(BaseModel):
    incident_id: str
    report_id: str
    report_expires_at: datetime
    minutes_since_transfer: int
    urgency: str
    steps: list[GoldenHourStep]
    missing_fields: list[str]


class BotReply(BaseModel):
    reply_text: str
    report_id: str | None = None
    incident_id: str | None = None
    schedule_reminder: bool = False
