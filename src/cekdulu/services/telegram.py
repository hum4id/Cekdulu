"""Penerjemah Telegram Update -> aksi CekDulu -> balasan teks (stateless).

Dipakai oleh: (1) endpoint ``/v1/telegram/process`` yang dipanggil workflow n8n, dan
(2) ``python -m cekdulu.bot`` (long-polling) sebagai jalur cadangan tanpa n8n/webhook publik.
"""

from __future__ import annotations

import contextlib
import html
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from cekdulu.domain.schemas import (
    BotReply,
    ImageAnalysisResult,
    IncidentCreate,
    QrisAssessment,
    RiskLevel,
    TextAnalysisResult,
)
from cekdulu.ml.tactics import parse_amount_idr
from cekdulu.services.analysis import AnalysisService
from cekdulu.vision.image_io import InvalidImageError

logger = logging.getLogger(__name__)

ICON = {RiskLevel.LOW: "🟢", RiskLevel.MEDIUM: "🟡", RiskLevel.HIGH: "🟠", RiskLevel.CRITICAL: "🔴"}

HELP_TEXT = (
    "<b>CekDulu</b>: cek dulu sebelum transfer.\n\n"
    "• Teruskan/tempel <b>pesan mencurigakan</b> → saya nilai risikonya.\n"
    "• Kirim <b>foto/screenshot</b> (chat, poster, stiker QRIS) → saya baca QR & teksnya.\n"
    "• Sudah terlanjur transfer? Kirim <b>/lapor</b> lalu isi format berikut dalam SATU pesan:\n\n"
    "<code>/lapor\nwaktu: 14:05\nnominal: 2.500.000\nbank asal: BRI\nbank tujuan: BCA\n"
    "rekening: 1234567890\nnama: Nama Pemilik\nkontak pelaku: 0812xxxx\nmedia: WhatsApp\n"
    "kronologi: ceritakan singkat</code>\n\n"
    "⚠️ CekDulu adalah alat bantu, bukan jaminan. Selalu verifikasi lewat kanal resmi."
)

_FIELD_MAP = {
    "waktu": "transfer_time",
    "jam": "transfer_time",
    "nominal": "amount_idr",
    "jumlah": "amount_idr",
    "bank asal": "source_bank",
    "bank pengirim": "source_bank",
    "bank tujuan": "destination_bank",
    "bank": "destination_bank",
    "rekening": "destination_account",
    "no rek": "destination_account",
    "norek": "destination_account",
    "nama": "destination_name",
    "nama pemilik": "destination_name",
    "kontak pelaku": "scammer_contact",
    "kontak": "scammer_contact",
    "media": "channel",
    "kanal": "channel",
    "kronologi": "chronology",
    "cerita": "chronology",
}
_TIME_RE = re.compile(r"^(\d{1,2})[:.](\d{2})$")


def parse_report_form(text: str, now: datetime | None = None) -> IncidentCreate:
    """Parse format ``kunci: nilai`` per baris. 'waktu' dapat berupa HH:MM (hari ini, WIB)
    atau ISO 8601. Baris tanpa kunci dianggap lanjutan kronologi."""
    now = now or datetime.now(UTC)
    values: dict[str, Any] = {}
    last_key: str | None = None
    body = re.sub(r"^\s*/lapor\S*", "", text, count=1, flags=re.IGNORECASE)
    for line in body.splitlines():
        if ":" in line:
            raw_key, _, raw_val = line.partition(":")
            key = _FIELD_MAP.get(raw_key.strip().lower())
            if key:
                values[key] = raw_val.strip()
                last_key = key
                continue
        if last_key == "chronology" and line.strip():
            values["chronology"] = f"{values['chronology']}\n{line.strip()}"

    if "amount_idr" in values:
        values["amount_idr"] = parse_amount_idr(values["amount_idr"].replace(" ", ""))
    if "transfer_time" in values:
        raw = values.pop("transfer_time")
        match = _TIME_RE.match(raw)
        if match:
            wib_now = now.astimezone(UTC) + timedelta(hours=7)
            local = wib_now.replace(
                hour=int(match.group(1)), minute=int(match.group(2)), second=0, microsecond=0
            )
            if local > wib_now:
                local -= timedelta(days=1)
            values["transfer_time"] = (local - timedelta(hours=7)).replace(tzinfo=UTC)
        else:
            with contextlib.suppress(ValueError):
                values["transfer_time"] = datetime.fromisoformat(raw)
    return IncidentCreate(**{k: v for k, v in values.items() if v not in (None, "")})


def format_text_result(r: TextAnalysisResult) -> str:
    lines = [
        f"{ICON[r.risk_level]} <b>{html.escape(r.risk_label)}</b> (skor {r.risk_score:.0%})",
        "",
        html.escape(r.explanation),
    ]
    if r.tactics:
        lines += ["", "<b>Ciri yang terdeteksi:</b>"]
        lines += [f"• {html.escape(t.label)}" for t in r.tactics[:5]]
    lines += ["", "<b>Saran:</b>"] + [f"• {html.escape(a)}" for a in r.recommended_actions]
    return "\n".join(lines)


def format_qr_result(q: QrisAssessment) -> str:
    head = f"{ICON[q.risk_level]} <b>QR ({q.raw_type})</b>"
    if q.info and q.info.merchant_name:
        head += f": merchant <b>{html.escape(q.info.merchant_name)}</b>"
        if q.info.merchant_city:
            head += f", {html.escape(q.info.merchant_city)}"
    return "\n".join([head] + [f"• {html.escape(f.message)}" for f in q.findings])


def format_image_result(r: ImageAnalysisResult) -> str:
    parts = [format_qr_result(q) for q in r.qr_codes]
    if r.text_analysis:
        parts.append(format_text_result(r.text_analysis))
    parts += [f"ℹ️ {html.escape(w)}" for w in r.warnings]
    return "\n\n".join(parts)


class TelegramUpdateHandler:
    def __init__(self, service: AnalysisService, bot_token: str | None) -> None:
        self._svc = service
        self._token = bot_token
        self._http = httpx.Client(timeout=20)

    def _download_file(self, file_id: str) -> bytes:
        if not self._token:
            raise RuntimeError("CEKDULU_TELEGRAM_BOT_TOKEN belum diisi; foto tidak dapat diunduh.")
        base = f"https://api.telegram.org/bot{self._token}"
        meta = self._http.get(f"{base}/getFile", params={"file_id": file_id}).json()
        if not meta.get("ok"):
            raise RuntimeError("Telegram getFile gagal")
        path = meta["result"]["file_path"]
        resp = self._http.get(f"https://api.telegram.org/file/bot{self._token}/{path}")
        resp.raise_for_status()
        return resp.content

    def handle(self, update: dict[str, Any]) -> BotReply:
        msg = update.get("message") or update.get("edited_message") or {}
        text: str = (msg.get("text") or "").strip()
        caption: str | None = msg.get("caption")

        if text.startswith(("/start", "/help", "/bantuan")):
            return BotReply(reply_text=HELP_TEXT)
        if text.lower().startswith("/lapor"):
            return self._handle_report(text)

        photo_id = None
        if msg.get("photo"):
            photo_id = max(msg["photo"], key=lambda p: p.get("file_size", 0))["file_id"]
        elif (msg.get("document") or {}).get("mime_type", "").startswith("image/"):
            photo_id = msg["document"]["file_id"]
        if photo_id:
            try:
                data = self._download_file(photo_id)
                result = self._svc.analyze_image(data, caption=caption, channel="telegram")
                return BotReply(reply_text=format_image_result(result))
            except (InvalidImageError, RuntimeError, httpx.HTTPError) as exc:
                logger.warning("telegram_image_failed", extra={"error": str(exc)})
                return BotReply(reply_text=f"Gambar tidak dapat diproses: {html.escape(str(exc))}")

        if text:
            result = self._svc.analyze_text(text, channel="telegram")
            return BotReply(reply_text=format_text_result(result))
        return BotReply(reply_text="Kirim teks pesan atau foto yang ingin dicek. Ketik /help.")

    def _handle_report(self, text: str) -> BotReply:
        if text.strip().lower() == "/lapor":
            return BotReply(
                reply_text=(
                    "🚨 <b>Jika baru saja transfer: TELEPON CALL CENTER RESMI BANK ANDA SEKARANG</b> "
                    "(nomor di balik kartu ATM/aplikasi resmi) untuk minta penundaan transaksi.\n\n"
                    "Lalu kirim data dalam satu pesan agar saya buatkan PDF laporan untuk bank & IASC:\n\n"
                    + HELP_TEXT.split("\n\n", 2)[2]
                )
            )
        try:
            data = parse_report_form(text)
            result, _ = self._svc.create_incident(data)
        except ValueError as exc:
            return BotReply(reply_text=f"Format laporan belum valid: {html.escape(str(exc))}")
        lines = [
            f"🚨 <b>{html.escape(result.urgency)}</b>",
            f"Sejak transfer: ±{result.minutes_since_transfer} menit",
            "",
        ]
        lines += [
            f"<b>{s.order}. {html.escape(s.title)}</b>\n{html.escape(s.detail)}"
            for s in result.steps
        ]
        if data.destination_account:
            count = self._svc.account_report_count(data.destination_account)
            if count > 1:
                lines += ["", f"⚠️ Rekening tujuan ini sudah dilaporkan {count} kali di CekDulu."]
        if result.missing_fields:
            lines += ["", "Data belum lengkap: " + ", ".join(result.missing_fields)]
        lines += [
            "",
            "📄 PDF laporan kronologi terlampir (berlaku "
            f"{self._svc.settings.report_ttl_hours} jam).",
        ]
        return BotReply(
            reply_text="\n".join(lines),
            report_id=result.report_id,
            incident_id=result.incident_id,
            schedule_reminder=result.minutes_since_transfer <= 12 * 60,
        )
