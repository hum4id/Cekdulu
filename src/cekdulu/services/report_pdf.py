"""Generator PDF 'Laporan Kronologi' siap lampir ke bank/IASC (ReportLab, tanpa dependensi sistem)."""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta, timezone
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from cekdulu.domain.schemas import GoldenHourStep, IncidentCreate
from cekdulu.services.golden_hour import OJK_CHANNELS

WIB = timezone(timedelta(hours=7), name="WIB")


def _fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(WIB).strftime("%d-%m-%Y %H:%M WIB")


def _fmt_idr(amount: int | None) -> str:
    return "-" if amount is None else "Rp" + f"{amount:,}".replace(",", ".")


def _p(text: str | None, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text or "-").replace("\n", "<br/>"), style)


def render_incident_pdf(
    incident_id: str,
    data: IncidentCreate,
    transfer_time: datetime,
    steps: list[GoldenHourStep],
    generated_at: datetime | None = None,
) -> bytes:
    generated_at = generated_at or datetime.now(UTC)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Laporan Kronologi Dugaan Penipuan",
        author="CekDulu",
    )
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=10)

    rows = [
        ["Nomor referensi", incident_id],
        ["Waktu transfer", _fmt_dt(transfer_time)],
        ["Nominal", _fmt_idr(data.amount_idr)],
        ["Bank/e-wallet pengirim", data.source_bank or "-"],
        ["Bank/e-wallet tujuan", data.destination_bank or "-"],
        ["Rekening/akun tujuan", data.destination_account or "-"],
        ["Nama pemilik rekening tujuan", data.destination_name or "-"],
        ["Kontak/akun pelaku", data.scammer_contact or "-"],
        ["Media/kanal penipuan", data.channel or "-"],
    ]
    table = Table([[_p(k, body), _p(v, body)] for k, v in rows], colWidths=[58 * mm, 116 * mm])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7C3")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F3F6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story = [
        Paragraph("Laporan Kronologi Dugaan Penipuan Transaksi Keuangan", h1),
        _p(f"Disusun: {_fmt_dt(generated_at)}", small),
        Spacer(1, 6),
        Paragraph("Data Transaksi", h2),
        table,
        Paragraph("Kronologi (keterangan pelapor)", h2),
        _p(data.chronology, body),
        Paragraph("Langkah yang harus segera dilakukan", h2),
    ]
    for step in steps:
        story.append(_p(f"{step.order}. {step.title} — {step.detail}", body))
    story.append(Paragraph("Kanal resmi pelaporan OJK", h2))
    for name, value in OJK_CHANNELS.items():
        story.append(_p(f"{name}: {value}", body))
    story += [
        Spacer(1, 10),
        _p(
            "Dokumen ini disusun otomatis oleh CekDulu berdasarkan keterangan pelapor dan BUKAN "
            "dokumen resmi OJK, bank, maupun kepolisian. Keakuratan isi menjadi tanggung jawab "
            "pelapor. Simpan dokumen ini secara pribadi karena berisi data transaksi.",
            small,
        ),
    ]
    doc.build(story)
    return buf.getvalue()
