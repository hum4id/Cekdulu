"""Demo web CekDulu (Gradio) — dipakai untuk Hugging Face Spaces dan presentasi.

Memanggil ``AnalysisService`` secara in-process (tanpa HTTP), sehingga satu Space cukup.
Jalankan lokal: ``python demo/app.py`` lalu buka http://localhost:7860
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Di repo: demo/app.py -> root = parent. Di bundle HF Space: app.py berada di root.
ROOT = HERE if (HERE / "src").exists() else HERE.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CEKDULU_MODEL_DIR", str(ROOT / "models" / "baseline"))
os.environ.setdefault("CEKDULU_DATABASE_URL", f"sqlite:///{tempfile.gettempdir()}/cekdulu-demo.db")
os.environ.setdefault("CEKDULU_LOG_JSON", "false")

import gradio as gr  # noqa: E402

from cekdulu.api.container import build_container  # noqa: E402
from cekdulu.config import Settings  # noqa: E402
from cekdulu.domain.schemas import IncidentCreate  # noqa: E402
from cekdulu.services.telegram import ICON  # noqa: E402

container = build_container(Settings())
svc = container.service

DISCLAIMER = (
    "⚠️ **CekDulu adalah alat bantu**, bukan jaminan. Hasil dapat keliru, terutama untuk modus "
    "baru. Selalu verifikasi lewat kanal resmi. Jangan memasukkan PIN/OTP/password ke sini."
)
EXAMPLES = [
    "Paket Anda tertahan. Cek foto paket di aplikasi berikut: LIHAT_FOTO_PAKET.apk",
    "Halo kak, saya CS resmi bank. Mohon sebutkan kode OTP yang barusan masuk ya",
    "Selamat! Anda pemenang undian. Transfer biaya admin pajak hadiah Rp2.500.000 segera",
    "Nikmati kuota 30GB 30 hari hanya Rp55 ribu. Beli di aplikasi resmi. S&K berlaku",
    "Nanti jam 7 jadi kumpul di rumah Dina kan? Aku bawa martabak",
]


def check_text(text: str) -> tuple[str, dict]:
    if not text or not text.strip():
        return "Masukkan teks pesan terlebih dahulu.", {}
    r = svc.analyze_text(text, channel="demo")
    md = [f"## {ICON[r.risk_level]} {r.risk_label} — skor {r.risk_score:.0%}", r.explanation]
    if r.tactics:
        md.append(
            "**Ciri yang terdeteksi**\n"
            + "\n".join(f"- {t.label} — _“{t.evidence}”_" for t in r.tactics)
        )
    md.append("**Saran**\n" + "\n".join(f"- {a}" for a in r.recommended_actions))
    md.append(
        f"<sub>model: {r.model_version} · penjelasan: {r.explanation_source} · "
        f"{r.latency_ms:.0f} ms</sub>"
    )
    return "\n\n".join(md), {k.value: v for k, v in r.probabilities.items()}


def check_image(image, caption: str) -> str:
    if image is None:
        return "Unggah gambar terlebih dahulu."
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    r = svc.analyze_image(buf.getvalue(), caption=caption or None, channel="demo")
    parts = [f"## {ICON[r.overall_risk_level]} Risiko keseluruhan: {r.overall_risk_level.value}"]
    for q in r.qr_codes:
        head = f"### QR ({q.raw_type})"
        if q.info and q.info.merchant_name:
            head += (
                f" — merchant **{q.info.merchant_name}** ({q.info.merchant_city or '-'}), "
                f"NMID {q.info.nmid or '-'}, CRC {'valid' if q.info.crc_valid else 'TIDAK valid'}"
            )
        parts.append(head + "\n" + "\n".join(f"- {f.message}" for f in q.findings))
    if r.ocr_text:
        parts.append(f"**Teks terbaca ({r.ocr_source})**\n```\n{r.ocr_text[:800]}\n```")
    if r.text_analysis:
        t = r.text_analysis
        parts.append(f"**Analisis teks:** {t.risk_label} ({t.risk_score:.0%}) — {t.explanation}")
    parts += [f"_ℹ️ {w}_" for w in r.warnings]
    return "\n\n".join(parts)


def golden_hour(amount, source_bank, dest_bank, account, name, contact, channel, chronology):
    data = IncidentCreate(
        amount_idr=int(amount) if amount else None,
        source_bank=source_bank or None,
        destination_bank=dest_bank or None,
        destination_account=account or None,
        destination_name=name or None,
        scammer_contact=contact or None,
        channel=channel or None,
        chronology=chronology or None,
    )
    result, pdf = svc.create_incident(data)
    path = Path(tempfile.gettempdir()) / f"laporan-cekdulu-{result.report_id[:8]}.pdf"
    path.write_bytes(pdf)
    md = [f"## 🚨 {result.urgency}"]
    md += [f"**{s.order}. {s.title}**  \n{s.detail}" for s in result.steps]
    if result.missing_fields:
        md.append("Data belum lengkap: " + ", ".join(result.missing_fields))
    return "\n\n".join(md), str(path)


with gr.Blocks(title="CekDulu") as demo:
    gr.Markdown(
        "# CekDulu\nCek dulu sebelum transfer: deteksi penipuan dari pesan, gambar/QRIS, "
        "dan panduan golden hour bila sudah terlanjur transfer."
    )
    gr.Markdown(DISCLAIMER)
    with gr.Tab("Cek Pesan"):
        inp = gr.Textbox(lines=6, label="Tempel pesan mencurigakan")
        btn = gr.Button("Cek risiko", variant="primary")
        out_md = gr.Markdown()
        out_lbl = gr.Label(label="Probabilitas model", num_top_classes=3)
        btn.click(check_text, inp, [out_md, out_lbl])
        gr.Examples(EXAMPLES, inp)
    with gr.Tab("Cek Gambar / QRIS"):
        img = gr.Image(type="pil", label="Foto stiker QRIS, poster, atau screenshot chat")
        cap = gr.Textbox(
            label="Konteks (opsional): apa kata pengirim?",
            placeholder="mis. 'scan QR ini untuk menerima refund'",
        )
        btn2 = gr.Button("Analisis gambar", variant="primary")
        out2 = gr.Markdown()
        btn2.click(check_image, [img, cap], out2)
    with gr.Tab("Sudah Terlanjur Transfer"):
        gr.Markdown(
            "**Langkah pertama: telepon call center RESMI bank Anda sekarang.** "
            "Isi data di bawah untuk membuat PDF laporan kronologi."
        )
        with gr.Row():
            amount = gr.Number(label="Nominal (Rp)", precision=0)
            source_bank = gr.Textbox(label="Bank/e-wallet pengirim")
            dest_bank = gr.Textbox(label="Bank/e-wallet tujuan")
        with gr.Row():
            account = gr.Textbox(label="Rekening/akun tujuan")
            name = gr.Textbox(label="Nama pemilik rekening tujuan")
            contact = gr.Textbox(label="Kontak/akun pelaku")
        channel = gr.Textbox(label="Media penipuan (WhatsApp, Instagram, ...)")
        chronology = gr.Textbox(lines=4, label="Kronologi singkat")
        btn3 = gr.Button("Buat panduan & PDF", variant="stop")
        out3 = gr.Markdown()
        pdf_out = gr.File(label="PDF laporan")
        btn3.click(
            golden_hour,
            [amount, source_bank, dest_bank, account, name, contact, channel, chronology],
            [out3, pdf_out],
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # noqa: S104 - container/Spaces butuh bind semua interface
        server_port=int(os.environ.get("PORT", "7860")),
    )
