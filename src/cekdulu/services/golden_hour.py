"""Panduan respons 'golden hour' pasca-transfer ke penipu.

Dasar: OJK menyatakan dana hasil penipuan dapat dipindahkan pelaku dalam < 1 jam, sementara
mayoritas korban baru melapor setelah ~12 jam. OJK mengimbau korban segera menghubungi bank
atau penyedia jasa pembayaran agar transaksi dapat ditunda/dana diblokir, lalu melapor ke IASC.
Kanal resmi di bawah diambil dari siaran pers OJK SP 09/GKPB/OJK/I/2026.
"""

from __future__ import annotations

from datetime import UTC, datetime

from cekdulu.domain.schemas import GoldenHourStep

OJK_CHANNELS = {
    "IASC (pelaporan & laporan polisi terintegrasi)": "https://iasc.ojk.go.id",
    "Kontak OJK": "157",
    "WhatsApp OJK": "081157157157",
    "Email": "konsumen@ojk.go.id",
}


def minutes_since(transfer_time: datetime, now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    if transfer_time.tzinfo is None:
        transfer_time = transfer_time.replace(tzinfo=UTC)
    return max(0, int((now - transfer_time).total_seconds() // 60))


def urgency_label(minutes: int) -> str:
    if minutes <= 60:
        return "KRITIS: masih dalam 1 jam pertama. Peluang dana tertahan paling besar SEKARANG."
    if minutes <= 12 * 60:
        return "TINGGI: segera lapor. Setiap menit menurunkan peluang pemblokiran dana."
    return "TETAP LAPORKAN: peluang pemulihan menurun, tetapi laporan membantu pemblokiran rekening pelaku."


def build_steps(source_bank: str | None) -> list[GoldenHourStep]:
    bank = source_bank or "bank/e-wallet yang Anda pakai untuk transfer"
    return [
        GoldenHourStep(
            order=1,
            title=f"Telepon {bank} SEKARANG",
            detail="Hubungi call center RESMI (nomor di balik kartu ATM atau di aplikasi resmi, "
            "bukan dari hasil pencarian/medsos). Minta penundaan transaksi dan pemblokiran "
            "rekening tujuan. Catat nomor tiket pengaduan.",
        ),
        GoldenHourStep(
            order=2,
            title="Laporkan ke IASC OJK",
            detail="Isi laporan di https://iasc.ojk.go.id (atau Kontak OJK 157 / WhatsApp "
            "081157157157). Lampirkan PDF laporan kronologi dari CekDulu.",
        ),
        GoldenHourStep(
            order=3,
            title="Amankan bukti",
            detail="Jangan hapus chat. Simpan screenshot percakapan, bukti transfer, nomor/akun "
            "pelaku, dan tautan yang dikirim.",
        ),
        GoldenHourStep(
            order=4,
            title="Amankan akun Anda",
            detail="Jika Anda sempat memberi OTP/PIN atau memasang APK: putuskan internet ponsel, "
            "ganti PIN/password m-banking dari perangkat lain, dan sampaikan ke bank.",
        ),
        GoldenHourStep(
            order=5,
            title="Waspada penipuan lanjutan",
            detail="Jangan transfer lagi dan abaikan tawaran 'jasa pemulihan dana' yang meminta "
            "biaya. Pemulihan resmi hanya melalui bank, IASC, dan kepolisian.",
        ),
    ]


REQUIRED_FIELDS = ("amount_idr", "destination_bank", "destination_account", "transfer_time")
