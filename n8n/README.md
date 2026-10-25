# Workflow n8n: CekDulu Telegram Bot

`workflows/cekdulu-telegram.json` (diuji impor di n8n 2.44.6):

Telegram Trigger → **CekDulu API** (`POST /v1/telegram/process`, retry 2x) → Kirim Balasan →
Ada Laporan PDF? → Unduh PDF → Kirim PDF → Perlu Pengingat? → Tunggu 15 Menit → Kirim Pengingat.
Bila API gagal, cabang error mengirim pesan "layanan sibuk + segera telepon bank".

Kebutuhan: credential **Telegram API** bernama `Telegram CekDulu`, env `CEKDULU_API_URL`,
`CEKDULU_API_KEY`, dan `N8N_BLOCK_ENV_ACCESS_IN_NODE=false`. Langkah lengkap: `docs/DEPLOYMENT.md`.
