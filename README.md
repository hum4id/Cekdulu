# CekDulu

**Sistem triase penipuan keuangan multimodal untuk pencegahan pra-transfer dan respons *golden hour*.**

Final Project AI Engineering Bootcamp (tipe: Machine Learning, dengan modul Text, Image, dan n8n).

> CekDulu adalah alat bantu keputusan, bukan jaminan. Model dapat keliru, terutama terhadap modus baru.
> Pengguna selalu diarahkan ke kanal resmi bank dan OJK.

## Masalah

OJK melalui Indonesia Anti-Scam Centre (IASC) mencatat 432.637 laporan dengan kerugian Rp9,1 triliun
(22 Nov 2024 – 14 Jan 2026), dan baru sekitar Rp161 miliar yang dikembalikan ke korban. OJK juga
menyebut sekitar 80% korban baru melapor ±12 jam setelah kejadian, padahal dana dapat dipindahkan
pelaku dalam kurang dari 1 jam. Masalahnya bukan hanya *deteksi*, tetapi *waktu*.

CekDulu bekerja di dua titik:

| Momen | Yang dilakukan CekDulu |
|---|---|
| **Sebelum transfer** | Pengguna meneruskan chat/screenshot/QR ke bot → skor risiko + taktik manipulasi yang terdeteksi + saran tindakan, dalam hitungan detik. |
| **Sesudah transfer** | Alur *golden hour*: arahkan ke call center resmi bank, buat PDF laporan kronologi untuk bank & IASC, kirim pengingat otomatis 15 menit kemudian (n8n). |

## Arsitektur singkat

```
Telegram ──► n8n (webhook, routing, PDF, pengingat) ──► CekDulu API (FastAPI)
                                                          ├─ Text  : TF-IDF+LR / IndoBERT  ─┐
                                                          ├─ Rules : taktik manipulasi ─────┼─► RiskPolicy ─► verdict
                                                          ├─ Image : QR decode (OpenCV) + QRIS parser (EMVCo, CRC16)
                                                          ├─ OCR   : LLM multimodal → Tesseract (fallback)
                                                          ├─ LLM   : penjelasan bahasa awam (opsional, tidak menentukan verdict)
                                                          └─ PDF + SQLite (PII dimasking/HMAC, TTL laporan)
```

## Hasil evaluasi (model baseline yang dikirim)

Dataset: SMS berbahasa Indonesia berlabel normal/penipuan/promo (Rahmi & Wibisono, UPI 2016), 1.140 baris
setelah deduplikasi. Test set 171 baris (50 penipuan), dievaluasi satu kali.

| Set | Sistem | Macro-F1 | Recall penipuan | Precision penipuan |
|---|---|---|---|---|
| Test in-distribution | TF-IDF word+char + LR, threshold 0,37 | 0,950 (95% CI 0,909–0,984) | 0,90 (0,805–0,979) | 1,00 (45/45; batas bawah Clopper–Pearson ≈0,92) |
| Challenge OOD modus 2025–2026 (n=85) | Model saja | 0,597 | 0,45 (0,29–0,61) | 0,69 |
| Challenge OOD modus 2025–2026 (n=85) | Model + RiskPolicy | 0,736 | 0,78 (0,63–0,89) | 0,80 |

Temuan utama: skor tinggi pada test in-distribution **tidak** bertahan pada modus terkini (APK, QRIS,
investasi bodong, lowongan tugas). Lapisan aturan taktik menaikkan recall OOD dari 0,45 ke 0,78, tetapi
challenge set ini disusun pengembang (bukan pesan nyata), sehingga angka OOD adalah indikasi pergeseran
distribusi, bukan akurasi lapangan.

Latensi inference model (CPU, 1 pesan): p50 ≈2 ms, p95 ≈3 ms. End-to-end `/v1/analyze/text` tanpa LLM ≈20 ms.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
make install-dev                                        # dependensi + paket editable
make train                                              # split data (cek SHA-256) + latih + evaluasi test
make evaluate                                           # evaluasi challenge set OOD
make test                                               # 70 test unit + integrasi
cp .env.example .env                                    # isi API key bila perlu
make run                                                # API di http://localhost:8000/docs
make demo                                               # UI Gradio di http://localhost:7860
```

Contoh panggilan:

```bash
curl -X POST localhost:8000/v1/analyze/text -H 'Content-Type: application/json' \
  -d '{"text":"Paket Anda tertahan, cek foto di aplikasi berikut LIHAT_FOTO_PAKET.apk"}'
# -> risk_level CRITICAL, tactics [APP_INSTALL, URGENCY], explanation, recommended_actions
```

Di mode `development` tanpa `CEKDULU_API_KEYS`, autentikasi dimatikan. Di `production`, API key wajib.

## Endpoint

| Method | Path | Fungsi |
|---|---|---|
| POST | `/v1/analyze/text` | Skor risiko teks, taktik, entitas (termasking), penjelasan, saran |
| POST | `/v1/analyze/image` | Multipart gambar (+ `caption`): decode QR, nilai QRIS/URL, OCR, analisis teks |
| POST | `/v1/qris/parse` | Validasi payload QRIS (CRC, NMID, merchant/nominal vs klaim, konteks "terima dana") |
| POST | `/v1/incidents` | Panduan golden hour + PDF laporan (berlaku 24 jam) |
| GET | `/v1/reports/{id}.pdf` | Unduh PDF laporan |
| POST | `/v1/telegram/process` | Proses satu Telegram Update (dipanggil n8n) |
| GET | `/healthz`, `/readyz`, `/metrics` | Liveness, readiness (model+DB), Prometheus |

## Struktur repositori

```
cekdulu/
├── src/cekdulu/
│   ├── api/            # FastAPI: app factory, routes, auth/rate limit, DI container
│   ├── domain/         # Kontrak data Pydantic
│   ├── ml/             # preprocessing, classifier (sklearn/transformers), tactics, risk policy
│   ├── qris/           # parser EMVCo + CRC16 + penilaian risiko QR
│   ├── vision/         # validasi gambar, decode QR (OpenCV), OCR berantai
│   ├── llm/            # provider Gemini/OpenAI-compatible, retry, fallback model, prompt
│   ├── services/       # orkestrasi analisis, golden hour, PDF, handler Telegram
│   ├── storage/        # SQLAlchemy (SQLite/PostgreSQL), retensi
│   ├── training/       # split deterministik, metrik, bootstrap CI, tuning threshold
│   ├── security/       # masking PII & HMAC
│   └── bot/            # bot Telegram long-polling (cadangan tanpa n8n)
├── scripts/            # prepare_data, train_baseline, train_indobert, evaluate_ood, build_hf_space
├── configs/            # YAML konfigurasi data & training
├── data/               # raw (dengan SHA-256), processed (split), ood (challenge set)
├── models/baseline/    # model.joblib + metadata.json (versi, threshold, hash data)
├── reports/baseline/   # metrics.json, ood_metrics.json, confusion_matrix.png, test_errors.csv
├── n8n/workflows/      # workflow Telegram siap import
├── demo/               # Gradio (Hugging Face Spaces)
├── notebooks/          # Colab untuk fine-tune IndoBERT
├── tests/              # unit + integrasi
├── Dockerfile, docker-compose.yml, Makefile, .github/workflows/ci.yml
```

## Sumber data masalah

- OJK, Siaran Pers SP 09/GKPB/OJK/I/2026 (kanal resmi IASC, Kontak 157, WA 081157157157).
- detikFinance (22 Jan 2026) & Kontan (22 Jan 2026): kerugian Rp9,1 T, 432.637 laporan, pengembalian Rp161 M, modus teratas, 80% korban melapor setelah 12 jam.
- Katadata/iNews (4–5 Agu 2026): 1.379 laporan penipuan berbasis AI hingga akhir Juni 2026.

## Lisensi

Kode: MIT. Dataset mengikuti ketentuan pemilik aslinya.
