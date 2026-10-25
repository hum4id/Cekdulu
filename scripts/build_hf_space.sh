#!/usr/bin/env bash
# Rakit folder siap-push ke Hugging Face Space (SDK Gradio).
# Pemakaian: bash scripts/build_hf_space.sh && cd dist/hf-space && git init && ... push ke Space
set -euo pipefail
OUT=dist/hf-space
rm -rf "$OUT" && mkdir -p "$OUT/models"
cp -r src "$OUT/src"
cp -r models/baseline "$OUT/models/baseline"
cp demo/app.py "$OUT/app.py"
sed 's#-r ../requirements.txt#-r requirements-core.txt#' demo/requirements.txt > "$OUT/requirements.txt"
cp requirements.txt "$OUT/requirements-core.txt"
echo "tesseract-ocr" > "$OUT/packages.txt"
echo "tesseract-ocr-ind" >> "$OUT/packages.txt"
cat > "$OUT/README.md" <<'MD'
---
title: CekDulu
emoji: 🛡️
colorFrom: indigo
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---
CekDulu — triase penipuan keuangan (teks, gambar/QRIS, golden hour). Alat bantu, bukan jaminan.
MD
find "$OUT" -name "__pycache__" -prune -exec rm -rf {} +
echo "Bundle siap di $OUT"
