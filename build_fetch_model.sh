#!/usr/bin/env bash
set -e

echo "[Build] Fetching model from Google Drive..."
mkdir -p model
# ← 네 공유 URL을 그대로 넣으세요 (파일 ID 대신 URL도 OK)
gdown --fuzzy "https://drive.google.com/file/d/17GUwwi3f8W6zoBESaKhMJ3wrpbFMZSs1/view?usp=drive_link" -O model/spendai_pipeline.pkl

echo "[Build] Model fetched successfully:"
ls -lh model/spendai_pipeline.pkl
