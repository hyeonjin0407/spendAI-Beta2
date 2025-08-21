#!/usr/bin/env bash
set -e

echo "[Build] Fetching model from Google Drive..."
mkdir -p model

# Google Drive 파일 ID로 다운로드
gdown --id 17GUwwi3f8W6zoBESaKhMJ3wrpbFMZSs1 -O model/spendai_pipeline.pkl

echo "[Build] Model fetched successfully:"
ls -lh model/spendai_pipeline.pkl
