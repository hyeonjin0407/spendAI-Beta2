#!/usr/bin/env bash
set -e

echo "[Build] Fetching model from Google Drive..."
mkdir -p model

# ✔️ 링크 공유: 'Anyone with the link' + Viewer 로 설정 필수
#    아래 FILE_ID 를 실제 pkl 파일의 ID로 교체하세요.
FILE_ID="YOUR_REAL_FILE_ID"

# gdown으로 '파일 자체'를 받습니다 (페이지/HTML이 아니라).
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O model/spendai_pipeline.pkl

echo "[Build] Model fetched. Verifying with joblib.load ..."

python - <<'PY'
import os, joblib, sys
p = "model/spendai_pipeline.pkl"
print("[Verify] path:", p)
print("[Verify] size:", os.path.getsize(p), "bytes")
try:
    obj = joblib.load(p)
    print("[Verify] type:", type(obj))
    if isinstance(obj, dict):
        print("[Verify] keys:", list(obj.keys()))
        folds = obj.get("folds")
        print("[Verify] folds:", len(folds) if folds else 0)
    print("[Verify] OK")
    sys.exit(0)
except Exception as e:
    print("[Verify] load FAILED:", e)
    sys.exit(1)
PY

ls -lh model/spendai_pipeline.pkl
echo "[Build] Model verified."
