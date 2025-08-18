# main.py — SpendAI (model + heuristic + mini UI)
# 저장 형식: 금액(원),제품명,당시 기분,후회 여부,구매 이유  (딱 5컬럼)
# 저장 경로: C:\Users\USER\Desktop\spendAI\spendAI\xpend\data\purchase_data.csv

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import json, csv, os, math, traceback

import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ROOT = Path(__file__).resolve().parent

# ★ 요구 경로 고정
DATA_DIR = ROOT / "spendAI" / "xpend" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = DATA_DIR / "purchase_data.csv"
PREF_FILE = DATA_DIR / "user_preferences.json"

MODEL_DIR = ROOT / "model"
PIPELINE_PATH = MODEL_DIR / "spendai_pipeline.pkl"
MODEL_PATH    = MODEL_DIR / "spendai_model.pkl"
PREPROC_PATH  = MODEL_DIR / "spendai_preprocessor.pkl"

pipeline = None
model = None
preproc = None
model_loaded = False
model_mode = "heuristic"  # "pipeline" | "model+preproc" | "model_only" | "heuristic"

def _log(msg: str):
    print(f"[SpendAI] {msg}", flush=True)

# -------- Model loading --------
try:
    if PIPELINE_PATH.exists():
        pipeline = joblib.load(PIPELINE_PATH)
        model_loaded = True
        model_mode = "pipeline"
        _log(f"Loaded pipeline: {PIPELINE_PATH}")
    elif MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        if PREPROC_PATH.exists():
            preproc = joblib.load(PREPROC_PATH)
            model_mode = "model+preproc"
            _log(f"Loaded model + preproc: {MODEL_PATH}, {PREPROC_PATH}")
        else:
            model_mode = "model_only"
            _log(f"Loaded model only: {MODEL_PATH}")
        model_loaded = True
    else:
        _log("No model files found. Using heuristic fallback.")
except Exception as e:
    _log(f"Model load failed: {e}\n{traceback.format_exc()}")
    pipeline = None
    model = None
    preproc = None
    model_loaded = False
    model_mode = "heuristic"

# -------- Heuristic with alignment --------
def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

REASONS_IMPULSIVE = {"즉흥 구매", "스트레스 해소용", "온라인 광고 보고"}
REASONS_PLANNED   = {"필요", "계획된 지출", "기념일 선물로"}
REASONS_SOCIAL    = {"친구 추천으로"}
REASONS_PRICE     = {"가격이 좋아서"}
REASONS_WISHLIST  = {"평소에 사고 싶었음", "새로운 취미 시작해서"}

CATS_UP   = {"전자제품", "전자기기", "의류"}
CATS_DOWN = {"식료품", "생활용품"}

PREF_ALIGN = {
    "planned_spending": REASONS_PLANNED,
    "electronics_lover": {"필요", "평소에 사고 싶었음", "계획된 지출"},
    "hobby_spender": {"평소에 사고 싶었음", "새로운 취미 시작해서", "기념일 선물로"},
    "food_lover": {"필요", "가격이 좋아서"},
    "budget_shopper": {"가격이 좋아서", "필요"},
    "gift_giver": {"기념일 선물로"},
}
PREF_MISALIGN = {
    "planned_spending": REASONS_IMPULSIVE,
    "electronics_lover": {"즉흥 구매", "온라인 광고 보고"},
    "hobby_spender": {"온라인 광고 보고"},
    "food_lover": {"즉흥 구매"},
    "budget_shopper": {"즉흥 구매", "스트레스 해소용"},
    "gift_giver": {"즉흥 구매"},
}

def _alignment_delta(user_type: str, reason: str, category: str) -> float:
    ut = (user_type or "").strip()
    r  = (reason or "").strip()
    c  = (category or "").strip()

    delta = 0.0
    if ut in PREF_ALIGN and r in PREF_ALIGN[ut]:
        delta -= 0.25
    if ut in PREF_MISALIGN and r in PREF_MISALIGN[ut]:
        delta += 0.25

    if ut == "electronics_lover":
        if c in {"전자제품", "전자기기"}: delta -= 0.08
        if r in REASONS_IMPULSIVE:        delta += 0.05
    elif ut == "hobby_spender":
        if c in {"취미"}:                 delta -= 0.06
    elif ut == "food_lover":
        if c in {"식료품", "외식"}:       delta -= 0.06
    return delta

def _heuristic_regret(p: dict) -> float:
    price     = float(p.get("금액(원)", 0))
    mood      = float(p.get("당시 기분", 3))  # 1~5
    reason    = str(p.get("구매 이유", "")).strip()
    category  = str(p.get("항목", "")).strip()
    day       = str(p.get("요일", "")).strip()
    month     = int(p.get("월", 1))
    user_type = str(p.get("user_type", "") or "").strip()

    z = 0.0
    z += (price - 100000.0) / 100000.0 * 0.9
    z += (3.0 - mood) * 0.35
    if reason in REASONS_IMPULSIVE: z += 0.5
    if reason in REASONS_PLANNED:   z -= 0.3
    if category in CATS_UP:         z += 0.2
    if category in CATS_DOWN:       z -= 0.15
    if day in {"금요일","토요일"}:  z += 0.1
    if month in {11,12}:            z += 0.1

    if user_type == "planned_spending": z -= 0.1
    elif user_type in {"hobby_spender","electronics_lover"}: z += 0.1

    z += _alignment_delta(user_type, reason, category)
    return max(0.02, min(0.98, _sigmoid(z)))

_FEATURES = ["금액(원)","당시 기분","항목","구매 이유","요일","월","user_type"]

def _predict_with_model(payload: dict) -> float:
    try:
        if model_loaded:
            if model_mode == "pipeline" and pipeline is not None:
                df = pd.DataFrame([{k: payload.get(k) for k in _FEATURES}])
                y = pipeline.predict(df)
                p = float(y[0])
                return max(0.0, min(1.0, p))
            elif model_mode == "model+preproc" and model is not None and preproc is not None:
                df = pd.DataFrame([{k: payload.get(k) for k in _FEATURES}])
                X = preproc.transform(df)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    p = float(proba[1] if proba.shape[1] > 1 else proba)
                else:
                    y = model.predict(X)
                    p = float(y[0]) if isinstance(y[0], (int,float)) else 0.0
                return max(0.0, min(1.0, p))
            elif model_mode == "model_only" and model is not None:
                cats = {"전자제품": 1, "전자기기": 1, "의류": 1, "식료품": -1, "생활용품": -1}
                reasons = {
                    "즉흥 구매": 1, "스트레스 해소용": 1, "온라인 광고 보고": 1,
                    "필요": -1, "계획된 지출": -1, "기념일 선물로": -1
                }
                days = {"금요일": 1, "토요일": 1}
                price = float(payload.get("금액(원)", 0))
                mood  = float(payload.get("당시 기분", 3))
                month = int(payload.get("월", 1))
                c = cats.get(str(payload.get("항목","")), 0)
                r = reasons.get(str(payload.get("구매 이유","")), 0)
                d = days.get(str(payload.get("요일","")), 0)
                u = 1 if str(payload.get("user_type","")) in {"hobby_spender","electronics_lover"} else (-1 if str(payload.get("user_type",""))=="planned_spending" else 0)
                Xa = np.array([[price, mood, month, c, r, d, u]], dtype=float)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(Xa)[0]
                    p = float(proba[1] if proba.shape[1] > 1 else proba)
                else:
                    y = model.predict(Xa)
                    p = float(y[0]) if isinstance(y[0], (int,float)) else 0.0
                return max(0.0, min(1.0, p))
    except Exception as e:
        _log(f"Predict failed with model: {e}\n{traceback.format_exc()} — fallback to heuristic.")
    return _heuristic_regret(payload)

# -------- APIs --------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(model_loaded),
        "mode": model_mode,
        "data_path": str(DATA_FILE),
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    required = ['금액(원)', '당시 기분', '항목', '구매 이유', '요일', '월', 'user_type']
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': f"Missing field(s): {', '.join(missing)}"}), 400
    p = _predict_with_model(data)
    return jsonify({
        "regret_probability": float(p),
        "user_type": data.get("user_type"),
        "note": "model" if model_loaded else "heuristic"
    })

@app.route("/save_data", methods=["POST"])
def save_data():
    """
    CSV는 오직 5컬럼만:
      금액(원),제품명,당시 기분,후회 여부,구매 이유
    """
    data = request.get_json(force=True, silent=True) or {}
    fieldnames = ['금액(원)', '제품명', '당시 기분', '후회 여부', '구매 이유']
    missing = [k for k in fieldnames if k not in data]
    if missing:
        return jsonify({'error': f"Missing field(s): {', '.join(missing)}"}), 400

    is_new = not DATA_FILE.exists()
    with DATA_FILE.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        row = {k: data.get(k) for k in fieldnames}
        w.writerow(row)

    return jsonify({"status": "success", "path": str(DATA_FILE)})

@app.route("/get_preferences", methods=["GET"])
def get_preferences():
    if PREF_FILE.exists():
        return jsonify(json.loads(PREF_FILE.read_text(encoding="utf-8")))
    return jsonify({"계획 지출 선호":0.5, "음식 선호":0.5, "즉흥 구매 성향":0.5, "저가 선호":0.5})

@app.route("/set_preferences", methods=["POST"])
def set_preferences():
    prefs = request.get_json(force=True, silent=True) or {}
    PREF_FILE.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"status": "success"})

# -------- Mini UI (정렬 개선) --------
@app.route("/", methods=["GET"])
def index_page():
    html = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>SpendAI - Mini UI</title>
  <style>
    :root { --bg:#0b0c10; --card:#16171c; --ink:#e6e6e6; --acc:#4da3ff; --mut:#9aa0a6; --field:#0f1116; --line:#2a2d33; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Apple SD Gothic Neo, Noto Sans KR, "맑은 고딕", sans-serif; background:var(--bg); color:var(--ink);}
    .wrap { max-width: 960px; margin: 32px auto; padding: 0 16px; }
    .card { background:var(--card); border-radius:16px; padding:24px; box-shadow: 0 4px 20px rgba(0,0,0,.35); }
    h1 { font-size: 24px; margin:0 0 12px; }
    h2 { font-size: 18px; margin:28px 0 12px; color: var(--acc); }
    label { display:block; font-weight:600; margin:0 0 6px; }
    .grid-2 { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:14px; }
    .grid-3 { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:14px; }
    .row { margin-bottom:16px; }
    input, select { width:100%; height:44px; padding:10px 12px; border-radius:10px; border:1px solid var(--line); background:var(--field); color:var(--ink); outline:none; appearance:none; }
    input[type="range"] { height:auto; }
    .btn { cursor:pointer; border:none; padding:12px 16px; border-radius:10px; font-weight:700; }
    .btn.primary { background:var(--acc); color:#0b0c10; }
    .btn.ghost { background:transparent; border:1px solid var(--line); color: var(--ink); }
    .actions { display:flex; gap:10px; margin-top:6px; flex-wrap:wrap; align-items:center; }
    .result { margin-top:12px; padding:12px; background:var(--field); border:1px solid var(--line); border-radius:10px; white-space:pre-wrap; }
    .muted { color: var(--mut); font-size: 12px; }
    @media (max-width: 720px) {
      .grid-2, .grid-3 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>SpendAI – Mini UI</h1>
      <div class="muted">모델 모드/저장 경로는 /health에서 확인 가능</div>

      <h2>예측 (POST /predict)</h2>
      <div class="grid-3 row">
        <div>
          <label>금액(원)</label>
          <input id="price" type="number" value="100000" min="0" step="1000"/>
        </div>
        <div>
          <label>당시 기분</label>
          <select id="mood">
            <option value="1">매우나쁨 (1)</option>
            <option value="2">나쁨 (2)</option>
            <option value="3" selected>보통 (3)</option>
            <option value="4">좋음 (4)</option>
            <option value="5">행복 (5)</option>
          </select>
        </div>
        <div>
          <label>소비 성향</label>
          <select id="userType">
            <option value="planned_spending" selected>계획 지출 선호</option>
            <option value="food_lover">음식 선호</option>
            <option value="electronics_lover">전자제품 선호</option>
            <option value="hobby_spender">취미 소비 선호</option>
            <option value="budget_shopper">저가 상품 선호</option>
            <option value="gift_giver">기념일/선물 선호</option>
          </select>
        </div>
      </div>

      <div class="grid-3 row">
        <div>
          <label>항목</label>
          <select id="category">
            <option>전자제품</option>
            <option>가전</option>
            <option>식료품</option>
            <option>의류</option>
            <option>취미</option>
            <option>외식</option>
            <option>교통</option>
            <option>생활용품</option>
            <option>전자기기</option>
          </select>
        </div>
        <div>
          <label>구매 이유</label>
          <select id="reason">
            <option>필요</option>
            <option>계획된 지출</option>
            <option>기념일 선물로</option>
            <option>스트레스 해소용</option>
            <option>친구 추천으로</option>
            <option>가격이 좋아서</option>
            <option>평소에 사고 싶었음</option>
            <option>새로운 취미 시작해서</option>
            <option>즉흥 구매</option>
            <option>온라인 광고 보고</option>
          </select>
        </div>
        <div>
          <label>요일 / 월</label>
          <div class="grid-2" style="gap:8px">
            <select id="day">
              <option>월요일</option><option>화요일</option><option>수요일</option>
              <option>목요일</option><option>금요일</option><option>토요일</option><option>일요일</option>
            </select>
            <input id="month" type="number" min="1" max="12" value="8"/>
          </div>
        </div>
      </div>

      <div class="actions">
        <button class="btn primary" onclick="predict()">예측하기</button>
        <button class="btn ghost" onclick="clearResult()">결과 지우기</button>
        <div class="muted" id="modeHint"></div>
      </div>
      <div class="result" id="resultPredict">결과 대기 중</div>

      <h2>데이터 저장 (POST /save_data)</h2>
      <div class="grid-2 row">
        <div>
          <label>금액(원)</label>
          <input id="s_price" type="number" min="0" placeholder="예: 28900"/>
        </div>
        <div>
          <label>제품명</label>
          <input id="s_product" type="text" placeholder='예: "푸라닭 치킨(순살)"'/>
        </div>
      </div>
      <div class="grid-3 row">
        <div>
          <label>당시 기분</label>
          <select id="s_mood">
            <option>매우나쁨</option><option>나쁨</option><option selected>보통</option>
            <option>좋음</option><option>행복</option>
          </select>
        </div>
        <div>
          <label>후회 여부</label>
          <select id="s_regret">
            <option>아니오</option><option>예</option>
          </select>
        </div>
        <div>
          <label>구매 이유</label>
          <select id="s_reason">
            <option>필요</option>
            <option>계획된 지출</option>
            <option>기념일 선물로</option>
            <option>스트레스 해소용</option>
            <option>친구 추천으로</option>
            <option>가격이 좋아서</option>
            <option>평소에 사고 싶었음</option>
            <option>새로운 취미 시작해서</option>
            <option>즉흥 구매</option>
            <option>온라인 광고 보고</option>
          </select>
        </div>
      </div>
      <div class="actions">
        <button class="btn primary" onclick="saveData()">저장하기</button>
        <div class="muted">CSV는 5컬럼 고정: 금액(원),제품명,당시 기분,후회 여부,구매 이유</div>
      </div>
      <div class="result" id="resultSave">-</div>
    </div>
  </div>

<script>
async function fetchJSON(path, opts) {
  const res = await fetch(path, Object.assign({ headers: { 'Content-Type': 'application/json' } }, opts || {}));
  const txt = await res.text();
  try { return { ok: res.ok, code: res.status, json: JSON.parse(txt) }; }
  catch { return { ok: res.ok, code: res.status, json: { raw: txt } }; }
}

async function predict() {
  const payload = {
    "금액(원)": Number(document.getElementById('price').value || 0),
    "당시 기분": Number(document.getElementById('mood').value || 3),
    "항목": document.getElementById('category').value,
    "구매 이유": document.getElementById('reason').value,
    "요일": document.getElementById('day').value,
    "월": Number(document.getElementById('month').value || 1),
    "user_type": document.getElementById('userType').value
  };
  const r = await fetchJSON('/predict', { method: 'POST', body: JSON.stringify(payload) });
  const el = document.getElementById('resultPredict');
  const hint = document.getElementById('modeHint');
  if (r.ok) {
    const p = r.json.regret_probability;
    el.textContent = `후회 확률: ${(p*100).toFixed(2)}%  (유형: ${r.json.user_type||payload.user_type})`;
    hint.textContent = r.json.note ? `mode: ${r.json.note}` : '';
  } else {
    el.textContent = `오류(${r.code}): ${JSON.stringify(r.json)}`;
  }
}

function clearResult() {
  document.getElementById('resultPredict').textContent = '결과 대기 중';
}

async function saveData() {
  const payload = {
    "금액(원)": Number(document.getElementById('s_price').value || 0),
    "제품명": document.getElementById('s_product').value || "",
    "당시 기분": document.getElementById('s_mood').value,
    "후회 여부": document.getElementById('s_regret').value,
    "구매 이유": document.getElementById('s_reason').value
  };
  const r = await fetchJSON('/save_data', { method: 'POST', body: JSON.stringify(payload) });
  const el = document.getElementById('resultSave');
  el.textContent = r.ok ? `저장 성공 ✅ (${r.json.path})` : `실패(${r.code}): ${JSON.stringify(r.json)}`;
}

// 초기 health 표시
(async () => {
  try {
    const r = await fetchJSON('/health');
    const hint = document.getElementById('modeHint');
    if (r.ok && r.json) {
      hint.textContent = `mode: ${r.json.mode} (model_loaded=${r.json.model_loaded})`;
    }
  } catch (_) {}
})();
</script>
</body></html>
    """.strip()
    return Response(html, mimetype="text/html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 61006))
    app.run(host="0.0.0.0", port=port)
