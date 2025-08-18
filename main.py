# main.py — SpendAI (Premium UI + API, single file)
# - / : 고급 다크 테마 UI (반응형, 게이지/미니차트/토스트/탭)
# - /health, /predict, /save_data, /get_preferences, /set_preferences
# - 모델 자동 로드 (pipeline -> model+preproc -> model_only -> heuristic)
# - 휴리스틱은 성향(user_type) vs 사유(reason) 정렬/충돌 가중치 적용
# - CSV 저장 5컬럼 고정: 금액(원),제품명,당시 기분,후회 여부,구매 이유
# - 데이터 경로: 로컬(Windows) 강제 또는 프로젝트상 상대경로

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import json, csv, os, math, traceback, random

# ML
try:
    import joblib
    import numpy as np
    import pandas as pd
except Exception:
    joblib = None
    np = None
    pd = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------- Paths --------
ROOT = Path(__file__).resolve().parent

# ---- 로컬(Windows)에서 VSCode CSV로 바로 저장하고 싶을 때 ----
# 1) 기본 윈도우 로컬 CSV 경로
LOCAL_CSV = Path(r"C:\Users\USER\Desktop\spendAI\data\purchase_data.csv")
# 2) 환경변수로 덮어쓰고 싶으면 FORCE_LOCAL_CSV=1, LOCAL_CSV_PATH=... 사용
USE_LOCAL_FORCE = os.environ.get("FORCE_LOCAL_CSV", "1") == "1"  # 기본 ON
LOCAL_CSV_ENV = os.environ.get("LOCAL_CSV_PATH", "")

# 최종 데이터 경로 결정
if os.name == "nt" and USE_LOCAL_FORCE:
    # 윈도우 + 로컬 강제 사용
    if LOCAL_CSV_ENV:
        DATA_FILE = Path(LOCAL_CSV_ENV)
    else:
        DATA_FILE = LOCAL_CSV
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREF_FILE = DATA_FILE.parent / "user_preferences.json"
else:
    # (클라우드/기타 OS) 프로젝트 상대 경로
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
def _try_load_models():
    global pipeline, model, preproc, model_loaded, model_mode
    try:
        if joblib is None:
            _log("joblib not available, skip model load (heuristic).")
            return
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

_try_load_models()

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

CATS_UP   = {"전자제품", "전자기기", "의류", "가전"}  # 가전 추가
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

    # 성향-카테고리 시너지/페널티 (미세)
    if ut == "electronics_lover":
        if c in {"전자제품", "전자기기"}: delta -= 0.08
        if r in REASONS_IMPULSIVE:        delta += 0.05
    elif ut == "hobby_spender":
        if c in {"취미"}:                 delta -= 0.06
    elif ut == "food_lover":
        if c in {"식료품", "외식"}:       delta -= 0.06
    return delta

def _heuristic_regret(p: dict) -> float:
    price     = float(p.get("금액(원)", 0) or 0)
    mood      = float(p.get("당시 기분", 3) or 3)  # 1~5
    reason    = str(p.get("구매 이유", "") or "").strip()
    category  = str(p.get("항목", "") or "").strip()
    day       = str(p.get("요일", "") or "").strip()
    month     = int(p.get("월", 1) or 1)
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

def _as_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        try:
            # numpy scalar 등
            return float(getattr(v, "item", lambda: default)())
        except Exception:
            return float(default)

def _extract_proba_one(proba_row) -> float:
    """
    predict_proba 반환값 안전 처리:
    - (2,) like [p0, p1] -> p1
    - (1,) like [p1]     -> 그 값
    - 스칼라              -> 그 값
    - dict-like           -> 클래스 1 또는 최대값
    """
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None

    # numpy array / list / tuple
    try:
        # numpy array -> flatten
        if _np is not None and isinstance(proba_row, _np.ndarray):
            arr = proba_row.flatten()
            if arr.size >= 2:
                return _as_float(arr[-1], 0.0)  # 보통 양성 클래스 확률
            elif arr.size == 1:
                return _as_float(arr[0], 0.0)
            return 0.0
    except Exception:
        pass

    # list/tuple
    if isinstance(proba_row, (list, tuple)):
        if len(proba_row) >= 2:
            return _as_float(proba_row[-1], 0.0)
        elif len(proba_row) == 1:
            return _as_float(proba_row[0], 0.0)
        return 0.0

    # dict-like
    if isinstance(proba_row, dict):
        # 클래스 1 우선, 없으면 최대값
        if 1 in proba_row:
            return _as_float(proba_row[1], 0.0)
        if "1" in proba_row:
            return _as_float(proba_row["1"], 0.0)
        try:
            return max((_as_float(v, 0.0) for v in proba_row.values()), default=0.0)
        except Exception:
            return 0.0

    # 스칼라
    return _as_float(proba_row, 0.0)

def _predict_with_model(payload: dict) -> float:
    try:
        if model_loaded and pd is not None:
            if model_mode == "pipeline" and pipeline is not None:
                df = pd.DataFrame([{k: payload.get(k) for k in _FEATURES}])
                y = pipeline.predict(df)
                p = _as_float(y[0], 0.0)
                return max(0.0, min(1.0, p))

            elif model_mode == "model+preproc" and model is not None and preproc is not None:
                df = pd.DataFrame([{k: payload.get(k) for k in _FEATURES}])
                X = preproc.transform(df)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    # (n_samples, n_classes) -> row 0
                    proba_row = proba[0] if getattr(proba, "ndim", 2) >= 1 else proba
                    p = _extract_proba_one(proba_row)
                else:
                    y = model.predict(X)
                    p = _as_float(y[0], 0.0)
                return max(0.0, min(1.0, p))

            elif model_mode == "model_only" and model is not None and np is not None:
                # 최소 수치 벡터 (권장 X)
                cats = {"전자제품": 1, "전자기기": 1, "의류": 1, "가전": 1, "식료품": -1, "생활용품": -1}
                reasons = {
                    "즉흥 구매": 1, "스트레스 해소용": 1, "온라인 광고 보고": 1,
                    "필요": -1, "계획된 지출": -1, "기념일 선물로": -1
                }
                days = {"금요일": 1, "토요일": 1}
                price = _as_float(payload.get("금액(원)", 0), 0)
                mood  = _as_float(payload.get("당시 기분", 3), 3)
                month = int(payload.get("월", 1) or 1)
                c = cats.get(str(payload.get("항목","") or ""), 0)
                r = reasons.get(str(payload.get("구매 이유","") or ""), 0)
                d = days.get(str(payload.get("요일","") or ""), 0)
                u = 1 if str(payload.get("user_type","") or "") in {"hobby_spender","electronics_lover"} else (-1 if str(payload.get("user_type","") or "")=="planned_spending" else 0)
                Xa = np.array([[price, mood, month, c, r, d, u]], dtype=float)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(Xa)
                    proba_row = proba[0] if getattr(proba, "ndim", 2) >= 1 else proba
                    p = _extract_proba_one(proba_row)
                else:
                    y = model.predict(Xa)
                    p = _as_float(y[0], 0.0)
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
        "regret_probability": float(max(0.0, min(1.0, p))),
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

    # 대상 경로 보장
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

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
    PREF_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREF_FILE.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"status": "success"})

# -------- Premium UI --------
@app.route("/", methods=["GET"])
def index_page():
    html = """
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SpendAI – Predict & Log</title>
<style>
  :root{
    --bg:#0b0c10; --panel:#13151b; --card:#181b22; --ink:#e7e8ea; --mut:#9aa0a6;
    --line:#2a2e36; --acc:#66b3ff; --acc-2:#7ee787; --warn:#ffb86b; --danger:#ff6b6b;
  }
  *{box-sizing:border-box}
  body{margin:0;background:linear-gradient(120deg,#0b0c10 0%,#0e1117 100%);color:var(--ink);
       font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,"Apple SD Gothic Neo","Noto Sans KR","맑은 고딕",sans-serif;}
  .nav{position:sticky;top:0;z-index:100;backdrop-filter:blur(10px);background:rgba(11,12,16,.6);
       border-bottom:1px solid var(--line)}
  .navin{display:flex;gap:16px;align-items:center;max-width:1120px;margin:0 auto;padding:12px 16px}
  .brand{font-weight:800;letter-spacing:.4px}
  .chip{padding:4px 10px;border-radius:999px;border:1px solid var(--line);font-size:12px;color:var(--mut)}
  .wrap{max-width:1120px;margin:24px auto;padding:0 16px}
  .tabs{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
  .tab{cursor:pointer;padding:10px 14px;border-radius:10px;border:1px solid var(--line);background:var(--panel);color:var(--ink);font-weight:700}
  .tab.active{background:var(--acc);color:#0b0c10;border-color:transparent}
  .grid-3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
  .grid-2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
  @media(max-width:860px){.grid-3{grid-template-columns:1fr} .grid-2{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}
  label{display:block;font-size:13px;color:var(--mut);margin:0 0 8px 2px}
  input,select{width:100%;height:44px;border-radius:12px;border:1px solid var(--line);background:#0f1116;color:var(--ink);padding:10px 12px;outline:none}
  input[type="range"]{height:auto}
  .btn{cursor:pointer;border:none;border-radius:12px;padding:12px 16px;font-weight:800}
  .btn.primary{background:var(--acc);color:#0b0c10}
  .btn.ghost{background:transparent;border:1px solid var(--line);color:var(--ink)}
  .section-title{font-size:14px;color:var(--mut);margin:0 0 12px 4px;letter-spacing:.4px}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  .result{margin-top:12px;padding:14px;border:1px dashed var(--line);border-radius:12px;background:#0f1116}
  .cards{display:grid;grid-template-columns:2.2fr 1fr;gap:16px}
  @media(max-width:1020px){.cards{grid-template-columns:1fr}}
  .metric{display:flex;gap:14px;align-items:center}
  .kpi{font-size:28px;font-weight:900}
  .desc{color:var(--mut);font-size:12px}
  .gauge{--p:0;position:relative;height:14px;border-radius:999px;background:#0f1116;border:1px solid var(--line);overflow:hidden}
  .gauge>i{position:absolute;inset:0;width:calc(var(--p)*1%);background:linear-gradient(90deg,var(--warn),var(--danger));transition:width .35s ease}
  .pill{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;padding:8px 12px;background:#0f1116}
  .tag{font-size:11px;color:var(--mut)}
  .toast{position:fixed;right:16px;bottom:16px;display:none;max-width:320px;background:#10131a;border:1px solid var(--line);border-radius:12px;padding:12px 14px;box-shadow:0 8px 24px rgba(0,0,0,.4)}
  .toast.show{display:block;animation:pop .2s ease}
  @keyframes pop{from{transform:translateY(6px);opacity:.6}to{transform:none;opacity:1}}
  .miniChart{height:64px;width:100%;border:1px solid var(--line);border-radius:10px;background:#0f1116;position:relative;overflow:hidden}
  .miniChart canvas{position:absolute;inset:0}
  .mut{color:var(--mut)}
</style>
</head>
<body>
  <div class="nav"><div class="navin">
    <div class="brand">SpendAI</div>
    <div class="chip" id="modeChip">mode: loading…</div>
    <div class="chip">data: __DATA_FILE_NAME__</div>
  </div></div>

  <div class="wrap">
    <div class="tabs">
      <button class="tab active" data-panel="predict">예측</button>
      <button class="tab" data-panel="save">데이터 저장</button>
      <button class="tab" data-panel="about">상태</button>
    </div>

    <div class="cards" id="panel-predict">
      <div class="card">
        <div class="section-title">예측 입력</div>
        <div class="grid-3">
          <div><label>금액(원)</label><input id="price" type="number" value="100000" min="0" step="1000"/></div>
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

        <div class="grid-3" style="margin-top:14px">
          <div>
            <label>항목</label>
            <select id="category">
              <option>전자제품</option><option>가전</option><option>식료품</option><option>의류</option>
              <option>취미</option><option>외식</option><option>교통</option><option>생활용품</option><option>전자기기</option>
            </select>
          </div>
          <div>
            <label>구매 이유</label>
            <select id="reason">
              <option>필요</option><option>계획된 지출</option><option>기념일 선물로</option><option>스트레스 해소용</option>
              <option>친구 추천으로</option><option>가격이 좋아서</option><option>평소에 사고 싶었음</option><option>새로운 취미 시작해서</option>
              <option>즉흥 구매</option><option>온라인 광고 보고</option>
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

        <div class="row" style="margin-top:16px">
          <button class="btn primary" onclick="predict()">예측하기</button>
          <button class="btn ghost" onclick="clearResult()">초기화</button>
          <span class="mut" id="modeHint"></span>
        </div>

        <div class="result" id="resultPredict">결과 대기 중</div>
      </div>

      <div class="card">
        <div class="section-title">결과</div>
        <div class="metric"><div class="kpi" id="kpiPercent">--%</div></div>
        <div class="gauge" style="margin-top:10px" id="gauge"><i></i></div>
        <div class="desc" style="margin-top:10px" id="descText">입력 후 예측을 실행하세요.</div>

        <div class="section-title" style="margin-top:20px">최근 추정 추이</div>
        <div class="miniChart"><canvas id="spark"></canvas></div>
      </div>
    </div>

    <div class="cards" id="panel-save" style="display:none">
      <div class="card">
        <div class="section-title">데이터 저장</div>
        <div class="grid-2">
          <div><label>금액(원)</label><input id="s_price" type="number" min="0" placeholder="예: 28900"/></div>
          <div><label>제품명</label><input id="s_product" type="text" placeholder='예: "푸라닭 치킨(순살)"'/></div>
        </div>
        <div class="grid-3" style="margin-top:14px">
          <div>
            <label>당시 기분</label>
            <select id="s_mood">
              <option>매우나쁨</option><option>나쁨</option><option selected>보통</option><option>좋음</option><option>행복</option>
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
              <option>필요</option><option>계획된 지출</option><option>기념일 선물로</option><option>스트레스 해소용</option>
              <option>친구 추천으로</option><option>가격이 좋아서</option><option>평소에 사고 싶었음</option><option>새로운 취미 시작해서</option>
              <option>즉흥 구매</option><option>온라인 광고 보고</option>
            </select>
          </div>
        </div>
        <div class="row" style="margin-top:16px">
          <button class="btn primary" onclick="saveData()">저장하기</button>
          <span class="mut">CSV 5컬럼 고정: 금액(원),제품명,당시 기분,후회 여부,구매 이유</span>
        </div>
        <div class="result" id="resultSave">-</div>
      </div>

      <div class="card">
        <div class="section-title">Tip</div>
        <div class="pill"><span class="tag">저장 경로</span><b style="font-size:12px">__DATA_FILE_PATH__</b></div>
      </div>
    </div>

    <div class="cards" id="panel-about" style="display:none">
      <div class="card">
        <div class="section-title">상태</div>
        <div class="pill"><span class="tag">모드</span><b id="stateMode">-</b></div>
        <div class="pill" style="margin-top:8px"><span class="tag">데이터 경로</span><b>__DATA_FILE_PATH__</b></div>
        <div class="result" id="stateRaw" style="margin-top:12px">-</div>
      </div>
      <div class="card">
        <div class="section-title">가이드</div>
        <div class="mut">/predict, /save_data, /get_preferences, /set_preferences 를 사용할 수 있어요.</div>
      </div>
    </div>

  </div>

  <div class="toast" id="toast"><b>완료</b><div id="toastText" class="mut"></div></div>

<script>
const $ = (q)=>document.querySelector(q);
const $$ = (q)=>document.querySelectorAll(q);
function toast(msg){ const t=$("#toast"); $("#toastText").textContent=msg; t.classList.add("show"); setTimeout(()=>t.classList.remove("show"), 2200); }

function switchTab(k){
  $$(".tab").forEach(b=>b.classList.toggle("active", b.dataset.panel===k));
  ["predict","save","about"].forEach(id=>{
    $("#panel-"+id).style.display = (id===k) ? "" : "none";
  });
}
$$(".tab").forEach(b=>b.addEventListener("click", ()=>switchTab(b.dataset.panel)));

async function fetchJSON(path, opts) {
  const res = await fetch(path, Object.assign({ headers: { 'Content-Type': 'application/json' } }, opts||{}));
  const txt = await res.text();
  try { return { ok: res.ok, code: res.status, json: JSON.parse(txt) }; }
  catch { return { ok: res.ok, code: res.status, json: { raw: txt } }; }
}

let hist = [];

function setGauge(p){
  const g=$("#gauge"); const i=$("#gauge i");
  const pct=Math.max(0,Math.min(100,Math.round(p*100)));
  g.style.setProperty("--p", pct);
  i.style.width = pct + "%";
  $("#kpiPercent").textContent = pct.toFixed(0)+"%";
  $("#descText").textContent = pct<35 ? "안정적인 지출로 보입니다." : (pct<70 ? "경계 구간입니다." : "후회 가능성이 높아요.");
}

function drawSpark(){
  const c = $("#spark");
  const dpr = window.devicePixelRatio||1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = w*dpr; c.height = h*dpr;
  const ctx = c.getContext("2d"); ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = "#66b3ff"; ctx.lineWidth = 2; ctx.lineJoin="round";
  if(hist.length===0) return;
  const max=1,min=0;
  ctx.beginPath();
  hist.forEach((v,idx)=>{
    const x = idx*(w/Math.max(1,hist.length-1));
    const y = h - (v-min)/(max-min)*h;
    if(idx===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
}

async function predict(){
  const payload = {
    "금액(원)": Number($("#price").value||0),
    "당시 기분": Number($("#mood").value||3),
    "항목": $("#category").value,
    "구매 이유": $("#reason").value,
    "요일": $("#day").value,
    "월": Number($("#month").value||1),
    "user_type": $("#userType").value
  };
  const r = await fetchJSON('/predict', { method:'POST', body: JSON.stringify(payload) });
  const el = $("#resultPredict");
  const hint = $("#modeHint");
  if(r.ok){ 
    const p = Number(r.json.regret_probability||0);
    const ut = r.json.user_type||payload.user_type;
    el.textContent = `후회 확률: ${(p*100).toFixed(2)}%  (유형: ${ut})`;
    hint.textContent = r.json.note ? `mode: ${r.json.note}` : '';
    hist.push(p); if(hist.length>24) hist = hist.slice(-24);
    setGauge(p); drawSpark(); toast("예측 완료");
  } else {
    el.textContent = `오류(${r.code}): ${JSON.stringify(r.json)}`;
  }
}

function clearResult(){
  $("#resultPredict").textContent = "결과 대기 중";
  setGauge(0); drawSpark();
}

async function saveData(){
  const payload = {
    "금액(원)": Number($("#s_price").value||0),
    "제품명": $("#s_product").value||"",
    "당시 기분": $("#s_mood").value,
    "후회 여부": $("#s_regret").value,
    "구매 이유": $("#s_reason").value
  };
  const r = await fetchJSON('/save_data', { method:'POST', body: JSON.stringify(payload) });
  const el = $("#resultSave");
  if(r.ok){ el.textContent = `저장 성공 ✅ (${r.json.path})`; toast("저장 완료"); }
  else { el.textContent = `실패(${r.code}): ${JSON.stringify(r.json)}`; }
}

// 초기 상태 로딩
(async ()=>{
  try{
    const r = await fetchJSON('/health');
    if(r.ok&&r.json){
      $("#modeChip").textContent = `mode: ${r.json.mode} (model_loaded=${r.json.model_loaded})`;
      $("#stateMode").textContent = `${r.json.mode}, loaded=${r.json.model_loaded}`;
      $("#stateRaw").textContent = JSON.stringify(r.json, null, 2);
      // 데모용 스파크 초기화
      hist = Array.from({length:12},()=>Math.max(0,Math.min(1, (Math.random()*0.5)+0.25 )));
      setGauge(hist[hist.length-1]||0);
      drawSpark();
    }
  }catch(e){}
})();
</script>
</body>
</html>
    """.strip()

    # 안전한 플레이스홀더 치환 (중괄호 충돌 없음)
    html = html.replace("__DATA_FILE_NAME__", DATA_FILE.name)
    html = html.replace("__DATA_FILE_PATH__", DATA_FILE.as_posix())

    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 61006))
    app.run(host="0.0.0.0", port=port)
