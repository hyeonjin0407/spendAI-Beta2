# main.py — SpendAI (Premium UI + API, single file)
# - / : 고급 다크 테마 UI (반응형, 미니차트/토스트/탭/가이드)
# - /health, /predict, /save_data, /get_preferences, /set_preferences, /debug_fs
# - 모델 자동 로드 (xgb_ensemble -> pipeline -> dict{'pipeline'|'model'+'preproc'} -> model_only -> heuristic)
# - 예측 안정화: NaN/범위 방어, 제품명 공백 기본 토큰
# - CSV 저장 5컬럼 고정: 금액(원),제품명,당시 기분,후회 여부,구매 이유
# - 데이터 경로: Windows 로컬 강제(기본) 또는 프로젝트 상대(spendAI/xpend/data)
# - 런타임 백업 다운로드(gdown): MODEL_GDRIVE_URL 환경변수로 구글드라이브에서 모델 내려받기

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import json, csv, os, math, traceback, sys, subprocess

# ML
try:
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
except Exception:
    joblib = None
    np = None
    pd = None
    sklearn = None

# XGBoost & SciPy (앙상블 번들용)
try:
    import xgboost as xgb
    import scipy.sparse as sp
except Exception:
    xgb = None
    sp = None

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------- Paths --------
ROOT = Path(__file__).resolve().parent

# ---- 로컬(Windows) CSV 강제 저장 옵션 ----
LOCAL_CSV = Path(r"C:\Users\USER\Desktop\spendAI\data\purchase_data.csv")
USE_LOCAL_FORCE = os.environ.get("FORCE_LOCAL_CSV", "1") == "1"  # 기본 ON
LOCAL_CSV_ENV = os.environ.get("LOCAL_CSV_PATH", "")

if os.name == "nt" and USE_LOCAL_FORCE:
    DATA_FILE = Path(LOCAL_CSV_ENV) if LOCAL_CSV_ENV else LOCAL_CSV
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREF_FILE = DATA_FILE.parent / "user_preferences.json"
else:
    DATA_DIR = ROOT / "spendAI" / "xpend" / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_FILE = DATA_DIR / "purchase_data.csv"
    PREF_FILE = DATA_DIR / "user_preferences.json"

MODEL_DIR = ROOT / "model"
PIPELINE_PATH = MODEL_DIR / "spendai_pipeline.pkl"
ALT_PIPELINE_PATH = ROOT / "spendai_pipeline.pkl"  # 루트에 저장된 경우도 허용
MODEL_PATH    = MODEL_DIR / "spendai_model.pkl"
PREPROC_PATH  = MODEL_DIR / "spendai_preprocessor.pkl"

# -------- Model holders --------
pipeline = None          # sklearn Pipeline or estimator
model = None             # estimator only
preproc = None           # preprocessor only
ensemble_bundle = None   # xgb 앙상블 번들(dict)
model_loaded = False
model_mode = "heuristic"  # "xgb_ensemble" | "pipeline" | "model+preproc" | "model_only" | "heuristic"

def _log(msg: str):
    print("[SpendAI] " + str(msg), flush=True)

# -------- Utils --------
def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def _as_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        try:
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

    try:
        if _np is not None and isinstance(proba_row, _np.ndarray):
            arr = proba_row.flatten()
            if arr.size >= 2:
                return _as_float(arr[-1], 0.0)
            elif arr.size == 1:
                return _as_float(arr[0], 0.0)
            return 0.0
    except Exception:
        pass

    if isinstance(proba_row, (list, tuple)):
        if len(proba_row) >= 2:
            return _as_float(proba_row[-1], 0.0)
        elif len(proba_row) == 1:
            return _as_float(proba_row[0], 0.0)
        return 0.0

    if isinstance(proba_row, dict):
        if 1 in proba_row:
            return _as_float(proba_row[1], 0.0)
        if "1" in proba_row:
            return _as_float(proba_row["1"], 0.0)
        try:
            return max((_as_float(v, 0.0) for v in proba_row.values()), default=0.0)
        except Exception:
            return 0.0

    return _as_float(proba_row, 0.0)

# -------- Heuristic Model --------
REASONS_IMPULSIVE = {"즉흥 구매", "스트레스 해소용", "온라인 광고 보고"}
REASONS_PLANNED   = {"필요", "계획된 지출", "기념일 선물로"}
CATS_UP   = {"전자제품", "전자기기", "의류", "가전"}
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
    price     = _as_float(p.get("금액(원)", 0), 0)
    mood      = _as_float(p.get("당시 기분", 3), 3)  # 1~5
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

# -------- Payload -> training feature row (for ensemble) --------
def _row_from_payload(payload: dict):
    # 요일 → 0(월)~6(일)
    dow_map = {"월요일":0,"화요일":1,"수요일":2,"목요일":3,"금요일":4,"토요일":5,"일요일":6}
    # 감정 정수(1~5) → 라벨
    feeling_map = {1:"매우나쁨", 2:"나쁨", 3:"보통", 4:"좋음", 5:"행복"}

    amount  = _as_float(payload.get("금액(원)", 0), 0)
    month   = int(payload.get("월", 0) or 0)
    dow_ko  = str(payload.get("요일","") or "")
    feeling_num = int(payload.get("당시 기분", 3) or 3)
    product = str(payload.get("제품명", "") or "")
    if not product.strip():
        product = "상품"  # 텍스트 피처 기본 토큰

    row = {
        "amount": amount,
        "category": str(payload.get("항목","") or "기타"),
        "reason":   str(payload.get("구매 이유","") or "필요"),
        "feeling":  feeling_map.get(feeling_num, "보통"),
        "month":    month,
        "dow":      dow_map.get(dow_ko, 0),
        "product":  product,
    }
    if np is not None:
        row["amount_log"] = float(np.log1p(max(0.0, row["amount"])))
    else:
        row["amount_log"] = 0.0
    if pd is not None:
        return pd.DataFrame([row])
    raise RuntimeError("pandas not available")

# -------- Model loading --------
def _try_load_models():
    global pipeline, model, preproc, ensemble_bundle, model_loaded, model_mode
    try:
        if joblib is None:
            _log("joblib not available, skip model load (heuristic).")
            return

        # 최우선: 파일이 없으면 런타임 백업 다운로드(gdown) 시도 (환경변수 필요)
        if (not PIPELINE_PATH.exists()) and os.environ.get("MODEL_GDRIVE_URL"):
            try:
                _log("Pipeline not found; trying runtime gdown via MODEL_GDRIVE_URL ...")
                subprocess.check_call(["gdown", "--fuzzy", os.environ["MODEL_GDRIVE_URL"], "-O", str(PIPELINE_PATH)])
                _log("Runtime gdown success.")
            except Exception as e:
                _log("Runtime gdown failed: " + str(e))

        # 우선순위: model/spendai_pipeline.pkl -> ./spendai_pipeline.pkl
        path = PIPELINE_PATH if PIPELINE_PATH.exists() else (ALT_PIPELINE_PATH if ALT_PIPELINE_PATH.exists() else None)
        if path is not None:
            obj = joblib.load(path)

            # 1) XGB 앙상블 번들(dict) 지원
            if isinstance(obj, dict) and "folds" in obj:
                if xgb is None or sp is None:
                    _log("xgboost/scipy not installed; cannot use ensemble bundle. Fallback to others.")
                else:
                    fs = obj.get("folds") or []
                    if fs and all(k in fs[0] for k in ["pre_catnum","tfidf_word","tfidf_char","booster"]):
                        ensemble_bundle = obj
                        model_loaded = True
                        model_mode = "xgb_ensemble"
                        _log("Loaded XGB ensemble bundle: " + str(path) + " (folds=" + str(len(fs)) + ")")
                        return  # 가장 우선

            # 2) dict로 저장된 (pipeline) 또는 (model+preproc)
            if isinstance(obj, dict):
                cand = obj.get("pipeline") or obj.get("estimator") or obj.get("model")
                if cand is not None and hasattr(cand, "predict"):
                    pipeline = cand
                    model_loaded = True
                    model_mode = "pipeline"
                    _log("Loaded pipeline from dict key: " + str(path))
                    return
                elif "preproc" in obj and "model" in obj and hasattr(obj["model"], "predict"):
                    preproc = obj["preproc"]; model = obj["model"]
                    model_loaded = True
                    model_mode = "model+preproc"
                    _log("Loaded model+preproc from dict: " + str(path))
                    return
                else:
                    _log("Dict loaded but no usable estimator found; continue to explicit files...")
            else:
                # 3) 객체가 estimator면 pipeline 모드
                if hasattr(obj, "predict"):
                    pipeline = obj
                    model_loaded = True
                    model_mode = "pipeline"
                    _log("Loaded pipeline object: " + str(path))
                    return
                else:
                    _log("Object at PIPELINE_PATH has no predict(); continue...")

        # 4) 명시적 모델/전처리자 경로
        if MODEL_PATH.exists():
            try:
                model = joblib.load(MODEL_PATH)
                if PREPROC_PATH.exists():
                    preproc = joblib.load(PREPROC_PATH)
                    model_mode = "model+preproc"
                    _log("Loaded model+preproc: " + str(MODEL_PATH) + ", " + str(PREPROC_PATH))
                else:
                    model_mode = "model_only"
                    _log("Loaded model only: " + str(MODEL_PATH))
                model_loaded = True
                return
            except Exception as e:
                _log("Explicit model load failed: " + str(e) + "\n" + traceback.format_exc())

        # 5) 모두 실패 → heuristic
        _log("No usable model found. Using heuristic fallback.")
    except Exception as e:
        _log("Model load failed: " + str(e) + "\n" + traceback.format_exc())
        pipeline = None; model = None; preproc = None; ensemble_bundle = None
        model_loaded = False; model_mode = "heuristic"

# -------- Ensemble predict --------
def _predict_with_ensemble(payload: dict) -> float:
    if ensemble_bundle is None or xgb is None or sp is None:
        return _heuristic_regret(payload)
    try:
        df = _row_from_payload(payload)  # pandas DF 1행
        preds = []
        for f in ensemble_bundle.get("folds", []):
            pre_catnum = f["pre_catnum"]
            tfidf_word = f["tfidf_word"]
            tfidf_char = f["tfidf_char"]
            booster    = f["booster"]
            best_iter  = int(f.get("best_iter", 0))

            M_cn = pre_catnum.transform(df)
            M_w  = tfidf_word.transform(df["product"])
            M_c  = tfidf_char.transform(df["product"])
            Xmat = sp.hstack([M_cn, M_w, M_c], format="csr")

            dmat = xgb.DMatrix(Xmat)
            proba = booster.predict(dmat, iteration_range=(0, best_iter+1))
            val = float(proba[0])
            # 안정화: NaN/범위 방어
            if not (0.0 <= val <= 1.0) or (val != val):
                val = 0.5
            preds.append(val)

        if not preds:
            return _heuristic_regret(payload)
        p = float(sum(preds) / len(preds))
        if not (0.0 <= p <= 1.0) or (p != p):
            p = 0.5
        return max(0.0, min(1.0, p))
    except Exception as e:
        _log("Ensemble predict failed: " + str(e) + "\n" + traceback.format_exc() + " — fallback heuristic.")
        return _heuristic_regret(payload)

# -------- Prediction --------
def _predict_with_model(payload: dict) -> float:
    # 0) ensemble 우선
    if model_mode == "xgb_ensemble":
        return _predict_with_ensemble(payload)

    try:
        if model_loaded and pd is not None:
            # 1) pipeline 모드
            if model_mode == "pipeline" and pipeline is not None:
                df = pd.DataFrame([{k: payload.get(k) for k in _FEATURES}])
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(df)
                    proba_row = proba[0] if getattr(proba, "ndim", 2) >= 1 else proba
                    p = _extract_proba_one(proba_row)
                else:
                    y = pipeline.predict(df)
                    p = _as_float(y[0], 0.0)
                return max(0.0, min(1.0, p))

            # 2) model+preproc 모드
            if model_mode == "model+preproc" and model is not None and preproc is not None:
                df = pd.DataFrame([{k: payload.get(k) for k in _FEATURES}])
                X = preproc.transform(df)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    proba_row = proba[0] if getattr(proba, "ndim", 2) >= 1 else proba
                    p = _extract_proba_one(proba_row)
                else:
                    y = model.predict(X)
                    p = _as_float(y[0], 0.0)
                return max(0.0, min(1.0, p))

            # 3) model_only (간소화 임베딩)
            if model_mode == "model_only" and model is not None and np is not None:
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
        _log("Predict failed with model: " + str(e) + "\n" + traceback.format_exc() + " — fallback to heuristic.")

    return _heuristic_regret(payload)

# -------- APIs --------
@app.after_request
def add_no_cache_headers(resp):
    # 캐시로 UI가 안바뀌는 현상 방지
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(model_loaded),
        "mode": model_mode,
        "data_path": str(DATA_FILE),
        "pipeline_path": str(PIPELINE_PATH),
        "alt_pipeline_path": str(ALT_PIPELINE_PATH),
        "pipeline_path_exists": PIPELINE_PATH.exists(),
        "alt_pipeline_path_exists": ALT_PIPELINE_PATH.exists(),
        "model_path_exists": MODEL_PATH.exists(),
        "preproc_path_exists": PREPROC_PATH.exists(),
        "py": sys.version.split(" ")[0],
        "numpy": getattr(np, "__version__", None),
        "pandas": getattr(pd, "__version__", None),
        "sklearn": getattr(sklearn, "__version__", None),
        "xgboost": getattr(xgb, "__version__", None) if xgb else None,
        "now": datetime.now().isoformat(timespec="seconds"),
    })

@app.route("/debug_fs", methods=["GET"])
def debug_fs():
    paths = [
        str(ROOT),
        str(MODEL_DIR),
        str(PIPELINE_PATH),
        str(ALT_PIPELINE_PATH),
    ]
    listing = {}
    for p in paths:
        try:
            if os.path.isdir(p):
                listing[p] = sorted([name + "  (" + str(os.path.getsize(os.path.join(p, name))) + "B)"
                                     for name in os.listdir(p)])
            elif os.path.isfile(p):
                listing[p] = ["<FILE> size=" + str(os.path.getsize(p)) + "B"]
            else:
                listing[p] = ["<MISSING>"]
        except Exception as e:
            listing[p] = ["<ERROR> " + str(e)]
    return jsonify({"cwd": os.getcwd(), "tree": listing})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    required = ['금액(원)', '당시 기분', '항목', '구매 이유', '요일', '월', 'user_type']
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': "Missing field(s): " + ", ".join(missing)}), 400
    p = _predict_with_model(data)
    return jsonify({
        "regret_probability": float(max(0.0, min(1.0, p))),
        "user_type": data.get("user_type"),
        "note": model_mode if model_loaded else "heuristic"
    })

@app.route("/save_data", methods=["POST"])
def save_data():
    """
    CSV는 오직 5컬럼만: 금액(원),제품명,당시 기분,후회 여부,구매 이유
    """
    data = request.get_json(force=True, silent=True) or {}
    fieldnames = ['금액(원)', '제품명', '당시 기분', '후회 여부', '구매 이유']
    missing = [k for k in fieldnames if k not in data]
    if missing:
        return jsonify({'error': "Missing field(s): " + ", ".join(missing)}), 400

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

# -------- Premium UI (no f-string — placeholders replaced) --------
@app.route("/", methods=["GET"])
def index_page():
    html = """
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SpendAI – Smart Spending</title>
<style>
  :root{
    --bg:#0b0c10; --grad1:#0b0c10; --grad2:#11131a; --panel:#0f1218; --card:#121623; --glass:rgba(255,255,255,.04);
    --ink:#eef0f3; --mut:#9aa0a6; --line:#2a2e36; --acc:#7aa7ff; --acc-2:#7ee787; --warn:#ffb86b; --danger:#ff6b6b;
    --shadow:0 10px 30px rgba(0,0,0,.45);
  }
  *{box-sizing:border-box} html,body{height:100%}
  body{margin:0;background:radial-gradient(1200px 600px at 20% -10%, #172033 0%, transparent 60%),
               radial-gradient(1400px 800px at 120% 20%, #1b1d28 0%, transparent 60%),
               linear-gradient(120deg,var(--grad1) 0%,var(--grad2) 100%);
       color:var(--ink);
       font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,"Apple SD Gothic Neo","Noto Sans KR","맑은 고딕",sans-serif;}
  .nav{position:sticky;top:0;z-index:100;backdrop-filter:blur(10px);background:rgba(12,14,18,.55);border-bottom:1px solid var(--line)}
  .navin{display:flex;gap:16px;align-items:center;max-width:1160px;margin:0 auto;padding:12px 16px}
  .brand{font-weight:900;letter-spacing:.2px;font-size:18px;display:flex;align-items:center;gap:10px}
  .logo{width:22px;height:22px;border-radius:6px;background:linear-gradient(135deg,var(--acc),#a2b6ff);box-shadow:0 2px 12px rgba(122,167,255,.6)}
  .chip{padding:6px 10px;border-radius:999px;border:1px solid var(--line);font-size:12px;color:var(--mut);background:var(--glass)}
  .wrap{max-width:1160px;margin:26px auto;padding:0 16px}

  .tabs{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
  .tab{cursor:pointer;padding:10px 14px;border-radius:12px;border:1px solid var(--line);background:var(--glass);color:var(--ink);font-weight:700;transition:all .2s ease}
  .tab:hover{transform:translateY(-1px)}
  .tab.active{background:linear-gradient(90deg,var(--acc),#99c1ff);color:#0b0c10;border-color:transparent}

  .cards{display:grid;grid-template-columns:1.6fr 1fr;gap:16px}
  @media(max-width:1020px){.cards{grid-template-columns:1fr}}

  .card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:var(--shadow)}
  .section-title{font-size:13px;color:var(--mut);margin:0 0 12px 4px;letter-spacing:.4px;text-transform:uppercase}
  label{display:block;font-size:12px;color:var(--mut);margin:0 0 8px 2px}
  input,select{width:100%;height:44px;border-radius:12px;border:1px solid var(--line);background:#0f1116;color:var(--ink);padding:10px 12px;outline:none}
  input[type="number"]::-webkit-outer-spin-button, input[type="number"]::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
  .grid-3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
  .grid-2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
  @media(max-width:860px){.grid-3,.grid-2{grid-template-columns:1fr}}

  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  .btn{cursor:pointer;border:none;border-radius:12px;padding:12px 16px;font-weight:800;transition:transform .06s ease, filter .1s ease}
  .btn:active{transform:translateY(1px)}
  .btn.primary{background:linear-gradient(90deg,var(--acc),#99c1ff);color:#0b0c10}
  .btn.ghost{background:transparent;border:1px solid var(--line);color:var(--ink)}

  .metric{display:flex;gap:14px;align-items:center}
  .kpi{font-size:34px;font-weight:900}
  .desc{color:var(--mut);font-size:12px}
  .gauge{--p:0;position:relative;height:12px;border-radius:999px;background:#0e1115;border:1px solid var(--line);overflow:hidden}
  .gauge>i{position:absolute;inset:0;width:calc(var(--p)*1%);background:linear-gradient(90deg,var(--warn),var(--danger));transition:width .35s ease}

  .result{margin-top:12px;padding:14px;border:1px dashed var(--line);border-radius:12px;background:#0f1116;white-space:pre-wrap}
  .mut{color:var(--mut)}

  .pill{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;padding:8px 12px;background:#0f1116;margin:4px 4px 0 0}
  .pill b{font-size:12px}

  .miniChart{height:64px;width:100%;border:1px solid var(--line);border-radius:12px;background:#0f1116;position:relative;overflow:hidden}
  .miniChart canvas{position:absolute;inset:0}

  .toast{position:fixed;right:16px;bottom:16px;display:none;max-width:320px;background:#10131a;border:1px solid var(--line);border-radius:12px;padding:12px 14px;box-shadow:0 8px 24px rgba(0,0,0,.4)}
  .toast.show{display:block;animation:pop .18s ease}
  @keyframes pop{from{transform:translateY(6px);opacity:.6}to{transform:none;opacity:1}}

  .tip{font-size:12px;line-height:1.6;color:var(--mut)}
</style>
</head>
<body>
  <div class="nav"><div class="navin">
    <div class="brand"><div class="logo"></div>SpendAI</div>
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
        <div class="pill"><span class="mut">저장 경로</span><b style="font-size:12px">__DATA_FILE_PATH__</b></div>
        <div class="tip" style="margin-top:10px">
          Windows에서 로컬 강제 저장을 끄려면 환경변수 <code>FORCE_LOCAL_CSV=0</code> 로 실행하세요.
        </div>
      </div>
    </div>

    <div class="cards" id="panel-about" style="display:none">
      <div class="card">
        <div class="section-title">상태</div>
        <div class="pill"><span class="mut">모드</span><b id="stateMode">-</b></div>
        <div class="pill" style="margin-top:8px"><span class="mut">데이터 경로</span><b>__DATA_FILE_PATH__</b></div>
        <div class="result" id="stateRaw" style="margin-top:12px">-</div>
      </div>
      <div class="card">
        <div class="section-title">가이드</div>
        <div class="tip">
          <b>엔드포인트</b>: /predict, /save_data, /get_preferences, /set_preferences<br/>
          <b>모델 로드 순서</b>: xgb_ensemble → pipeline.pkl → dict{pipeline/model+preproc} → model_only → heuristic
        </div>
      </div>
    </div>

  </div>

  <div class="toast" id="toast"><b>완료</b><div id="toastText" class="mut"></div></div>

<script>
const $ = (q)=>document.querySelector(q);
const $$ = (q)=>document.querySelectorAll(q);
function toast(msg){ const t=document.querySelector("#toast"); document.querySelector("#toastText").textContent=msg; t.classList.add("show"); setTimeout(()=>t.classList.remove("show"), 1800); }

function switchTab(k){
  $$(".tab").forEach(b=>b.classList.toggle("active", b.dataset.panel===k));
  ["predict","save","about"].forEach(id=>{ document.querySelector("#panel-"+id).style.display = (id===k) ? "" : "none"; });
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
  const g=document.querySelector("#gauge"); const i=document.querySelector("#gauge i");
  const pct=Math.max(0,Math.min(100,Math.round(p*100)));
  g.style.setProperty("--p", pct);
  i.style.width = pct + "%";
  document.querySelector("#kpiPercent").textContent = pct.toFixed(0)+"%";
  document.querySelector("#descText").textContent = pct<35 ? "안정적인 지출로 보입니다." : (pct<70 ? "경계 구간입니다." : "후회 가능성이 높아요.");
}

function drawSpark(){
  const c = document.querySelector("#spark");
  const dpr = window.devicePixelRatio||1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = w*dpr; c.height = h*dpr;
  const ctx = c.getContext("2d"); ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = "#7aa7ff"; ctx.lineWidth = 2; ctx.lineJoin="round";
  if(hist.length===0) return;
  const max=1,min=0; ctx.beginPath();
  hist.forEach((v,idx)=>{ const x = idx*(w/Math.max(1,hist.length-1)); const y = h - (v-min)/(max-min)*h; if(idx===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
  ctx.stroke();
}

async function predict(){
  const payload = {
    "금액(원)": Number(document.querySelector("#price").value||0),
    "당시 기분": Number(document.querySelector("#mood").value||3),
    "항목": document.querySelector("#category").value,
    "구매 이유": document.querySelector("#reason").value,
    "요일": document.querySelector("#day").value,
    "월": Number(document.querySelector("#month").value||1),
    "user_type": document.querySelector("#userType").value,
    "제품명": document.querySelector("#s_product") ? (document.querySelector("#s_product").value||"") : ""
  };
  const r = await fetchJSON('/predict', { method:'POST', body: JSON.stringify(payload) });
  const el = document.querySelector("#resultPredict");
  const hint = document.querySelector("#modeHint");
  if(r.ok){
    const p = Number(r.json.regret_probability||0);
    const ut = r.json.user_type||payload.user_type;
    el.textContent = "후회 확률: " + (p*100).toFixed(2) + "%  (유형: " + ut + ")";
    hint.textContent = r.json.note ? ("mode: " + r.json.note) : '';
    hist.push(p); if(hist.length>24) hist = hist.slice(-24);
    setGauge(p); drawSpark(); toast("예측 완료");
  } else {
    el.textContent = "오류(" + r.code + "): " + JSON.stringify(r.json);
  }
}

function clearResult(){ document.querySelector("#resultPredict").textContent = "결과 대기 중"; setGauge(0); drawSpark(); }

async function saveData(){
  const payload = {
    "금액(원)": Number(document.querySelector("#s_price").value||0),
    "제품명": document.querySelector("#s_product").value||"",
    "당시 기분": document.querySelector("#s_mood").value,
    "후회 여부": document.querySelector("#s_regret").value,
    "구매 이유": document.querySelector("#s_reason").value
  };
  const r = await fetchJSON('/save_data', { method:'POST', body: JSON.stringify(payload) });
  const el = document.querySelector("#resultSave");
  if(r.ok){ el.textContent = "저장 성공 ✅ (" + r.json.path + ")"; toast("저장 완료"); }
  else { el.textContent = "실패(" + r.code + "): " + JSON.stringify(r.json); }
}

// 초기 상태 로딩
(async ()=>{
  try{
    const r = await fetchJSON('/health');
    if(r.ok&&r.json){
      document.querySelector("#modeChip").textContent = "mode: " + r.json.mode + " (loaded=" + r.json.model_loaded + ")";
      document.querySelector("#stateMode").textContent = r.json.mode + ", loaded=" + r.json.model_loaded;
      document.querySelector("#stateRaw").textContent = JSON.stringify(r.json, null, 2);
      hist = Array.from({length:12},()=>Math.max(0,Math.min(1, (Math.random()*0.5)+0.25 )));
      setGauge(hist[hist.length-1]||0); drawSpark();
    }
  }catch(e){}
})();
</script>
</body>
</html>
    """

    # 안전한 플레이스홀더 치환
    html = html.replace("__DATA_FILE_NAME__", DATA_FILE.name)
    html = html.replace("__DATA_FILE_PATH__", DATA_FILE.as_posix())

    resp = make_response(html)
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    return resp

if __name__ == "__main__":
    # 서버 시작 전에 모델 로드 시도
    _try_load_models()
    port = int(os.environ.get("PORT", 61006))
    app.run(host="0.0.0.0", port=port)
