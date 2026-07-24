"""
NADiSSP Inference API (Flask)
==============================
Full Flask API replacing FastAPI for environments without FastAPI.
"""

from __future__ import annotations
import os, sys, json, time, subprocess, threading
from collections import deque

import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory, abort
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.nadissp import extract_features, CHANNELS, SEQ_LEN, ModelWrapper

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH     = os.path.join(BASE_DIR, "checkpoints", "nadissp_model.joblib")
METRICS_PATH  = os.path.join(BASE_DIR, "checkpoints", "metrics.json")
HISTORY_PATH  = os.path.join(BASE_DIR, "checkpoints", "train_history.json")
ABLATION_PATH = os.path.join(BASE_DIR, "checkpoints", "ablation_results.json")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
FIG_DIR       = os.path.join(RESULTS_DIR, "figures")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

app = Flask(__name__, static_folder=None)
app.config["JSON_SORT_KEYS"] = False

# ── State ─────────────────────────────────────────────────────────────────────
_model       = None
_latency_log = deque(maxlen=1000)
_run_status  = {}   # script_name → status dict
_lock        = threading.Lock()


def _load_model():
    """Load the trained model from disk."""
    global _model
    
    if not os.path.exists(CKPT_PATH):
        print("⚠ Model not found. Run train.py first.")
        return False
    
    try:
        # Load the model dictionary
        model_dict = joblib.load(CKPT_PATH)
        
        # Extract components
        clf = model_dict.get('clf')
        reg = model_dict.get('reg')
        dom_clf = model_dict.get('dom_clf')
        scaler = model_dict.get('scaler')
        imputer = model_dict.get('imputer')
        
        if None in (clf, reg, dom_clf, scaler, imputer):
            print("⚠ Model components missing. Check model file.")
            return False
        
        # Create wrapper
        _model = ModelWrapper(clf, reg, dom_clf, scaler, imputer)
        print(f"✓ Model loaded: features={scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else '?'}")
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


# Load at startup
_load_model()


# ── CORS helper ───────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path=""):
    return "", 204


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route("/")
def root():
    p = os.path.join(DASHBOARD_DIR, "index.html")
    return send_file(p) if os.path.exists(p) else jsonify({"message":"NADiSSP API","docs":"/docs"})


@app.route("/figures/<path:name>")
def serve_figure_static(name):
    return send_from_directory(FIG_DIR, name)


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if _model else "degraded",
        "model_loaded": _model is not None,
        "channels": CHANNELS,
        "seq_len": SEQ_LEN,
        "backend": "sklearn",
    })


# ── Inference helpers ─────────────────────────────────────────────────────────
def _risk(p: float) -> str:
    return "critical" if p >= 0.7 else "elevated" if p >= 0.4 else \
           "watch"    if p >= 0.15 else "normal"


def _prepare(sequence: list) -> np.ndarray:
    """Prepare sequence with all required channels."""
    # Create array with all channels
    arr = np.zeros((len(sequence), len(CHANNELS)), dtype=np.float32)
    for i, s in enumerate(sequence):
        for j, ch in enumerate(CHANNELS):
            arr[i, j] = s.get(ch, 0.0)
    
    # Pad or truncate to SEQ_LEN
    if len(arr) < SEQ_LEN:
        if len(arr) > 0:
            pad = np.repeat(arr[-1:], SEQ_LEN - len(arr), axis=0)
        else:
            pad = np.zeros((SEQ_LEN - len(arr), len(CHANNELS)))
        arr = np.vstack([arr, pad])
    elif len(arr) > SEQ_LEN:
        arr = arr[-SEQ_LEN:]
    
    return arr


def _infer_one(item: dict) -> dict:
    if _model is None:
        return {"error": "Model not loaded. Run scripts/train.py first."}
    
    t0 = time.perf_counter()
    x = _prepare(item.get("sequence", []))  # (SEQ_LEN, n_ch)
    out = _model.predict(x[np.newaxis])     # batch dimension

    lat = (time.perf_counter() - t0) * 1000.0
    _latency_log.append(lat)

    return {
        "asset_id":          item.get("asset_id", "unknown"),
        "rul_estimate":      round(float(out["rul_pred"][0]), 2),
        "rul_unit":          "cycles",
        "failure_probability": round(float(out["failure_prob"][0]), 4),
        "risk_level":        _risk(float(out["failure_prob"][0])),
        "weibull_scale":     1.0,  # Placeholder
        "weibull_shape":     1.0,  # Placeholder
        "latency_ms":        round(lat, 3),
        "model_version":     "1.0.0-sklearn",
    }


# ── Inference endpoints ───────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    return jsonify(_infer_one(data))


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)
    items = data.get("items", [])
    return jsonify([_infer_one(item) for item in items])


# ── Metrics ───────────────────────────────────────────────────────────────────
def _load_json(path, label):
    if not os.path.exists(path):
        abort(404, description=f"{label} not found")
    with open(path) as f:
        return json.load(f)


@app.route("/metrics")
def metrics():
    return jsonify(_load_json(METRICS_PATH, "Training metrics"))


@app.route("/metrics/latency")
def latency_metrics():
    if not _latency_log:
        return jsonify({"count": 0})
    arr = np.array(_latency_log)
    return jsonify({
        "count":             len(arr),
        "p50_ms":            round(float(np.percentile(arr,50)), 3),
        "p95_ms":            round(float(np.percentile(arr,95)), 3),
        "p99_ms":            round(float(np.percentile(arr,99)), 3),
        "max_ms":            round(float(arr.max()), 3),
        "within_120ms_pct":  round(float((arr<120).mean()*100), 2),
        "target_ms":         120.0,
    })


@app.route("/metrics/history")
def training_history():
    return jsonify(_load_json(HISTORY_PATH, "Training history"))


@app.route("/metrics/ablation")
def ablation_results():
    return jsonify(_load_json(ABLATION_PATH, "Ablation results"))


# ── Results ───────────────────────────────────────────────────────────────────
@app.route("/results")
def results_index():
    figs = sorted(os.listdir(FIG_DIR)) if os.path.isdir(FIG_DIR) else []
    return jsonify({
        "figures": [f for f in figs if f.endswith(".png")],
        "tables": {
            "tco":     os.path.exists(os.path.join(RESULTS_DIR,"tco_summary.json")),
            "network": os.path.exists(os.path.join(RESULTS_DIR,"network_test.json")),
            "domain":  os.path.exists(os.path.join(RESULTS_DIR,"domain_shift.json")),
            "shap":    os.path.exists(os.path.join(RESULTS_DIR,"shap_attribution.json")),
            "simclr":  os.path.exists(os.path.join(BASE_DIR,"checkpoints","simclr_history.json")),
        }
    })


@app.route("/results/figures")
def figures_list():
    figs = sorted(os.listdir(FIG_DIR)) if os.path.isdir(FIG_DIR) else []
    return jsonify({"figures": [{"name":f,"url":f"/figures/{f}"}
                                 for f in figs if f.endswith(".png")]})


@app.route("/results/figure/<name>")
def serve_figure(name):
    p = os.path.join(FIG_DIR, name)
    if not os.path.exists(p): abort(404)
    return send_file(p, mimetype="image/png")


@app.route("/results/tco")
def tco_results():
    return jsonify(_load_json(os.path.join(RESULTS_DIR,"tco_summary.json"),"TCO results"))


@app.route("/results/network")
def network_results():
    return jsonify(_load_json(os.path.join(RESULTS_DIR,"network_test.json"),"Network results"))


@app.route("/results/domain")
def domain_results():
    return jsonify(_load_json(os.path.join(RESULTS_DIR,"domain_shift.json"),"Domain results"))


@app.route("/results/simclr")
def simclr_results():
    p = os.path.join(BASE_DIR, "checkpoints", "simclr_history.json")
    if not os.path.exists(p):
        return jsonify([])
    return jsonify(_load_json(p, "SimCLR pre-training history"))


@app.route("/results/shap")
def shap_results():
    return jsonify(_load_json(os.path.join(RESULTS_DIR,"shap_attribution.json"),"SHAP results"))


# ── Background task runner ────────────────────────────────────────────────────
def _run_script(script_name: str, extra_args: list = None):
    scripts_map = {
        "train":         os.path.join(BASE_DIR, "scripts", "train.py"),
        "evaluate":      os.path.join(BASE_DIR, "scripts", "evaluate.py"),
        "ablation":      os.path.join(BASE_DIR, "scripts", "ablation.py"),
        "tco_simulation": os.path.join(BASE_DIR, "scripts", "tco_simulation.py"),
        "network_test":  os.path.join(BASE_DIR, "scripts", "network_test.py"),
        "generate":      os.path.join(BASE_DIR, "data", "generate_datasets.py"),
        "pretrain":      os.path.join(BASE_DIR, "scripts", "pretrain_simclr.py"),
    }
    script_path = scripts_map.get(script_name)
    if not script_path:
        with _lock:
            _run_status[script_name] = {"status": "error", "detail": "Unknown script"}
        return

    cmd = [sys.executable, script_path] + (extra_args or [])
    with _lock:
        _run_status[script_name] = {"status": "running", "started": time.time()}

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        elapsed = time.time() - _run_status[script_name].get("started", time.time())
        with _lock:
            _run_status[script_name] = {
                "status":     "done" if r.returncode == 0 else "error",
                "returncode": r.returncode,
                "stdout":     r.stdout[-4000:],
                "stderr":     r.stderr[-1000:],
                "elapsed":    round(elapsed, 1),
            }
        # Reload model if train just completed
        if script_name == "train" and r.returncode == 0:
            _load_model()
    except subprocess.TimeoutExpired:
        with _lock:
            _run_status[script_name] = {"status": "timeout"}
    except Exception as e:
        with _lock:
            _run_status[script_name] = {"status": "error", "detail": str(e)}


def _start_bg(script_name, extra_args=None):
    t = threading.Thread(target=_run_script, args=(script_name, extra_args), daemon=True)
    t.start()


@app.route("/run/pretrain", methods=["POST"])
def run_pretrain():
    _start_bg("pretrain", ["--epochs","100","--tau","0.5","--target-only"])
    return jsonify({"message":"SimCLR pre-training started (Contribution 2)",
                    "poll":"/run/status/pretrain"})


@app.route("/run/train", methods=["POST"])
def run_train():
    _start_bg("train")
    return jsonify({"message":"Training started","poll":"/run/status/train"})


@app.route("/run/evaluate", methods=["POST"])
def run_evaluate():
    _start_bg("evaluate")
    return jsonify({"message":"Evaluation started","poll":"/run/status/evaluate"})


@app.route("/run/ablation", methods=["POST"])
def run_ablation():
    _start_bg("ablation")
    return jsonify({"message":"Ablation started","poll":"/run/status/ablation"})


@app.route("/run/tco", methods=["POST"])
def run_tco():
    _start_bg("tco_simulation", ["--trials","10000","--horizon","5","--sensitivity","--national","5"])
    return jsonify({"message":"TCO simulation started","poll":"/run/status/tco_simulation"})


@app.route("/run/network-test", methods=["POST"])
def run_network_test():
    _start_bg("network_test")
    return jsonify({"message":"Network test started","poll":"/run/status/network_test"})


@app.route("/run/generate", methods=["POST"])
def run_generate():
    _start_bg("generate")
    return jsonify({"message":"Data generation started","poll":"/run/status/generate"})


@app.route("/run/status/<script>")
def run_status(script):
    with _lock:
        return jsonify(_run_status.get(script, {"status": "not_started"}))


# ── API documentation ─────────────────────────────────────────────────────────
@app.route("/docs")
def docs():
    endpoints = [
        {"method":"GET",  "path":"/",                    "description":"Dashboard SPA"},
        {"method":"GET",  "path":"/health",              "description":"Liveness + model info"},
        {"method":"POST", "path":"/predict",             "description":"Single-unit RUL + failure probability"},
        {"method":"POST", "path":"/predict/batch",       "description":"Batch prediction"},
        {"method":"GET",  "path":"/metrics",             "description":"Test metrics (AUPRC, RMSE, domain acc)"},
        {"method":"GET",  "path":"/metrics/latency",     "description":"Rolling inference latency stats"},
        {"method":"GET",  "path":"/metrics/history",     "description":"Per-epoch training history"},
        {"method":"GET",  "path":"/metrics/ablation",    "description":"Ablation study results"},
        {"method":"GET",  "path":"/results",             "description":"All results index"},
        {"method":"GET",  "path":"/results/figures",     "description":"List generated figures"},
        {"method":"GET",  "path":"/results/figure/<n>",  "description":"Serve figure PNG"},
        {"method":"GET",  "path":"/results/tco",         "description":"TCO/NPV simulation results"},
        {"method":"GET",  "path":"/results/network",     "description":"Network resilience results"},
        {"method":"GET",  "path":"/results/domain",      "description":"Domain shift diagnostics"},
        {"method":"GET",  "path":"/results/shap",        "description":"Feature attribution table"},
        {"method":"GET",  "path":"/results/simclr",      "description":"SimCLR pre-training history"},
        {"method":"POST", "path":"/run/pretrain",        "description":"Trigger SimCLR pre-training"},
        {"method":"POST", "path":"/run/train",           "description":"Trigger training pipeline"},
        {"method":"POST", "path":"/run/evaluate",        "description":"Trigger figure/table generation"},
        {"method":"POST", "path":"/run/ablation",        "description":"Trigger ablation study"},
        {"method":"POST", "path":"/run/tco",             "description":"Trigger TCO simulation"},
        {"method":"POST", "path":"/run/network-test",    "description":"Trigger network test"},
        {"method":"POST", "path":"/run/generate",        "description":"Trigger data generation"},
        {"method":"GET",  "path":"/run/status/<script>", "description":"Poll background task status"},
    ]
    return jsonify({
        "name": "NADiSSP Inference & Research API",
        "version": "1.0.0",
        "backend": "Flask (sklearn inference)",
        "endpoints": endpoints,
        "predict_schema": {
            "asset_id": "string",
            "asset_class": "one of cmaps_turbofan|ai4i_manufacturing|3w_offshore_well|espset_pump",
            "sequence": [{"pressure_1":0.5,"pressure_2":0.5,"vibration_x":0.3,
                           "vibration_y":0.3,"temperature_1":0.4,"temperature_2":0.4,
                           "motor_current":0.5,"frequency":0.6,"torque":0.55,
                           "rotational_speed":0.7}],
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"NADiSSP API starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)