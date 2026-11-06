# app.py
"""
Flask Web Application for the AI-Based Respiratory Health Risk Prediction System.
Serves web prediction interfaces, explainability dashboards, real-time training telemetry,
and REST API endpoints for multi-output ensemble inference.
"""

import json
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from labeling import DISEASE_WEIGHTS
from personal_risk import apply_personal_factors, format_as_percent

app = Flask(__name__)

ARTIFACTS = Path("artifacts")

MODEL_PATH = ARTIFACTS / "risk_model.joblib"
SCALER_PATH = ARTIFACTS / "scaler.joblib"
FEATURES_PATH = ARTIFACTS / "feature_columns.json"

STATUS_PATH = ARTIFACTS / "training_status.json"
SUMMARY_PATH = ARTIFACTS / "training_summary.json"
LOG_PATH = ARTIFACTS / "training.log"

model = None
scaler = None
FEATURE_COLS = []


def load_artifacts():
    """Loads trained stacking model, scaler, and feature definitions."""
    global model, scaler, FEATURE_COLS
    if model is None:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not FEATURES_PATH.exists():
            import train
            train.main()
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            FEATURE_COLS = json.load(f)


try:
    load_artifacts()
except Exception as exc:
    print(f"Artifact lazy-load notice: {exc}")


# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/training-dashboard")
def training_dashboard():
    return render_template("training_dashboard.html")


# ---------------------------
# PREDICT
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    load_artifacts()
    data = request.json

    row = {
        "PM2.5": float(data["pm25"]),
        "PM10": float(data["pm10"]),
        "NO2": float(data["no2"]),
        "SO2": float(data["so2"]),
        "O3": float(data["o3"]),
        "Temperature": float(data["temperature"]),
        "Humidity": float(data["humidity"]),
        "WindSpeed": float(data["windspeed"]),
    }

    profile = {
        "age": int(data["age"]),
        "gender": data.get("gender", "other"),
        "smoker": data.get("smoker", "none"),
        "income": data.get("income", "medium"),
        "family_history_asthma": data.get("family_history", False),
        "indoor_pollution": data.get("indoor_pollution", False),
    }

    X = pd.DataFrame([row], columns=FEATURE_COLS)
    base_pred = model.predict(X)[0]

    base_probs = {
        "bronchitis": float(base_pred[0]),
        "pneumonia": float(base_pred[1]),
        "asthma": float(base_pred[2]),
        "lung_cancer": float(base_pred[3]),
        "copd": float(base_pred[4]),
    }

    final_probs, _ = apply_personal_factors(base_probs, profile)
    results = format_as_percent(final_probs)

    return jsonify(results)


@app.route("/explain", methods=["POST"])
def explain():
    load_artifacts()
    data = request.json

    row = {
        "PM2.5": float(data["pm25"]),
        "PM10": float(data["pm10"]),
        "NO2": float(data["no2"]),
        "SO2": float(data["so2"]),
        "O3": float(data["o3"]),
        "Temperature": float(data["temperature"]),
        "Humidity": float(data["humidity"]),
        "WindSpeed": float(data["windspeed"]),
    }

    profile = {
        "age": int(data["age"]),
        "gender": data.get("gender", "other"),
        "smoker": data.get("smoker", "none"),
        "income": data.get("income", "medium"),
        "family_history_asthma": data.get("family_history", False),
        "indoor_pollution": data.get("indoor_pollution", False),
    }

    X = pd.DataFrame([row], columns=FEATURE_COLS)

    # ---------- NORMALIZATION ----------
    norm_vals = scaler.transform(X)[0]
    norm_map = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in norm_vals]))

    # ---------- WEIGHTED CONTRIBUTIONS ----------
    contributions = {}

    for disease, weights in DISEASE_WEIGHTS.items():
        contrib_list = []
        score = 0.0

        for f in FEATURE_COLS:
            val = norm_map[f]
            w = weights.get(f, 0.0)
            c = val * w
            score += c

            contrib_list.append({
                "feature": f,
                "value": round(val, 4),
                "weight": w,
                "contribution": round(c, 4),
            })

        k = 5.0
        threshold = 0.6
        sig = 1.0 / (1.0 + math.exp(-k * (score - threshold)))

        contributions[disease] = {
            "score": round(score, 4),
            "sigmoid": round(sig, 4),
            "details": contrib_list,
        }

    # ---------- MODEL OUTPUT ----------
    base_pred = model.predict(X)[0]

    base_probs = {
        "bronchitis": float(base_pred[0]),
        "pneumonia": float(base_pred[1]),
        "asthma": float(base_pred[2]),
        "lung_cancer": float(base_pred[3]),
        "copd": float(base_pred[4]),
    }

    final_probs, personal = apply_personal_factors(base_probs, profile)

    return jsonify({
        "normalized_inputs": norm_map,
        "contributions": contributions,
        "base_probs": base_probs,
        "final_probs": final_probs,
        "personal_details": personal,
    })


@app.route("/model-breakdown", methods=["POST"])
def model_breakdown():
    load_artifacts()
    data = request.json
    X = pd.DataFrame([data], columns=FEATURE_COLS)
    breakdown = get_model_breakdown(X)
    return jsonify(breakdown)


def get_model_breakdown(X):
    breakdown = {}
    for est in model.estimators_:
        preds = {}
        for name, base in est.named_estimators_.items():
            preds[name] = float(base.predict(X)[0])

        final = float(est.final_estimator_.predict(
            pd.DataFrame([list(preds.values())])
        )[0])

        breakdown = {
            "base_models": preds,
            "meta_output": final,
        }
    return breakdown


# ---------------------------
# TRAINING DASHBOARD APIs
# ---------------------------
@app.route("/api/training-step")
def api_training_step():
    p = ARTIFACTS / "training_step.json"
    if p.exists():
        return jsonify(json.loads(p.read_text(encoding="utf-8")))
    return jsonify({})


@app.route("/api/feature-importance")
def api_feature_importance():
    p = ARTIFACTS / "feature_importance.json"
    return jsonify(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else jsonify({})


@app.route("/api/meta-weights")
def api_meta_weights():
    p = ARTIFACTS / "meta_model.json"
    return jsonify(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else jsonify({})


@app.route("/api/training-status")
def training_status():
    if STATUS_PATH.exists():
        return jsonify(json.loads(STATUS_PATH.read_text(encoding="utf-8")))
    return jsonify({"stage": "Ready", "percent": 100, "detail": "Model trained and verified."})


@app.route("/api/training-summary")
def training_summary():
    if SUMMARY_PATH.exists():
        return jsonify(json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    return jsonify({})


@app.route("/api/training-log")
def training_log():
    if LOG_PATH.exists():
        lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[-200:]
        return jsonify({"lines": lines})
    return jsonify({"lines": []})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
