import json
import joblib
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify

from personal_risk import apply_personal_factors, format_as_percent
from labeling import DISEASE_WEIGHTS

app = Flask(__name__)

ARTIFACTS = Path("artifacts")

MODEL_PATH = ARTIFACTS / "risk_model.joblib"
SCALER_PATH = ARTIFACTS / "scaler.joblib"
FEATURES_PATH = ARTIFACTS / "feature_columns.json"

STATUS_PATH = ARTIFACTS / "training_status.json"
SUMMARY_PATH = ARTIFACTS / "training_summary.json"
LOG_PATH = ARTIFACTS / "training.log"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)


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
        "gender": data["gender"],
        "smoker": data["smoker"],
        "income": data["income"],
        "family_history_asthma": data["family_history"],
        "indoor_pollution": data["indoor_pollution"],
    }

    X = pd.DataFrame([row], columns=FEATURE_COLS)
    base_pred = model.predict(X)[0]

    base_probs = {
        "bronchitis": base_pred[0],
        "pneumonia": base_pred[1],
        "asthma": base_pred[2],
        "lung_cancer": base_pred[3],
        "copd": base_pred[4],
    }

    final_probs, _ = apply_personal_factors(base_probs, profile)
    results = format_as_percent(final_probs)

    return jsonify(results)


@app.route("/explain", methods=["POST"])
def explain():

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
        "gender": data["gender"],
        "smoker": data["smoker"],
        "income": data["income"],
        "family_history_asthma": data["family_history"],
        "indoor_pollution": data["indoor_pollution"],
    }

    X = pd.DataFrame([row], columns=FEATURE_COLS)

    # ---------- NORMALIZATION ----------
    norm_vals = scaler.transform(X)[0]
    norm_map = dict(zip(FEATURE_COLS, norm_vals))

    # ---------- WEIGHTED CONTRIBUTIONS ----------
    contributions = {}

    for disease, weights in DISEASE_WEIGHTS.items():
        contrib_list = []
        score = 0

        for f in FEATURE_COLS:
            val = norm_map[f]
            w = weights.get(f, 0)
            c = val * w
            score += c

            contrib_list.append({
                "feature": f,
                "value": round(val,4),
                "weight": w,
                "contribution": round(c,4)
            })

        # sigmoid FIX
        import math
        k = 5
        threshold = 0.6
        sig = 1/(1+math.exp(-k*(score-threshold)))

        contributions[disease] = {
            "score": round(score,4),
            "sigmoid": round(sig,4),
            "details": contrib_list
        }

    # ---------- MODEL OUTPUT ----------
    base_pred = model.predict(X)[0]

    base_probs = {
        "bronchitis": base_pred[0],
        "pneumonia": base_pred[1],
        "asthma": base_pred[2],
        "lung_cancer": base_pred[3],
        "copd": base_pred[4],
    }

    final_probs, personal = apply_personal_factors(base_probs, profile)

    return jsonify({
        "normalized_inputs": norm_map,
        "contributions": contributions,
        "base_probs": base_probs,
        "final_probs": final_probs,
        "personal_details": personal
    })
@app.route("/model-breakdown", methods=["POST"])
def model_breakdown():
    data = request.json

    X = pd.DataFrame([data], columns=FEATURE_COLS)

    breakdown = get_model_breakdown(X)

    return jsonify(breakdown)

@app.route("/api/training-step")
def api_training_step():
    p = ARTIFACTS / "training_step.json"
    if p.exists():
        return jsonify(json.loads(p.read_text()))
    return jsonify({})

@app.route("/api/feature-importance")
def api_feature_importance():
    p = ARTIFACTS / "feature_importance.json"
    return jsonify(json.loads(p.read_text())) if p.exists() else jsonify({})

@app.route("/api/meta-weights")
def api_meta_weights():
    p = ARTIFACTS / "meta_model.json"
    return jsonify(json.loads(p.read_text())) if p.exists() else jsonify({})

# ---------------------------
# TRAINING DASHBOARD APIs
# ---------------------------
@app.route("/api/training-status")
def training_status():
    if STATUS_PATH.exists():
        return jsonify(json.loads(STATUS_PATH.read_text()))
    return jsonify({"stage": "No status", "percent": 0})


@app.route("/api/training-summary")
def training_summary():
    if SUMMARY_PATH.exists():
        return jsonify(json.loads(SUMMARY_PATH.read_text()))
    return jsonify({})

def get_model_breakdown(X):
    breakdown = {}

    for est in model.estimators_:
        stack = est

        preds = {}

        for name, base in stack.named_estimators_.items():
            preds[name] = float(base.predict(X)[0])

        final = float(stack.final_estimator_.predict(
            pd.DataFrame([list(preds.values())])
        )[0])

        breakdown = {
            "base_models": preds,
            "meta_output": final
        }

    return breakdown

@app.route("/api/training-log")
def training_log():
    if LOG_PATH.exists():
        lines = LOG_PATH.read_text().splitlines()[-200:]
        return jsonify({"lines": lines})
    return jsonify({"lines": []})


if __name__ == "__main__":
    app.run(debug=True)