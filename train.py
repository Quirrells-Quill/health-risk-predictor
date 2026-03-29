# -----------------------------
# 🔥 CRITICAL FIX (DO NOT REMOVE)
# -----------------------------
import matplotlib
matplotlib.use('Agg')

# -----------------------------
# IMPORTS
# -----------------------------
import json
import joblib
import pandas as pd
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns

from labeling import generate_continuous_targets, DISEASE_WEIGHTS

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = "data/air_quality_health.csv"

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

MODEL_PATH = ART / "risk_model.joblib"
SCALER_PATH = ART / "scaler.joblib"
FEATURES_PATH = ART / "feature_columns.json"

STATUS = ART / "training_status.json"
SUMMARY = ART / "training_summary.json"
LOG = ART / "training.log"
STEP = ART / "training_step.json"
FEATURE_IMP = ART / "feature_importance.json"
META = ART / "meta_model.json"

PLOT_DIR = ART / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# -----------------------------
# LOGGING HELPERS (RESTORED)
# -----------------------------
def now():
    return datetime.now().isoformat()

def write_status(stage, pct, detail=""):
    STATUS.write_text(json.dumps({
        "stage": stage,
        "percent": pct,
        "detail": detail,
        "timestamp": now()
    }, indent=2))

def write_log(msg):
    line = f"{now()} | {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def write_step(data):
    data["timestamp"] = now()
    STEP.write_text(json.dumps(data, indent=2))

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
def extract_feature_importance(model, feature_cols):
    importances = {f: [] for f in feature_cols}

    for est in model.estimators_:
        stack = est
        for name, base in stack.named_estimators_.items():
            if hasattr(base, "feature_importances_"):
                fi = base.feature_importances_
                for i, f in enumerate(feature_cols):
                    importances[f].append(float(fi[i]))

    return {f: float(np.mean(v)) if v else 0.0 for f, v in importances.items()}

# -----------------------------
# META MODEL WEIGHTS
# -----------------------------
def extract_meta_weights(model):
    weights = []
    for est in model.estimators_:
        final = est.final_estimator_
        if hasattr(final, "coef_"):
            weights.append([float(x) for x in final.coef_.ravel()])
    return weights

# -----------------------------
# MODEL BUILD
# -----------------------------
def build_model():

    write_status("Model Build", 45, "Initializing models")
    write_log("Initializing base models")
    write_step({"event": "init_models"})

    xgb = XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, n_jobs=-1)
    rf = RandomForestRegressor(n_estimators=150, n_jobs=-1)

    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])

    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=7))
    ])

    stack = StackingRegressor(
        estimators=[
            ("xgb", xgb),
            ("rf", rf),
            ("ridge", ridge),
            ("knn", knn)
        ],
        final_estimator=Ridge(),
        cv=3,
        n_jobs=-1
    )

    return MultiOutputRegressor(stack, n_jobs=-1)

# -----------------------------
# CONFUSION MATRIX HELPER
# -----------------------------
def categorize(y):
    return [0 if v < 0.33 else 1 if v < 0.66 else 2 for v in y]

# -----------------------------
# MAIN
# -----------------------------
def main():

    if LOG.exists():
        LOG.unlink()

    write_status("Start", 1, "Training started")
    write_log("Loading dataset")

    df = pd.read_csv(DATA_PATH)
    df.rename(columns={"PM2_5": "PM2.5"}, inplace=True)

    FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "O3", "Temperature", "Humidity", "WindSpeed"]
    TARGETS = ["bronchitis_risk", "pneumonia_risk", "asthma_risk", "lung_cancer_risk", "copd_risk"]

    write_status("Labeling", 20, "Generating labels")
    labeled_df, scaler = generate_continuous_targets(df, FEATURES)
    joblib.dump(scaler, SCALER_PATH)

    X = labeled_df[FEATURES]
    y = labeled_df[TARGETS]

    write_status("Split", 35, "Splitting dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model()

    write_status("Training", 60, "Training models")

    for m in ["XGBoost", "RandomForest", "Ridge", "KNN"]:
        write_log(f"Training {m}")
        write_step({"event": "model_start", "model": m})
        time.sleep(0.2)

    model.fit(X_train, y_train)
    write_log("Model training complete")

    preds = model.predict(X_test)

    # -----------------------------
    # METRICS
    # -----------------------------
    write_status("Evaluation", 85, "Evaluating model")

    metrics = {}

    for i, col in enumerate(TARGETS):
        mae = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
        rmse = np.sqrt(((y_test.iloc[:, i] - preds[:, i])**2).mean())
        r2 = r2_score(y_test.iloc[:, i], preds[:, i])

        metrics[col] = {"MAE": mae, "RMSE": rmse, "R2": r2}

        write_log(f"{col}: R2={r2:.3f}")
        write_step({"event": "metric", "disease": col})

    # -----------------------------
    # VISUALIZATIONS
    # -----------------------------
    write_log("Generating plots")

    for i, col in enumerate(TARGETS):

        y_true = y_test.iloc[:, i]
        y_pred = preds[:, i]

        # Confusion Matrix
        cm = confusion_matrix(categorize(y_true), categorize(y_pred))
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {col}")
        plt.savefig(PLOT_DIR / f"confusion_{col}.png")
        plt.close()

        # Scatter
        plt.figure()
        plt.scatter(y_true, y_pred)
        plt.title(f"Actual vs Predicted - {col}")
        plt.savefig(PLOT_DIR / f"scatter_{col}.png")
        plt.close()

        # Residual
        plt.figure()
        plt.hist(y_true - y_pred, bins=30)
        plt.title(f"Residuals - {col}")
        plt.savefig(PLOT_DIR / f"residual_{col}.png")
        plt.close()

    # R2 Comparison
    plt.figure()
    plt.bar(metrics.keys(), [v["R2"] for v in metrics.values()])
    plt.title("R2 Score Comparison")
    plt.xticks(rotation=45)
    plt.savefig(PLOT_DIR / "r2_scores.png")
    plt.close()

    # Feature Importance
    fi = extract_feature_importance(model, FEATURES)
    FEATURE_IMP.write_text(json.dumps(fi, indent=2))

    plt.figure()
    plt.bar(fi.keys(), fi.values())
    plt.title("Feature Importance")
    plt.xticks(rotation=45)
    plt.savefig(PLOT_DIR / "feature_importance.png")
    plt.close()

    # -----------------------------
    # SAVE
    # -----------------------------
    joblib.dump(model, MODEL_PATH)

    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURES, f)

    SUMMARY.write_text(json.dumps({
        "feature_columns": FEATURES,
        "metrics": metrics,
        "weights": DISEASE_WEIGHTS
    }, indent=2))

    write_status("Done", 100, "Training complete")
    write_log("Training finished successfully")


if __name__ == "__main__":
    main()