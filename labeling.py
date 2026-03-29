import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


DISEASE_WEIGHTS = {
    "bronchitis": {
        "PM2.5": 0.16,
        "PM10": 0.10,
        "NO2": 0.25,
        "SO2": 0.16,
        "O3": 0.05,
        "Temperature": 0.10,
        "Humidity": 0.09,
        "WindSpeed": 0.05,
    },
    "pneumonia": {
        "PM2.5": 0.22,
        "PM10": 0.18,
        "NO2": 0.10,
        "SO2": 0.08,
        "O3": 0.05,
        "Temperature": 0.12,
        "Humidity": 0.15,
        "WindSpeed": 0.05,
    },
    "asthma": {
        "PM2.5": 0.14,
        "PM10": 0.08,
        "NO2": 0.18,
        "SO2": 0.10,
        "O3": 0.20,
        "Temperature": 0.10,
        "Humidity": 0.12,
        "WindSpeed": 0.05,
    },
    "lung_cancer": {
        "PM2.5": 0.28,
        "PM10": 0.12,
        "NO2": 0.18,
        "SO2": 0.12,
        "O3": 0.05,
        "Temperature": 0.08,
        "Humidity": 0.08,
        "WindSpeed": 0.05,
    },
    "copd": {
        "PM2.5": 0.20,
        "PM10": 0.10,
        "NO2": 0.20,
        "SO2": 0.15,
        "O3": 0.05,
        "Temperature": 0.10,
        "Humidity": 0.10,
        "WindSpeed": 0.05,
    },
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_continuous_targets(
    df,
    pollutant_cols=None,
    k: float = 8.0,
    thresholds=None,
):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    if pollutant_cols is None:
        pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "O3", "Temperature", "Humidity", "WindSpeed"]

    if thresholds is None:
        thresholds = {
            "bronchitis": 0.45,
            "pneumonia": 0.50,
            "asthma": 0.48,
            "lung_cancer": 0.55,
            "copd": 0.50,
        }

    # ✅ Keep only available columns
    pollutant_cols = [col for col in pollutant_cols if col in df.columns]

    if len(pollutant_cols) < 3:
        raise ValueError(f"Not enough pollutant columns found: {pollutant_cols}")

    work = df.copy()

    # Clean missing
    work[pollutant_cols] = work[pollutant_cols].replace([np.inf, -np.inf], np.nan)
    work[pollutant_cols] = work[pollutant_cols].fillna(work[pollutant_cols].median())

    # Normalize
    scaler = MinMaxScaler()
    norm_values = scaler.fit_transform(work[pollutant_cols])
    norm_df = pd.DataFrame(norm_values, columns=pollutant_cols, index=work.index)

    for disease, weights in DISEASE_WEIGHTS.items():

        # ✅ Only use available pollutants
        filtered_weights = {k: v for k, v in weights.items() if k in pollutant_cols}

        # Normalize weights to sum = 1
        total = sum(filtered_weights.values())
        normalized_weights = {k: v / total for k, v in filtered_weights.items()}

        weight_vec = np.array([normalized_weights[col] for col in pollutant_cols])

        raw_score = norm_df[pollutant_cols].values @ weight_vec

        base_risk = 1 / (1 + np.exp(-k * (raw_score - thresholds[disease])))

        work[f"{disease}_risk"] = np.clip(base_risk, 0, 1)

    return work, scaler


def save_weights_json(path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DISEASE_WEIGHTS, f, indent=2)