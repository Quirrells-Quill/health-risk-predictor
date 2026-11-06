# tests/test_health_risk.py
"""
Automated unit test suite for the AI-Based Respiratory Health Risk Prediction System.
Validates epidemiological weighting, sigmoid transformations, Bayesian odds personal factors,
pediatric clinical gates, and Flask web API endpoints.
"""

import math
import numpy as np
import pandas as pd
import pytest

from labeling import DISEASE_WEIGHTS, sigmoid, generate_continuous_targets
from personal_risk import (
    probability_to_odds,
    odds_to_probability,
    disease_multiplier,
    apply_personal_factors,
    format_as_percent,
)
from app import app


class TestLabelingAndEpidemiology:
    """Validates the continuous risk labeling and environmental weighting engine."""

    def test_disease_weights_coverage(self):
        expected_diseases = {"bronchitis", "pneumonia", "asthma", "lung_cancer", "copd"}
        assert set(DISEASE_WEIGHTS.keys()) == expected_diseases

        for disease, weights in DISEASE_WEIGHTS.items():
            assert "PM2.5" in weights
            assert "NO2" in weights
            assert sum(weights.values()) > 0

    def test_sigmoid_transformation(self):
        assert math.isclose(float(sigmoid(np.array([0.0]))[0]), 0.5, abs_tol=1e-5)
        assert float(sigmoid(np.array([10.0]))[0]) > 0.99
        assert float(sigmoid(np.array([-10.0]))[0]) < 0.01

    def test_generate_continuous_targets(self):
        mock_data = pd.DataFrame({
            "PM2.5": [12.0, 55.0, 150.0],
            "PM10": [25.0, 90.0, 220.0],
            "NO2": [10.0, 30.0, 80.0],
            "SO2": [5.0, 15.0, 45.0],
            "O3": [15.0, 40.0, 95.0],
            "Temperature": [22.0, 28.0, 35.0],
            "Humidity": [45.0, 60.0, 85.0],
            "WindSpeed": [3.5, 2.1, 0.8],
        })
        labeled_df, scaler = generate_continuous_targets(mock_data)

        for disease in ["bronchitis", "pneumonia", "asthma", "lung_cancer", "copd"]:
            col = f"{disease}_risk"
            assert col in labeled_df.columns
            assert (labeled_df[col] >= 0.0).all()
            assert (labeled_df[col] <= 1.0).all()
            # Monotonicity: higher pollution should produce higher risk
            assert labeled_df[col].iloc[2] > labeled_df[col].iloc[0]


class TestPersonalRiskOddsModel:
    """Validates Bayesian clinical odds-ratio multiplier updates."""

    def test_odds_probability_bijection(self):
        test_probs = [0.01, 0.1, 0.5, 0.85, 0.99]
        for p in test_probs:
            odds = probability_to_odds(p)
            p_reconstructed = odds_to_probability(odds)
            assert math.isclose(p, p_reconstructed, rel_tol=1e-4)

    def test_pediatric_copd_age_gate(self):
        profile = {"age": 14, "smoker": "heavy", "indoor_pollution": True}
        mult, notes = disease_multiplier("copd", profile)
        assert mult == 0.0
        assert any("Age gate" in n for n in notes)

        final_probs, details = apply_personal_factors({"copd": 0.45}, profile)
        assert final_probs["copd"] == 0.0
        assert details["copd"]["final_prob"] == 0.0

    def test_smoking_multipliers(self):
        heavy_profile = {"age": 50, "smoker": "heavy"}
        mult_heavy, _ = disease_multiplier("lung_cancer", heavy_profile)
        assert mult_heavy >= 23.0

        light_profile = {"age": 50, "smoker": "light"}
        mult_light, _ = disease_multiplier("lung_cancer", light_profile)
        assert mult_light >= 6.0

    def test_asthma_genetic_predisposition(self):
        profile_fam = {"age": 25, "family_history_asthma": True}
        mult_fam, _ = disease_multiplier("asthma", profile_fam)
        assert mult_fam >= 4.0

    def test_format_as_percent(self):
        res = format_as_percent({"asthma": 0.45678, "copd": 0.1234})
        assert res["asthma"] == 45.68
        assert res["copd"] == 12.34


class TestFlaskEndpoints:
    """Validates Flask HTTP routes and inference endpoints."""

    @pytest.fixture
    def client(self):
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_home_page(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert b"Respiratory Health Risk" in res.data or b"Health Risk Engine" in res.data

    def test_dashboard_page(self, client):
        res = client.get("/dashboard")
        assert res.status_code == 200

    def test_training_dashboard_page(self, client):
        res = client.get("/training-dashboard")
        assert res.status_code == 200

    def test_predict_endpoint(self, client):
        payload = {
            "pm25": 45.0,
            "pm10": 80.0,
            "no2": 35.0,
            "so2": 15.0,
            "o3": 25.0,
            "temperature": 26.0,
            "humidity": 65.0,
            "windspeed": 2.5,
            "age": 35,
            "gender": "male",
            "smoker": "none",
            "income": "medium",
            "family_history": False,
            "indoor_pollution": False,
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert "asthma" in data
        assert "copd" in data
        assert "lung_cancer" in data
        assert "bronchitis" in data
        assert "pneumonia" in data
        assert 0.0 <= data["asthma"] <= 100.0

    def test_explain_endpoint(self, client):
        payload = {
            "pm25": 35.0,
            "pm10": 60.0,
            "no2": 25.0,
            "so2": 10.0,
            "o3": 20.0,
            "temperature": 24.0,
            "humidity": 50.0,
            "windspeed": 3.0,
            "age": 28,
            "gender": "female",
            "smoker": "light",
            "income": "low",
            "family_history": True,
            "indoor_pollution": True,
        }
        res = client.post("/explain", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert "normalized_inputs" in data
        assert "contributions" in data
        assert "final_probs" in data
        assert "personal_details" in data
