import numpy as np


def probability_to_odds(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return p / (1.0 - p)


def odds_to_probability(odds: float) -> float:
    odds = max(float(odds), 1e-12)
    return odds / (1.0 + odds)


def apply_personal_factors(base_probs: dict, profile: dict) -> dict:
    """
    base_probs: {'bronchitis': 0.21, 'pneumonia': 0.55, ...}
    profile example:
    {
        "age": 42,
        "smoker": "heavy",   # none / light / heavy
        "income": "low",     # low / medium / high
        "family_history_asthma": True,
        "indoor_pollution": True
    }
    """
    age = int(profile.get("age", 0))
    smoker = str(profile.get("smoker", "none")).lower()
    income = str(profile.get("income", "medium")).lower()
    family_history_asthma = bool(profile.get("family_history_asthma", False))
    indoor_pollution = bool(profile.get("indoor_pollution", False))

    final_probs = {}

    for disease, p_base in base_probs.items():
        p_base = float(np.clip(p_base, 1e-6, 1 - 1e-6))

        if disease == "copd" and age < 18:
            final_probs[disease] = 0.0
            continue

        multiplier = 1.0

        if disease == "lung_cancer":
            if smoker == "heavy":
                multiplier *= 23.0
            elif smoker == "light":
                multiplier *= 6.0

        if disease == "copd":
            if smoker == "heavy":
                multiplier *= 4.0
            elif smoker == "light":
                multiplier *= 2.0
            if income == "low" or indoor_pollution:
                multiplier *= 2.44

        if disease == "pneumonia":
            if smoker in {"light", "heavy"}:
                multiplier *= 2.0
            if age < 5 or age > 65:
                multiplier *= 1.5

        if disease == "asthma":
            if family_history_asthma:
                multiplier *= 4.0
            if age < 12:
                multiplier *= 1.3

        if disease == "bronchitis":
            if smoker in {"light", "heavy"}:
                multiplier *= 2.0
            if indoor_pollution:
                multiplier *= 1.5

        odds_base = probability_to_odds(p_base)
        odds_new = odds_base * multiplier
        final_probs[disease] = float(np.clip(odds_to_probability(odds_new), 0.0, 1.0))

    return final_probs


def format_as_percent(prob_dict: dict) -> dict:
    return {k: round(v * 100, 2) for k, v in prob_dict.items()}

import numpy as np

def probability_to_odds(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return p / (1.0 - p)

def odds_to_probability(odds: float) -> float:
    odds = max(float(odds), 1e-12)
    return odds / (1.0 + odds)

def disease_multiplier(disease: str, profile: dict) -> tuple[float, list[str]]:
    age = int(profile.get("age", 0))
    smoker = str(profile.get("smoker", "none")).lower()
    income = str(profile.get("income", "medium")).lower()
    gender = str(profile.get("gender", "other")).lower()
    family_history_asthma = bool(profile.get("family_history_asthma", False))
    indoor_pollution = bool(profile.get("indoor_pollution", False))

    multiplier = 1.0
    notes = []

    if disease == "copd" and age < 18:
        return 0.0, ["Age gate applied: COPD risk forced to 0 for age < 18"]

    if disease == "lung_cancer":
        if smoker == "heavy":
            multiplier *= 23.0
            notes.append("Heavy smoking multiplier x23")
        elif smoker == "light":
            multiplier *= 6.0
            notes.append("Light smoking multiplier x6")

    if disease == "copd":
        if smoker == "heavy":
            multiplier *= 4.0
            notes.append("Heavy smoking multiplier x4")
        elif smoker == "light":
            multiplier *= 2.0
            notes.append("Light smoking multiplier x2")
        if income == "low" or indoor_pollution:
            multiplier *= 2.44
            notes.append("Low income / indoor pollution multiplier x2.44")

    if disease == "pneumonia":
        if smoker in {"light", "heavy"}:
            multiplier *= 2.0
            notes.append("Smoking multiplier x2")
        if age < 5 or age > 65:
            multiplier *= 1.5
            notes.append("Age vulnerability multiplier x1.5")

    if disease == "asthma":
        if family_history_asthma:
            multiplier *= 4.0
            notes.append("Family history multiplier x4")

    if disease == "bronchitis":
        if smoker in {"light", "heavy"}:
            multiplier *= 2.0
            notes.append("Smoking multiplier x2")
        if indoor_pollution:
            multiplier *= 1.5
            notes.append("Indoor pollution multiplier x1.5")

    if gender == "male":
        if disease in {"lung_cancer", "copd"}:
            multiplier *= 1.05
            notes.append("Gender prior: male x1.05 for lung/cross-sectional respiratory risk")
    elif gender == "female":
        if disease == "asthma":
            multiplier *= 1.05
            notes.append("Gender prior: female x1.05 for asthma sensitivity")
        if disease == "pneumonia":
            multiplier *= 1.02
            notes.append("Gender prior: female x1.02 for pneumonia sensitivity")

    return multiplier, notes

def apply_personal_factors(base_probs: dict, profile: dict) -> tuple[dict, dict]:
    final_probs = {}
    details = {}

    for disease, p_base in base_probs.items():
        p_base = float(np.clip(p_base, 1e-6, 1 - 1e-6))
        mult, notes = disease_multiplier(disease, profile)

        if mult == 0.0:
            final_probs[disease] = 0.0
            details[disease] = {
                "base_prob": p_base,
                "multiplier": 0.0,
                "final_prob": 0.0,
                "delta_points": -p_base * 100,
                "relative_change_pct": -100.0,
                "notes": notes,
            }
            continue

        odds_base = probability_to_odds(p_base)
        odds_new = odds_base * mult
        p_final = odds_to_probability(odds_new)

        final_probs[disease] = float(np.clip(p_final, 0.0, 1.0))
        details[disease] = {
            "base_prob": p_base,
            "multiplier": mult,
            "final_prob": final_probs[disease],
            "delta_points": (final_probs[disease] - p_base) * 100,
            "relative_change_pct": ((final_probs[disease] - p_base) / p_base * 100) if p_base > 0 else 0.0,
            "notes": notes,
        }

    return final_probs, details

def format_as_percent(prob_dict: dict) -> dict:
    return {k: round(v * 100, 2) for k, v in prob_dict.items()}