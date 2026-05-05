"""
analyzer.py
───────────
Real-time time-score calculator for the Risk Scoring Engine.

Given a timestamp and event type, returns a time_score (0–1) representing
how risky that time period is historically.

Two modes:
  1. Prophet-based:     loads trained .pkl model, gets hourly forecast value,
                        normalizes to 0–1 against historical max
  2. Statistical fallback: loads .json hour distribution, looks up hour frequency

This is what the Risk Engine calls at inference time.
"""

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from loguru import logger
from functools import lru_cache
from typing import Optional

try:
    from prophet import Prophet
    _prophet_available = True
except ImportError:
    _prophet_available = False

TS_DIR     = Path(__file__).parent
MODELS_DIR = TS_DIR / "models_store"

EMERGENCY_TYPES = ["ACCIDENT", "FIRE", "FLOOD", "CRIME", "CROWD", "MEDICAL"]


# ── Model cache ───────────────────────────────────────────────────────────────
_prophet_models: dict = {}   # model_name → Prophet object
_stat_models:    dict = {}   # model_name → dict (statistical fallback)


def _load_prophet(model_name: str) -> Optional["Prophet"]:
    """Loads and caches a Prophet model from disk."""
    if model_name in _prophet_models:
        return _prophet_models[model_name]

    pkl_path = MODELS_DIR / f"prophet_{model_name}.pkl"
    if not pkl_path.exists():
        return None

    try:
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        _prophet_models[model_name] = model
        logger.info(f"Loaded Prophet model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load Prophet [{model_name}]: {e}")
        return None


def _load_stat_model(model_name: str) -> Optional[dict]:
    """Loads and caches a statistical fallback model from disk."""
    if model_name in _stat_models:
        return _stat_models[model_name]

    json_path = MODELS_DIR / f"stats_{model_name}.json"
    if not json_path.exists():
        return None

    try:
        data = json.loads(json_path.read_text())
        _stat_models[model_name] = data
        return data
    except Exception as e:
        logger.error(f"Failed to load stat model [{model_name}]: {e}")
        return None


# ── Core time-score calculation ───────────────────────────────────────────────
def get_time_score(
    timestamp: datetime,
    event_type: str = "overall",
) -> dict:
    """
    Returns a time_score (0.0–1.0) for a given datetime and event type.

    High score = historically dangerous time slot.
    Low  score = historically quiet time slot.

    Also returns an explanation dict for the Risk Engine's audit trail.

    Args:
        timestamp:  the datetime of the event
        event_type: one of our 7 classes, or "overall"

    Returns:
        {"time_score": float, "peak_hour": bool, "explanation": dict}
    """
    hour    = timestamp.hour
    weekday = timestamp.weekday()   # 0=Mon, 6=Sun

    # Try Prophet model first (most accurate)
    model_key = event_type if event_type in EMERGENCY_TYPES else "overall"
    prophet   = _load_prophet(model_key) or _load_prophet("overall")

    if prophet is not None and _prophet_available:
        return _score_with_prophet(prophet, timestamp, hour, weekday, model_key)

    # Try statistical fallback
    stat = _load_stat_model(model_key) or _load_stat_model("overall")
    if stat is not None:
        return _score_with_stats(stat, hour, weekday)

    # Last resort: hard-coded heuristics (same as Phase 1 simulator)
    return _score_with_heuristics(hour, weekday)


def _score_with_prophet(model, timestamp, hour, weekday, model_key) -> dict:
    """Score using a trained Prophet model."""
    try:
        future = pd.DataFrame({"ds": [timestamp]})
        forecast = model.predict(future)

        # yhat is the predicted incident count for this hour
        predicted = float(forecast["yhat"].iloc[0])
        predicted = max(predicted, 0.0)   # clip negatives

        # Normalize: we need 0–1 score.
        # Use yhat_upper as the max reference (95th percentile)
        yhat_upper = float(forecast["yhat_upper"].iloc[0])
        reference  = max(yhat_upper, 1.0)

        raw_score  = predicted / reference
        time_score = round(min(max(raw_score, 0.0), 1.0), 4)

        is_peak_hour = hour in _get_peak_hours(model)
        is_weekend   = weekday >= 5

        return {
            "time_score":   time_score,
            "peak_hour":    is_peak_hour,
            "weekend":      is_weekend,
            "method":       "prophet",
            "model_used":   model_key,
            "explanation": {
                "hour":           hour,
                "weekday":        weekday,
                "predicted_count": round(predicted, 2),
                "reference_max":  round(reference, 2),
                "is_peak_hour":   is_peak_hour,
            },
        }
    except Exception as e:
        logger.warning(f"Prophet scoring failed: {e}. Falling back to heuristics.")
        return _score_with_heuristics(hour, weekday)


def _score_with_stats(stat: dict, hour: int, weekday: int) -> dict:
    """Score using statistical hour distribution."""
    hour_dist    = {int(k): v for k, v in stat.get("hour_dist", {}).items()}
    peak_hours   = stat.get("peak_hours", [8, 9, 18, 19])

    if hour_dist:
        max_count  = max(hour_dist.values()) or 1
        hour_count = hour_dist.get(hour, 0)
        time_score = round(hour_count / max_count, 4)
    else:
        time_score = 0.5

    return {
        "time_score":  time_score,
        "peak_hour":   hour in peak_hours,
        "weekend":     weekday >= 5,
        "method":      "statistical",
        "explanation": {"hour": hour, "weekday": weekday, "peak_hours": peak_hours},
    }


def _score_with_heuristics(hour: int, weekday: int) -> dict:
    """Last-resort hard-coded heuristics (no trained model)."""
    if 8 <= hour <= 10 or 17 <= hour <= 20:
        score = 0.80      # rush hour
    elif 22 <= hour or hour <= 4:
        score = 0.70      # late night
    elif 11 <= hour <= 16:
        score = 0.40      # quiet afternoon
    else:
        score = 0.50

    # Weekend slight adjustment
    if weekday >= 5:
        score = round(score * 0.85, 4)

    return {
        "time_score":  score,
        "peak_hour":   (8 <= hour <= 10) or (17 <= hour <= 20),
        "weekend":     weekday >= 5,
        "method":      "heuristic",
        "explanation": {"hour": hour, "weekday": weekday},
    }


@lru_cache(maxsize=24)
def _get_peak_hours(model) -> list[int]:
    """
    Extracts the top-4 peak hours from a Prophet model's daily seasonality.
    lru_cache means we compute this once per model object.
    """
    try:
        future = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=24, freq="H")
        })
        forecast  = model.predict(future)
        daily_comp = forecast[["ds", "daily"]].copy()
        daily_comp["hour"] = daily_comp["ds"].dt.hour
        peak_hours = daily_comp.nlargest(4, "daily")["hour"].tolist()
        return peak_hours
    except Exception:
        return [8, 9, 17, 18]   # fallback


def get_hourly_risk_profile(event_type: str = "overall") -> dict:
    """
    Returns risk scores for all 24 hours of the day.
    Used by the heatmap and analytics endpoints.
    """
    base_date = datetime(2024, 1, 8)   # a Monday
    profile   = {}

    for hour in range(24):
        ts     = base_date.replace(hour=hour)
        result = get_time_score(ts, event_type)
        profile[hour] = {
            "time_score": result["time_score"],
            "is_peak":    result["peak_hour"],
        }

    return {
        "event_type":    event_type,
        "hourly_profile": profile,
        "peak_hours":    [h for h, v in profile.items() if v["is_peak"]],
    }


def get_models_status() -> dict:
    """Returns which models are loaded / available."""
    available = []
    for name in ["overall"] + EMERGENCY_TYPES:
        if (MODELS_DIR / f"prophet_{name}.pkl").exists():
            available.append(f"prophet:{name}")
        elif (MODELS_DIR / f"stats_{name}.json").exists():
            available.append(f"stats:{name}")

    return {
        "models_available":   available,
        "prophet_installed":  _prophet_available,
        "models_dir":         str(MODELS_DIR),
        "scoring_method":     "prophet" if _prophet_available and available else "heuristic",
    }