"""
prophet_model.py
────────────────
Facebook Prophet model for predicting incident peaks.

What it predicts:
  - Peak hours: which hours of the day have most incidents
  - Peak days: which days of the week are highest risk
  - Trend: is the city getting safer or more dangerous over time?
  - Forecast: how many incidents to expect in the next 24h / 7 days

Why Prophet?
  - Handles daily + weekly seasonality automatically (perfect for city data)
  - Robust to missing data (we won't have data for every hour)
  - Fast to train: <5 seconds on 30 days of data
  - Produces uncertainty intervals (not just point estimates)
  - Works without GPU

Two models trained:
  1. Overall incidents model — predicts total incident count per hour
  2. Per-type models (ACCIDENT, FIRE, etc.) — predict type-specific peaks

MLflow tracks:
  - Hyperparameters
  - Train MAE, RMSE
  - Forecast plots as artifacts
"""

import json
import pickle
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
from rich.console import Console
from rich.table import Table

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    _prophet_available = True
except ImportError:
    _prophet_available = False
    logger.warning("prophet not installed. Time-series will use statistical fallback.")

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    _sklearn_available = True
except ImportError:
    _sklearn_available = False

console = Console()

# ── Paths ─────────────────────────────────────────────────────────────────────
TS_DIR      = Path(__file__).parent
MODELS_DIR  = TS_DIR / "models_store"
MLFLOW_DIR  = Path(__file__).parent.parent.parent / "mlruns"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EMERGENCY_TYPES = ["ACCIDENT", "FIRE", "FLOOD", "CRIME", "CROWD", "MEDICAL"]


# ── Data preparation ──────────────────────────────────────────────────────────
def prepare_prophet_df(
    events: list[dict],
    event_type_filter: str = None,
    freq: str = "H",           # "H" = hourly, "D" = daily
) -> pd.DataFrame:
    """
    Converts raw event dicts into Prophet's required format:
      ds (datetime) | y (count)

    Prophet requires exactly these two columns — named ds and y.

    Args:
        events:             list of event dicts (from DB or simulator)
        event_type_filter:  if set, only count events of this type
        freq:               aggregation frequency ('H' = hourly, 'D' = daily)

    Returns:
        DataFrame with columns [ds, y] ready for Prophet.fit()
    """
    if not events:
        return pd.DataFrame(columns=["ds", "y"])

    df = pd.DataFrame(events)

    # Parse timestamps
    df["ds"] = pd.to_datetime(df["occurred_at"])
    df["ds"] = df["ds"].dt.tz_localize(None)   # Prophet needs timezone-naive

    # Filter by type if requested
    if event_type_filter:
        df = df[df["event_type"] == event_type_filter]

    if df.empty:
        return pd.DataFrame(columns=["ds", "y"])

    # Count events per time bucket
    df = df.set_index("ds").resample(freq).size().reset_index()
    df.columns = ["ds", "y"]

    # Fill gaps with 0 (Prophet needs continuous time series)
    full_range = pd.date_range(
        start=df["ds"].min(),
        end=df["ds"].max(),
        freq=freq,
    )
    df = df.set_index("ds").reindex(full_range, fill_value=0).reset_index()
    df.columns = ["ds", "y"]

    return df


def generate_synthetic_ts_data(days: int = 60) -> list[dict]:
    """
    Generates synthetic time-series data for training Prophet when DB is empty.
    Uses the same time-of-day patterns as our simulator.
    """
    import random

    ETYPES = ["ACCIDENT", "FIRE", "FLOOD", "CRIME", "CROWD", "MEDICAL"]

    # Hour weights (same as simulator)
    hour_weights = [1]*24
    for h in [8,9,10,17,18,19,20]: hour_weights[h] = 3.0
    for h in [22,23,0,1]:           hour_weights[h] = 2.0
    for h in [3,4,5]:               hour_weights[h] = 0.3

    events = []
    base   = datetime.utcnow() - timedelta(days=days)

    for day in range(days):
        date         = base + timedelta(days=day)
        weekday_mult = 1.3 if date.weekday() < 5 else 0.8
        n_events     = int(random.gauss(40, 8) * weekday_mult)

        for _ in range(max(n_events, 5)):
            hour   = random.choices(range(24), weights=hour_weights, k=1)[0]
            minute = random.randint(0, 59)
            ts     = date.replace(hour=hour, minute=minute, second=random.randint(0,59))
            etype  = random.choice(ETYPES)
            events.append({"occurred_at": ts.isoformat(), "event_type": etype})

    logger.info(f"Generated {len(events)} synthetic events for time-series training")
    return events


# ── Model training ────────────────────────────────────────────────────────────
def train_prophet_model(
    events: list[dict],
    model_name: str = "overall",
    event_type: str = None,
) -> dict:
    """
    Trains a single Prophet model and saves it to disk + MLflow.

    Args:
        events:     list of event dicts with 'occurred_at' and 'event_type'
        model_name: label for this model ('overall', 'ACCIDENT', 'FIRE', etc.)
        event_type: if set, filters events to this type before training

    Returns:
        dict with model metrics and forecast summary
    """
    if not _prophet_available:
        logger.warning("Prophet not available — skipping training, using statistical model")
        return _train_statistical_fallback(events, model_name, event_type)

    logger.info(f"Training Prophet model: [{model_name}]")

    # Prepare data
    df = prepare_prophet_df(events, event_type_filter=event_type, freq="H")

    if len(df) < 48:   # need at least 2 days of hourly data
        logger.warning(f"Not enough data for [{model_name}]: {len(df)} rows. Need ≥48.")
        return {"model_name": model_name, "status": "skipped", "reason": "insufficient_data"}

    # Train / test split (last 24h as holdout)
    split_idx = max(len(df) - 24, int(len(df) * 0.85))
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    # Build Prophet model
    # changepoint_prior_scale: flexibility of trend (higher = more flexible)
    # seasonality_prior_scale: strength of seasonality components
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,    # we don't have years of data
        interval_width=0.90,         # 90% confidence intervals
    )

    # Add Indian holiday effects (major festivals affect incident patterns)
    model.add_country_holidays(country_name="IN")

    # Suppress Prophet's verbose output
    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    # Fit
    model.fit(train_df)

    # Forecast on test period
    future_test = model.make_future_dataframe(periods=len(test_df), freq="H")
    forecast    = model.predict(future_test)

    # Evaluate on test set
    test_preds  = forecast.iloc[-len(test_df):]["yhat"].clip(lower=0).values
    test_actual = test_df["y"].values

    mae  = float(mean_absolute_error(test_actual, test_preds)) if _sklearn_available else 0.0
    rmse = float(np.sqrt(mean_squared_error(test_actual, test_preds))) if _sklearn_available else 0.0

    # Save model
    model_path = MODELS_DIR / f"prophet_{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Extract peak hour insights
    peak_insights = _extract_peak_insights(model, forecast)

    # MLflow logging
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("emergency-timeseries")

    with mlflow.start_run(run_name=f"prophet-{model_name}-{datetime.now().strftime('%H%M')}"):
        mlflow.log_params({
            "model_name":                model_name,
            "event_type":                event_type or "all",
            "changepoint_prior_scale":   0.05,
            "seasonality_prior_scale":   10.0,
            "train_rows":                len(train_df),
            "test_rows":                 len(test_df),
        })
        mlflow.log_metrics({"test_mae": mae, "test_rmse": rmse})
        mlflow.log_artifact(str(model_path))

    logger.success(
        f"Prophet [{model_name}] trained | MAE={mae:.2f} | RMSE={rmse:.2f}"
    )

    return {
        "model_name":    model_name,
        "status":        "trained",
        "train_rows":    len(train_df),
        "test_mae":      round(mae, 3),
        "test_rmse":     round(rmse, 3),
        "peak_insights": peak_insights,
        "model_path":    str(model_path),
    }


def train_all_models(events: list[dict]) -> dict:
    """
    Trains Prophet models for:
    - Overall incidents
    - Each emergency type (ACCIDENT, FIRE, FLOOD, CRIME, CROWD, MEDICAL)

    Returns dict of all results.
    """
    results = {}

    # Overall model
    results["overall"] = train_prophet_model(events, "overall")

    # Per-type models
    for etype in EMERGENCY_TYPES:
        type_events = [e for e in events if e.get("event_type") == etype]
        if len(type_events) < 20:
            logger.warning(f"Skipping [{etype}]: only {len(type_events)} events")
            continue
        results[etype] = train_prophet_model(events, etype, event_type=etype)

    # Save results summary
    summary_path = MODELS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "peak_insights"}
         for k, v in results.items()},
        indent=2
    ))

    return results


def _extract_peak_insights(model, forecast: pd.DataFrame) -> dict:
    """
    Extracts human-readable peak insights from a trained Prophet model.
    These become the 'explanation' field in risk scores.
    """
    # Daily seasonality component tells us hour-of-day patterns
    future_day = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=24, freq="H")
    })
    day_forecast = model.predict(future_day)

    hourly_trend = day_forecast[["ds", "daily"]].copy()
    hourly_trend["hour"] = hourly_trend["ds"].dt.hour
    hourly_trend = hourly_trend.sort_values("daily", ascending=False)

    peak_hours = hourly_trend.head(4)["hour"].tolist()
    safe_hours = hourly_trend.tail(4)["hour"].tolist()

    # Weekly seasonality
    future_week = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=7, freq="D")
    })
    week_forecast = model.predict(future_week)
    week_forecast["weekday"] = week_forecast["ds"].dt.day_name()
    peak_day = week_forecast.loc[week_forecast["weekly"].idxmax(), "weekday"]

    return {
        "peak_hours":         peak_hours,
        "safe_hours":         safe_hours,
        "peak_day_of_week":   peak_day,
        "overall_trend":      "rising" if forecast["trend"].iloc[-1] > forecast["trend"].iloc[0] else "stable",
    }


# ── Statistical fallback ──────────────────────────────────────────────────────
def _train_statistical_fallback(events, model_name, event_type):
    """
    Pure numpy/pandas fallback when Prophet is not installed.
    Calculates hour-of-day and day-of-week patterns from historical data.
    """
    if not events:
        return {"model_name": model_name, "status": "fallback_empty"}

    df = pd.DataFrame(events)
    df["ds"] = pd.to_datetime(df["occurred_at"])
    if event_type:
        df = df[df["event_type"] == event_type]

    if df.empty:
        return {"model_name": model_name, "status": "fallback_empty"}

    df["hour"]    = df["ds"].dt.hour
    df["weekday"] = df["ds"].dt.dayofweek

    hour_counts    = df["hour"].value_counts().sort_index()
    weekday_counts = df["weekday"].value_counts().sort_index()

    peak_hours   = hour_counts.nlargest(4).index.tolist()
    peak_weekday = weekday_counts.idxmax()

    # Save as JSON (no pickle needed)
    model_data = {
        "model_name":  model_name,
        "type":        "statistical",
        "hour_dist":   hour_counts.to_dict(),
        "weekday_dist": weekday_counts.to_dict(),
        "peak_hours":  peak_hours,
        "peak_weekday": int(peak_weekday),
    }
    model_path = MODELS_DIR / f"stats_{model_name}.json"
    model_path.write_text(json.dumps(model_data, indent=2))

    logger.success(f"Statistical model [{model_name}] saved. Peak hours: {peak_hours}")
    return {
        "model_name":    model_name,
        "status":        "statistical_fallback",
        "peak_hours":    peak_hours,
        "model_path":    str(model_path),
    }