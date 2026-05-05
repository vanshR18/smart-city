"""
scorer.py
─────────
Unified Risk Scoring Engine — fuses all signals into one explainable score.

Formula:
  risk_score (0–100) =
    (cv_score       × 0.50) +   ← computer vision confidence (dominant)
    (nlp_score      × 0.20) +   ← NLP urgency score
    (location_score × 0.20) +   ← historical hotspot risk of the area
    (time_score     × 0.10)     ← time-of-day risk factor

Why these weights?
  - CV is the most objective signal (camera doesn't lie) → 50%
  - NLP gives intent/urgency context the camera can't see → 20%
  - Location encodes long-term historical risk patterns → 20%
  - Time is the weakest signal (incidents happen any time) → 10%

The weights are configurable. Changing them and rerunning is a great
experiment to discuss in interviews — shows you understand the system
as a whole, not just individual components.

Design principles:
  1. Every score is explainable — "Why CRITICAL?" has a clear answer
  2. Missing signals degrade gracefully — any signal can be None
  3. Weights sum to 1.0 always (normalized even when signals missing)
  4. Outputs are validated — score always in [0, 100], level always valid
"""

import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from loguru import logger

# ── Weight configuration ──────────────────────────────────────────────────────
# Change these to experiment. Must sum to 1.0.
DEFAULT_WEIGHTS = {
    "cv_score":       0.50,
    "nlp_score":      0.20,
    "location_score": 0.20,
    "time_score":     0.10,
}

# Risk level thresholds
THRESHOLDS = {
    "CRITICAL": 75,
    "HIGH":     55,
    "MEDIUM":   35,
    "LOW":      0,
}

# Event-type severity boost — some types are inherently more serious
# regardless of model scores. Applied as additive adjustment.
EVENT_SEVERITY_BOOST = {
    "FIRE":     5.0,    # Fire spreads fast → always boost
    "ACCIDENT": 3.0,
    "MEDICAL":  3.0,
    "FLOOD":    2.0,
    "CRIME":    2.0,
    "CROWD":    1.0,
    "NORMAL":  -10.0,   # NORMAL events are penalized
}


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class RiskScore:
    """
    Full risk assessment output.
    Stored in RiskPrediction.explanation in the DB.
    Returned by every /risk/score API call.
    """
    # Final outputs
    risk_score:   float            # 0–100
    risk_level:   str              # LOW / MEDIUM / HIGH / CRITICAL

    # Input signals (all 0–1)
    cv_score:       Optional[float] = None
    nlp_score:      Optional[float] = None
    location_score: Optional[float] = None
    time_score:     Optional[float] = None

    # Context
    event_type:    str  = "UNKNOWN"
    area_name:     str  = "Unknown"
    timestamp:     str  = ""

    # Weights used (may differ from default if signals missing)
    weights_used:  dict = field(default_factory=dict)

    # Explanation — the "why" that makes this project stand out
    explanation:   dict = field(default_factory=dict)

    # Model tracing
    model_version: str = "v1.0-phase3"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_db_dict(self) -> dict:
        """Subset stored in DB RiskPrediction row."""
        return {
            "risk_score":     self.risk_score,
            "risk_level":     self.risk_level,
            "cv_score":       self.cv_score,
            "nlp_score":      self.nlp_score,
            "location_score": self.location_score,
            "time_score":     self.time_score,
            "model_version":  self.model_version,
            "explanation":    self.explanation,
        }


# ── Hotspot cache ─────────────────────────────────────────────────────────────
# Location scores come from the DB (LocationHotspot table).
# We cache them in memory to avoid DB queries on every request.
_hotspot_cache: dict[str, float] = {}   # area_name → risk_score
_cache_loaded   = False

def _get_location_score(area_name: str, fallback: float = 0.5) -> float:
    """
    Looks up historical risk score for an area.
    Returns fallback if area not in cache.
    """
    if area_name in _hotspot_cache:
        return _hotspot_cache[area_name]
    logger.debug(f"Area '{area_name}' not in hotspot cache, using fallback={fallback}")
    return fallback


def load_hotspot_cache(hotspots: list[dict]):
    """
    Populates the in-memory hotspot cache from DB records.
    Call this at startup (from main.py lifespan).

    Args:
        hotspots: list of dicts with 'area_name' and 'risk_score' keys
    """
    global _hotspot_cache, _cache_loaded
    _hotspot_cache = {h["area_name"]: float(h["risk_score"]) for h in hotspots}
    _cache_loaded  = True
    logger.info(f"Hotspot cache loaded: {len(_hotspot_cache)} areas")


def update_hotspot(area_name: str, risk_score: float):
    """Updates a single area in the cache (called after nightly recompute)."""
    _hotspot_cache[area_name] = risk_score


# ── Core scoring function ─────────────────────────────────────────────────────
def compute_risk_score(
    event_type:     str,
    area_name:      str,
    timestamp:      datetime,
    cv_score:       Optional[float] = None,
    nlp_score:      Optional[float] = None,
    location_score: Optional[float] = None,
    time_score:     Optional[float] = None,
    weights:        dict = None,
) -> RiskScore:
    """
    Computes the unified risk score from all available signals.

    Any signal can be None — the engine re-normalizes weights automatically.
    This means the system works even if only one signal is available.

    Args:
        event_type:     emergency category ("FIRE", "ACCIDENT", etc.)
        area_name:      location name for hotspot lookup
        timestamp:      when the event occurred
        cv_score:       0–1 from YOLOv8 (or None if no image)
        nlp_score:      0–1 from DistilBERT (or None if no text)
        location_score: 0–1 override (if None, looked up from hotspot cache)
        time_score:     0–1 override (if None, computed from timestamp)
        weights:        override DEFAULT_WEIGHTS (for experiments)

    Returns:
        RiskScore with full explanation
    """
    w = weights or DEFAULT_WEIGHTS

    # ── Step 1: Fill missing signals ──────────────────────────────────────────

    # Location: look up from hotspot cache if not provided
    if location_score is None:
        location_score = _get_location_score(area_name)

    # Time: compute from timestamp if not provided
    if time_score is None:
        try:
            from app.timeseries.analyzer import get_time_score
            ts_result  = get_time_score(timestamp, event_type)
            time_score = ts_result["time_score"]
            is_peak    = ts_result["peak_hour"]
        except Exception as e:
            logger.warning(f"Time score computation failed: {e}")
            # Fallback heuristic
            hour       = timestamp.hour
            time_score = 0.80 if (8<=hour<=10 or 17<=hour<=20) else (0.70 if hour>=22 or hour<=4 else 0.45)
            is_peak    = (8 <= hour <= 10) or (17 <= hour <= 20)
    else:
        hour    = timestamp.hour
        is_peak = (8 <= hour <= 10) or (17 <= hour <= 20)

    # ── Step 2: Re-normalize weights for missing signals ──────────────────────
    available = {
        "cv_score":       cv_score,
        "nlp_score":      nlp_score,
        "location_score": location_score,
        "time_score":     time_score,
    }

    # Build effective weights — zero out missing signals, re-normalize rest
    eff_weights = {}
    total_w     = 0.0
    for key, val in available.items():
        if val is not None:
            eff_weights[key] = w.get(key, 0.0)
            total_w         += w.get(key, 0.0)
        else:
            eff_weights[key] = 0.0

    if total_w == 0:
        logger.error("All signals are None — cannot compute risk score")
        return _zero_score(event_type, area_name, timestamp)

    # Normalize so weights sum to 1
    eff_weights = {k: v / total_w for k, v in eff_weights.items()}

    # ── Step 3: Weighted sum ──────────────────────────────────────────────────
    raw_score = (
        (cv_score       or 0.0) * eff_weights["cv_score"]       +
        (nlp_score      or 0.0) * eff_weights["nlp_score"]      +
        (location_score or 0.0) * eff_weights["location_score"] +
        (time_score     or 0.0) * eff_weights["time_score"]
    )

    # Scale to 0–100
    score_100 = raw_score * 100

    # ── Step 4: Apply event-type severity boost ───────────────────────────────
    boost        = EVENT_SEVERITY_BOOST.get(event_type, 0.0)
    score_100    = score_100 + boost
    score_100    = round(min(max(score_100, 0.0), 100.0), 2)

    # ── Step 5: Classify ─────────────────────────────────────────────────────
    if score_100 >= THRESHOLDS["CRITICAL"]:
        level = "CRITICAL"
    elif score_100 >= THRESHOLDS["HIGH"]:
        level = "HIGH"
    elif score_100 >= THRESHOLDS["MEDIUM"]:
        level = "MEDIUM"
    else:
        level = "LOW"

    # ── Step 6: Build explanation ─────────────────────────────────────────────
    # Find dominant signal (highest weighted contribution)
    contributions = {
        "cv":       (cv_score       or 0.0) * eff_weights["cv_score"],
        "nlp":      (nlp_score      or 0.0) * eff_weights["nlp_score"],
        "location": (location_score or 0.0) * eff_weights["location_score"],
        "time":     (time_score     or 0.0) * eff_weights["time_score"],
    }
    dominant = max(contributions, key=contributions.get)

    # Generate human-readable reason
    reasons = []
    if cv_score and cv_score > 0.7:
        reasons.append(f"strong visual detection ({cv_score:.0%} confidence)")
    if nlp_score and nlp_score > 0.7:
        reasons.append(f"high-urgency text ({nlp_score:.0%})")
    if location_score and location_score >= 0.7:
        reasons.append(f"high-risk area ({area_name})")
    if is_peak:
        reasons.append("peak incident hour")
    if not reasons:
        reasons.append(f"moderate {dominant} signal")

    explanation = {
        "dominant_signal":    dominant,
        "reasons":            reasons,
        "contributions": {
            "cv":       round(contributions["cv"], 4),
            "nlp":      round(contributions["nlp"], 4),
            "location": round(contributions["location"], 4),
            "time":     round(contributions["time"], 4),
        },
        "raw_signals": {
            "cv_score":       round(cv_score,       4) if cv_score       is not None else None,
            "nlp_score":      round(nlp_score,      4) if nlp_score      is not None else None,
            "location_score": round(location_score, 4) if location_score is not None else None,
            "time_score":     round(time_score,     4) if time_score     is not None else None,
        },
        "severity_boost":  boost,
        "is_peak_hour":    is_peak,
        "high_risk_area":  (location_score or 0) >= 0.7,
        "formula": {
            "weights_used":  {k: round(v, 4) for k, v in eff_weights.items()},
            "raw_score":     round(raw_score, 4),
            "boost_applied": boost,
            "final_score":   score_100,
        },
    }

    return RiskScore(
        risk_score=score_100,
        risk_level=level,
        cv_score=round(cv_score, 4) if cv_score is not None else None,
        nlp_score=round(nlp_score, 4) if nlp_score is not None else None,
        location_score=round(location_score, 4),
        time_score=round(time_score, 4),
        event_type=event_type,
        area_name=area_name,
        timestamp=timestamp.isoformat(),
        weights_used=eff_weights,
        explanation=explanation,
    )


# ── Convenience wrappers ──────────────────────────────────────────────────────
def score_from_text_only(text: str, area_name: str, timestamp: datetime = None) -> RiskScore:
    """
    Full pipeline: NLP → location → time → risk score.
    No CV signal (text-only input like a tweet or report).
    Used by /predict/text endpoint.
    """
    from app.nlp.inference import predict as nlp_predict

    ts         = timestamp or datetime.utcnow()
    nlp_result = nlp_predict(text)

    return compute_risk_score(
        event_type=nlp_result.label,
        area_name=area_name,
        timestamp=ts,
        cv_score=None,                        # no image
        nlp_score=nlp_result.urgency_score,
    )


def score_from_cv_only(cv_result, area_name: str, timestamp: datetime = None) -> RiskScore:
    """
    CV → location → time → risk score.
    No NLP signal (image/video only, no accompanying text).
    Used by /predict/image and /predict/video endpoints.
    """
    ts = timestamp or datetime.utcnow()

    return compute_risk_score(
        event_type=cv_result.label,
        area_name=area_name,
        timestamp=ts,
        cv_score=cv_result.cv_score,
        nlp_score=None,                       # no text
    )


def score_from_all_signals(
    cv_result,
    nlp_text: str,
    area_name: str,
    timestamp: datetime = None,
) -> RiskScore:
    """
    Full pipeline: CV + NLP + location + time → risk score.
    Most accurate. Used when both image and text are available.
    """
    from app.nlp.inference import predict as nlp_predict

    ts         = timestamp or datetime.utcnow()
    nlp_result = nlp_predict(nlp_text)

    # Use CV label if both agree, else use higher-confidence one
    if cv_result.label == nlp_result.label:
        event_type = cv_result.label
    elif cv_result.cv_score >= nlp_result.urgency_score:
        event_type = cv_result.label
    else:
        event_type = nlp_result.label

    return compute_risk_score(
        event_type=event_type,
        area_name=area_name,
        timestamp=ts,
        cv_score=cv_result.cv_score,
        nlp_score=nlp_result.urgency_score,
    )


def _zero_score(event_type, area_name, timestamp) -> RiskScore:
    """Returns a safe zero-score when all signals are missing."""
    return RiskScore(
        risk_score=0.0, risk_level="LOW",
        event_type=event_type, area_name=area_name,
        timestamp=timestamp.isoformat(),
        explanation={"error": "no signals available"},
    )


# ── Heatmap aggregation ───────────────────────────────────────────────────────
def compute_area_risk_summary(predictions: list[dict]) -> list[dict]:
    """
    Aggregates risk predictions by area for heatmap rendering.

    Args:
        predictions: list of dicts with area_name, risk_score, latitude, longitude

    Returns:
        list of area summaries sorted by risk (highest first)
    """
    from collections import defaultdict

    area_data = defaultdict(lambda: {
        "scores": [], "count": 0, "lat": 0.0, "lon": 0.0, "types": []
    })

    for pred in predictions:
        area  = pred.get("area_name", "Unknown")
        score = pred.get("risk_score", 0.0)
        area_data[area]["scores"].append(score)
        area_data[area]["count"]  += 1
        area_data[area]["lat"]     = pred.get("latitude",  0.0)
        area_data[area]["lon"]     = pred.get("longitude", 0.0)
        area_data[area]["types"].append(pred.get("event_type", "UNKNOWN"))

    summaries = []
    for area, data in area_data.items():
        scores    = data["scores"]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0

        # Heatmap weight = blend of average and max (don't ignore single big events)
        heat_weight = round((avg_score * 0.6) + (max_score * 0.4), 2)

        # Most common event type in this area
        from collections import Counter
        top_type = Counter(data["types"]).most_common(1)[0][0]

        summaries.append({
            "area_name":   area,
            "latitude":    data["lat"],
            "longitude":   data["lon"],
            "avg_score":   round(avg_score, 2),
            "max_score":   round(max_score, 2),
            "heat_weight": heat_weight,
            "incident_count": data["count"],
            "top_event_type": top_type,
        })

    return sorted(summaries, key=lambda x: x["heat_weight"], reverse=True)