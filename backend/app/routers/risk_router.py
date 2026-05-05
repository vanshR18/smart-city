"""
risk_router.py — schema-aligned version
Uses: Incident, Prediction, RiskLevel (from real events.py)
"""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional
from loguru import logger

from app.database import get_db
from app.risk_engine.scorer import (
    compute_risk_score, score_from_text_only,
    compute_area_risk_summary, RiskScore,
)
from app.timeseries.analyzer import (
    get_time_score, get_hourly_risk_profile, get_models_status,
)

router = APIRouter(prefix="/risk", tags=["Risk Engine"])


class ScoreRequest(BaseModel):
    event_type:     str   = "ACCIDENT"
    area_name:      str   = "Hazratganj"
    cv_score:       Optional[float] = None
    nlp_score:      Optional[float] = None
    location_score: Optional[float] = None
    time_score:     Optional[float] = None
    timestamp:      Optional[str]   = None


class TextScoreRequest(BaseModel):
    text:      str = "Bada accident hua hai Hazratganj pe"
    area_name: str = "Unknown"


@router.post("/score")
def risk_score(body: ScoreRequest):
    ts = datetime.fromisoformat(body.timestamp) if body.timestamp else datetime.utcnow()
    result = compute_risk_score(
        event_type=body.event_type, area_name=body.area_name, timestamp=ts,
        cv_score=body.cv_score, nlp_score=body.nlp_score,
        location_score=body.location_score, time_score=body.time_score,
    )
    return result.to_dict()


@router.post("/score/text")
def risk_score_from_text(body: TextScoreRequest):
    return score_from_text_only(body.text, body.area_name).to_dict()


@router.get("/heatmap")
def get_heatmap(
    hours_back: int = Query(default=24, ge=1, le=168),
    db: Session = Depends(get_db),
):
    from app.models.events import Incident, Prediction

    since = datetime.utcnow() - timedelta(hours=hours_back)
    rows  = (
        db.query(Incident)
        .join(Prediction)
        .filter(Incident.occurred_at >= since)
        .all()
    )

    predictions = [
        {
            "area_name":  r.area_name or "Unknown",
            "latitude":   r.latitude,
            "longitude":  r.longitude,
            "risk_score": r.predictions[-1].risk_score if r.predictions else 50.0,
            "event_type": r.incident_type.value,
        }
        for r in rows
        if r.latitude and r.longitude
    ]

    summaries = compute_area_risk_summary(predictions)
    return {
        "hours_back":   hours_back,
        "total_events": len(predictions),
        "areas":        summaries,
        "leaflet_heat": [
            [s["latitude"], s["longitude"], s["heat_weight"] / 100]
            for s in summaries if s["latitude"] and s["longitude"]
        ],
    }


@router.get("/time-profile")
def time_profile(event_type: str = Query(default="overall")):
    return get_hourly_risk_profile(event_type)


@router.get("/models-status")
def models_status():
    return {"timeseries": get_models_status(), "timestamp": datetime.utcnow().isoformat()}


@router.get("/explain/{risk_level}")
def explain_risk_level(risk_level: str):
    explanations = {
        "LOW":      {"label":"LOW",      "color":"#22c55e", "threshold":"0–24",  "description":"Routine. No action needed."},
        "MEDIUM":   {"label":"MEDIUM",   "color":"#f59e0b", "threshold":"25–49", "description":"Monitor. Prepare response."},
        "HIGH":     {"label":"HIGH",     "color":"#ef4444", "threshold":"50–74", "description":"Active emergency. Dispatch units."},
        "CRITICAL": {"label":"CRITICAL", "color":"#7f1d1d", "threshold":"75–100","description":"Major emergency. All units respond."},
    }
    level = risk_level.upper()
    if level not in explanations:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Unknown risk level: {risk_level}")
    return explanations[level]