"""
alerts_router.py — schema-aligned version
Uses: Incident, Prediction, RiskLevel (from real events.py)
"""

import asyncio
import json
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session, joinedload
from loguru import logger

from app.database import get_db
from app.models.events import Incident, Prediction, Alert, RiskLevel
from app.alerts.websocket_manager import ws_manager
from app.alerts.alert_engine import process_event, get_alert_stats

router = APIRouter(tags=["Alerts"])


@router.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """Real-time WebSocket feed — pushes every new event to connected dashboards."""
    await ws_manager.connect(ws)
    await ws_manager.send_to(ws, {
        "type":        "connected",
        "message":     "Connected to SmartCityAI live feed",
        "connections": ws_manager.connection_count,
        "timestamp":   datetime.utcnow().isoformat(),
    })
    try:
        while True:
            try:
                msg  = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                data = json.loads(msg)
                if data.get("type") == "ping":
                    await ws_manager.send_to(ws, {"type": "pong"})
            except asyncio.TimeoutError:
                await ws_manager.send_to(ws, {
                    "type":      "ping",
                    "timestamp": datetime.utcnow().isoformat(),
                })
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(ws)


@router.get("/alerts")
def get_alerts(
    limit:      int = Query(default=50, ge=1, le=500),
    hours_back: int = Query(default=24, ge=1, le=168),
    risk_level: str = Query(default=None),
    event_type: str = Query(default=None),
    db: Session = Depends(get_db),
):
    from app.models.events import IncidentType

    since = datetime.utcnow() - timedelta(hours=hours_back)
    q     = (
        db.query(Incident)
        .options(joinedload(Incident.predictions))
        .filter(Incident.occurred_at >= since)
        .order_by(Incident.occurred_at.desc())
    )

    if event_type:
        try:
            q = q.filter(Incident.incident_type == IncidentType(event_type.upper()))
        except ValueError:
            pass

    if risk_level:
        try:
            q = q.join(Prediction).filter(
                Prediction.risk_level == RiskLevel(risk_level.upper())
            )
        except ValueError:
            pass

    rows = q.limit(limit).all()

    return {
        "count":      len(rows),
        "hours_back": hours_back,
        "alerts": [
            {
                "id":          r.id,
                "event_type":  r.incident_type.value,
                "area_name":   r.area_name,
                "latitude":    r.latitude,
                "longitude":   r.longitude,
                "occurred_at": r.occurred_at.isoformat() if r.occurred_at else None,
                "risk_score":  r.predictions[-1].risk_score        if r.predictions else None,
                "risk_level":  r.predictions[-1].risk_level.value  if r.predictions else None,
                "alert_sent":  bool(r.alerts),
                "raw_input":   r.description,
                "explanation": {"reasons": [r.description]} if r.description else {},
            }
            for r in rows
        ],
    }


@router.get("/alerts/stats")
def alert_stats():
    return get_alert_stats()


@router.post("/alerts/test")
async def send_test_alert(db: Session = Depends(get_db)):
    test_event = {
        "id":          "test-00000000",
        "event_type":  "FIRE",
        "area_name":   "Kaiserbagh",
        "latitude":    26.8530,
        "longitude":   80.9350,
        "risk_score":  92.5,
        "risk_level":  "CRITICAL",
        "occurred_at": datetime.utcnow().isoformat(),
        "raw_input":   "TEST ALERT — Fire near Kaiserbagh market",
        "explanation": {
            "dominant_signal": "cv",
            "reasons": ["strong visual detection (94%)", "high-risk area"],
        },
    }
    actions = await process_event(test_event, db=db)
    return {"message": "Test alert sent", "actions": actions,
            "ws_connections": ws_manager.connection_count}


@router.get("/alerts/ws-status")
def ws_status():
    return {"active_connections": ws_manager.connection_count,
            "timestamp": datetime.utcnow().isoformat()}