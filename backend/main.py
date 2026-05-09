"""
main.py — SmartCityAI (Phase 6, schema-aligned)
All imports match the real events.py schema:
  Incident, Prediction, Alert, Hotspot, IncidentType, RiskLevel, etc.
"""

import asyncio
import redis as redis_lib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.database import engine, Base, get_db, check_db_connection, init_db
from app.models.events import (
    Incident, Prediction, Alert, Hotspot,
    IncidentType, RiskLevel, AlertStatus,
)
from app.simulator.data_generator import (
    generate_batch, generate_historical_data, generate_one_event,
    publish_to_redis, print_event_table, seed_hotspots,
    get_redis_client, LUCKNOW_AREAS,
)
from app.routers.nlp_router    import router as nlp_router
from app.routers.cv_router     import router as cv_router
from app.routers.risk_router   import router as risk_router
from app.routers.alerts_router import router as alerts_router
from app.routers.mlops_router  import router as mlops_router
from app.nlp.inference         import load_model as load_nlp_model
from app.alerts.alert_engine   import redis_stream_consumer

settings   = get_settings()
_scheduler = None


# ── Startup / shutdown 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    if not check_db_connection():
        raise RuntimeError("DB not reachable. Is Docker running?")

    # Creates PostGIS extension + all tables
    init_db()
    logger.info(" DB ready")

    # Seed hotspot data
    db = next(get_db())
    try:
        seed_hotspots(db)
        _load_hotspot_cache(db)
    finally:
        db.close()

    # Load NLP model (falls back to rule-based if not trained yet)
    load_nlp_model()

    # Redis stream consumer background task
    consumer_task = asyncio.create_task(
        redis_stream_consumer(settings.redis_url),
        name="redis-consumer",
    )

    # MLOps retraining scheduler
    try:
        from app.mlops.retrain import start_retraining_scheduler
        retrain_hours = int(getattr(settings, "retrain_interval_hours", 24))
        _scheduler    = start_retraining_scheduler(interval_hours=retrain_hours)
    except Exception as e:
        logger.warning(f"Retraining scheduler not started: {e}")

    logger.success(" SmartCityAI — all systems running")
    yield

    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    if _scheduler:
        _scheduler.shutdown(wait=False)
    logger.info("Shutdown complete")


# ── App 

app = FastAPI(
    title="SmartCityAI API",
    description="Emergency detection and risk scoring for Lucknow",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://project-pvmz7.vercel.app",
        "https://smartcity-570iscb4u-rohit-pal-s-projects1.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(nlp_router)
app.include_router(cv_router)
app.include_router(risk_router)
app.include_router(alerts_router)
app.include_router(mlops_router)


# ── Hotspot cache 

def _load_hotspot_cache(db: Session):
    """Load hotspot risk scores into Risk Engine memory."""
    try:
        from app.risk_engine.scorer import load_hotspot_cache
        hotspots = db.query(Hotspot).all()
        load_hotspot_cache([
            {"area_name": h.area_name, "risk_score": h.risk_weight}
            for h in hotspots
        ])
    except Exception as e:
        logger.warning(f"Hotspot cache load failed: {e}")


# ── Core routes 

@app.get("/")
def root():
    return {
        "app":     settings.app_name,
        "version": settings.app_version,
        "docs":    "/docs",
        "ws":      "ws://localhost:8000/ws/live",
        "mlflow":  "http://localhost:5000",
    }


@app.get("/health")
def health(db: Session = Depends(get_db)):
    from app.nlp.inference          import get_model_info
    from app.cv.detector            import get_cv_model_info
    from app.timeseries.analyzer    import get_models_status
    from app.alerts.alert_engine    import get_alert_stats
    from app.alerts.websocket_manager import ws_manager

    db_ok, redis_ok = check_db_connection(), False
    try:
        r = redis_lib.from_url(settings.redis_url, decode_responses=True)
        r.ping()
        redis_ok = True
    except Exception:
        pass

    return {
        "status":     "healthy" if (db_ok and redis_ok) else "degraded",
        "database":   "ok" if db_ok    else "down",
        "redis":      "ok" if redis_ok else "down",
        "nlp_model":  get_model_info(),
        "cv_model":   get_cv_model_info(),
        "timeseries": get_models_status(),
        "alerts":     get_alert_stats(),
        "websocket":  {"active_connections": ws_manager.connection_count},
        "scheduler":  {"running": _scheduler.running if _scheduler else False},
        "timestamp":  datetime.utcnow().isoformat(),
    }


@app.post("/simulate/batch")
async def simulate_batch(
    n: int = 20,
    db: Session = Depends(get_db),
):
    from app.alerts.alert_engine import process_event

    redis_client = get_redis_client()
    events       = generate_batch(n=n, db=db, redis_client=redis_client)

    # Process each event through the alert pipeline (WebSocket + Telegram)
    for e in events:
        await process_event({
            # Translate to the format process_event expects
            "id":          str(e["id"]),
            "event_type":  e["type"],
            "area_name":   e["area"],
            "latitude":    e["lat"],
            "longitude":   e["lon"],
            "risk_score":  e["score"],
            "risk_level":  e["risk_level"],
            "occurred_at": datetime.utcnow().isoformat(),
            "raw_input":   e["desc"],
            "explanation": {"dominant_signal": "cv", "reasons": [e["desc"]]},
        }, db=db)

    print_event_table(events)
    return {
        "generated": len(events),
        "summary": {
            lvl: sum(1 for e in events if e["risk_level"] == lvl)
            for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        },
        "events": events,
    }


@app.post("/simulate/seed-historical")
def seed_historical(
    days: int = 30,
    events_per_day: int = 50,
    db: Session = Depends(get_db),
):
    existing = db.query(Incident).count()
    if existing > 100:
        return {"message": f"DB already has {existing} events. Skipping."}

    total = generate_historical_data(db, days=days, events_per_day=events_per_day)
    return {"seeded": total, "days": days, "events_per_day": events_per_day}


@app.get("/events")
def get_events(
    limit:      int = 50,
    risk_level: str = None,
    event_type: str = None,
    db: Session = Depends(get_db),
):
    q = (
        db.query(Incident)
        .options(joinedload(Incident.predictions))
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
        "count": len(rows),
        "events": [
            {
                "id":          r.id,
                "event_type":  r.incident_type.value,
                "area_name":   r.area_name,
                "latitude":    r.latitude,
                "longitude":   r.longitude,
                "occurred_at": r.occurred_at.isoformat() if r.occurred_at else None,
                "risk_score":  r.predictions[-1].risk_score  if r.predictions else None,
                "risk_level":  r.predictions[-1].risk_level.value if r.predictions else None,
                "raw_input":   r.description,
                "is_active":   r.is_active,
            }
            for r in rows
        ],
    }


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total    = db.query(Incident).count()
    active   = db.query(Incident).filter(Incident.is_active == True).count()
    critical = db.query(Prediction).filter(Prediction.risk_level == RiskLevel.CRITICAL).count()
    high     = db.query(Prediction).filter(Prediction.risk_level == RiskLevel.HIGH).count()

    return {
        "total_events":  total,
        "active_events": active,
        "critical":      critical,
        "high":          high,
        "timestamp":     datetime.utcnow().isoformat(),
    }


@app.get("/alerts")
def get_alerts_route(
    limit:      int = 50,
    hours_back: int = 24,
    db: Session = Depends(get_db),
):
    since = datetime.utcnow() - timedelta(hours=hours_back)
    rows  = (
        db.query(Alert)
        .options(joinedload(Alert.incident))
        .filter(Alert.created_at >= since)
        .order_by(Alert.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "count": len(rows),
        "alerts": [
            {
                "id":         r.id,
                "channel":    r.channel,
                "status":     r.status.value,
                "message":    r.message,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "incident_id": r.incident_id,
            }
            for r in rows
        ],
    }