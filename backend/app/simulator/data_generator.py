"""
Smart City Data Simulator — Lucknow Edition
============================================
Generates realistic incident events across Lucknow's major areas.
Pushes events to:
  1. PostgreSQL (permanent storage)
  2. Redis Stream (real-time pipeline for ML models)
"""

import json
import random
import time
from datetime import datetime, timedelta

import numpy as np
from faker import Faker
from loguru import logger
from rich.console import Console
from rich.table import Table

from app.config import get_settings
from app.database import SessionLocal
from app.models.events import (
    Alert, AlertStatus, Hotspot, Incident,
    IncidentSource, IncidentType, Prediction, RiskLevel,
)

fake     = Faker("en_IN")
console  = Console()
settings = get_settings()


# ── Lucknow geography ─────────────────────────────────────────────────────────

LUCKNOW_AREAS = [
    {"name": "Hazratganj",      "lat": 26.8467, "lon": 80.9462, "risk": 0.7},
    {"name": "Charbagh",        "lat": 26.8331, "lon": 80.9239, "risk": 0.8},
    {"name": "Aminabad",        "lat": 26.8445, "lon": 80.9319, "risk": 0.6},
    {"name": "Gomti Nagar",     "lat": 26.8587, "lon": 81.0036, "risk": 0.4},
    {"name": "Aliganj",         "lat": 26.8862, "lon": 80.9652, "risk": 0.3},
    {"name": "Indira Nagar",    "lat": 26.8824, "lon": 81.0024, "risk": 0.3},
    {"name": "Alambagh",        "lat": 26.8186, "lon": 80.9110, "risk": 0.5},
    {"name": "Chinhat",         "lat": 26.8700, "lon": 81.0600, "risk": 0.4},
    {"name": "Vikas Nagar",     "lat": 26.9150, "lon": 80.9900, "risk": 0.3},
    {"name": "Transport Nagar", "lat": 26.8600, "lon": 80.8900, "risk": 0.6},
    {"name": "Hussainganj",     "lat": 26.8450, "lon": 80.9400, "risk": 0.5},
    {"name": "Mahanagar",       "lat": 26.8720, "lon": 80.9800, "risk": 0.4},
]

HOURLY_WEIGHTS = {
    IncidentType.ACCIDENT: [
        0.3, 0.2, 0.1, 0.1, 0.1, 0.2, 0.5, 0.9,
        1.0, 0.8, 0.6, 0.5, 0.6, 0.6, 0.7, 0.8,
        1.0, 0.9, 0.7, 0.5, 0.4, 0.4, 0.3, 0.3,
    ],
    IncidentType.FIRE: [
        0.3, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6,
        0.6, 0.5, 0.5, 0.5, 0.6, 0.5, 0.5, 0.5,
        0.6, 0.6, 0.5, 0.4, 0.4, 0.4, 0.3, 0.3,
    ],
    IncidentType.CRIME: [
        0.7, 0.8, 0.9, 0.9, 0.7, 0.4, 0.3, 0.2,
        0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.8,
    ],
    IncidentType.CROWD: [
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3,
        0.5, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7,
        0.8, 0.9, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1,
    ],
    IncidentType.MEDICAL: [
        0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4,
    ],
    IncidentType.FLOOD: [
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.3,
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3,
        0.4, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5,
    ],
}

DESCRIPTIONS = {
    IncidentType.ACCIDENT: [
        "Two-wheeler collision near traffic signal",
        "Truck overturned blocking main road",
        "Chain collision involving multiple vehicles",
        "Hit and run case, pedestrian injured",
        "Vehicle skidded on wet road, minor injuries",
        "Auto-rickshaw collided with car at intersection",
    ],
    IncidentType.FIRE: [
        "Fire reported in commercial building",
        "Short circuit caused fire in residential area",
        "Kitchen fire at dhaba spreading to adjacent shops",
        "Garbage dump fire causing heavy smoke",
        "Electrical fire in market area",
        "Fire in godown near railway line",
    ],
    IncidentType.FLOOD: [
        "Heavy waterlogging blocking underpass",
        "Low-lying areas inundated after heavy rain",
        "Drain overflow causing flooding on main road",
        "River water level rising near residential colony",
        "Road collapsed due to waterlogging",
    ],
    IncidentType.CRIME: [
        "Chain snatching reported near market",
        "Robbery at ATM kiosk",
        "Suspicious vehicle parked near government building",
        "Group conflict in crowded area",
        "Theft reported from parked vehicle",
    ],
    IncidentType.CROWD: [
        "Unmanaged crowd at railway station",
        "Stampede risk at religious gathering",
        "Overcrowding at bus terminal, situation tense",
        "Large gathering blocking road near stadium",
    ],
    IncidentType.MEDICAL: [
        "Person collapsed on road, ambulance needed",
        "Road accident victim requires immediate medical help",
        "Medical emergency at public place",
        "Heat stroke victim reported near market",
    ],
    IncidentType.NORMAL: [
        "Routine patrol completed, no incidents",
        "Traffic moving smoothly",
        "Area clear, no issues reported",
    ],
}


# ── Risk Scoring Engine ───────────────────────────────────────────────────────

class RiskScoringEngine:
    WEIGHTS = {"cv": 0.50, "nlp": 0.20, "location": 0.20, "time": 0.10}

    CV_BASE = {
        IncidentType.ACCIDENT: 0.85,
        IncidentType.FIRE:     0.90,
        IncidentType.FLOOD:    0.60,
        IncidentType.CRIME:    0.65,
        IncidentType.CROWD:    0.55,
        IncidentType.MEDICAL:  0.70,
        IncidentType.NORMAL:   0.05,
    }

    def compute(self, incident_type, area, hour, description=""):
        cv_base  = self.CV_BASE.get(incident_type, 0.5)
        cv_score = float(np.clip(cv_base + np.random.normal(0, 0.05), 0, 1))

        urgency_words = ["stampede", "fire", "explosion", "collapse", "dead", "critical", "severe"]
        nlp_base  = 0.6 if incident_type != IncidentType.NORMAL else 0.1
        bonus     = 0.15 if any(w in description.lower() for w in urgency_words) else 0.0
        nlp_score = float(np.clip(nlp_base + bonus + np.random.normal(0, 0.05), 0, 1))

        location_score = float(area["risk"])

        hour_weights = HOURLY_WEIGHTS.get(incident_type, [0.5] * 24)
        time_score   = float(hour_weights[hour])

        raw        = (cv_score * self.WEIGHTS["cv"] + nlp_score * self.WEIGHTS["nlp"] +
                      location_score * self.WEIGHTS["location"] + time_score * self.WEIGHTS["time"])
        risk_score = round(raw * 100, 2)

        if risk_score >= 75:   risk_level = RiskLevel.CRITICAL
        elif risk_score >= 50: risk_level = RiskLevel.HIGH
        elif risk_score >= 25: risk_level = RiskLevel.MEDIUM
        else:                  risk_level = RiskLevel.LOW

        explanation = (
            f"CV={cv_score:.2f}×0.5 + NLP={nlp_score:.2f}×0.2 + "
            f"Location={location_score:.2f}×0.2 + Time={time_score:.2f}×0.1 "
            f"→ Score={risk_score}"
        )
        return {
            "cv_score": cv_score, "nlp_score": nlp_score,
            "location_score": location_score, "time_score": time_score,
            "risk_score": risk_score, "risk_level": risk_level,
            "explanation": explanation,
        }


_engine = RiskScoringEngine()


# ── Redis ─────────────────────────────────────────────────────────────────────

def get_redis_client():
    try:
        import redis as redis_lib
        client = redis_lib.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        return None


def publish_to_redis(client, incident: Incident, prediction: Prediction):
    if client is None:
        return
    try:
        payload = {
            "incident_id": str(incident.id),
            "type":        incident.incident_type.value,
            "area":        incident.area_name or "",
            "lat":         str(incident.latitude),
            "lon":         str(incident.longitude),
            "risk_score":  str(prediction.risk_score),
            "risk_level":  prediction.risk_level.value,
            "description": incident.description or "",
            "occurred_at": incident.occurred_at.isoformat() if incident.occurred_at else "",
        }
        client.xadd("smartcity:events", payload, maxlen=1000)
    except Exception as e:
        logger.error(f"Redis publish failed: {e}")


# ── Hotspot seeder ────────────────────────────────────────────────────────────

def seed_hotspots(db):
    if db.query(Hotspot).count() > 0:
        logger.info("Hotspots already seeded. Skipping.")
        return
    db.add_all([
        Hotspot(
            area_name=a["name"], latitude=a["lat"],
            longitude=a["lon"], risk_weight=a["risk"], radius_meters=600.0,
        )
        for a in LUCKNOW_AREAS
    ])
    db.commit()
    logger.success(f"Seeded {len(LUCKNOW_AREAS)} hotspots.")


# ── Single event ──────────────────────────────────────────────────────────────

def generate_one_event(db, redis_client=None, occurred_at=None) -> dict:
    area          = random.choice(LUCKNOW_AREAS)
    lat           = area["lat"] + np.random.normal(0, 0.002)
    lon           = area["lon"] + np.random.normal(0, 0.002)

    all_types     = list(IncidentType)
    weights       = [0.20, 0.18, 0.12, 0.12, 0.10, 0.10, 0.18]
    incident_type = random.choices(all_types, weights=weights, k=1)[0]

    description   = random.choice(DESCRIPTIONS.get(incident_type, ["Unknown event"]))
    occurred_at   = occurred_at or (datetime.utcnow() - timedelta(minutes=random.randint(0, 30)))
    hour          = occurred_at.hour

    scores = _engine.compute(incident_type, area, hour, description)

    incident = Incident(
        incident_type = incident_type,
        source        = IncidentSource.SIMULATED,
        description   = description,
        latitude      = round(lat, 6),
        longitude     = round(lon, 6),
        location      = f"SRID=4326;POINT({lon} {lat})",
        area_name     = area["name"],
        occurred_at   = occurred_at,
        is_active     = True,
    )
    db.add(incident)
    db.flush()

    prediction = Prediction(
        incident_id    = incident.id,
        risk_score     = scores["risk_score"],
        risk_level     = scores["risk_level"],
        cv_score       = scores["cv_score"],
        nlp_score      = scores["nlp_score"],
        location_score = scores["location_score"],
        time_score     = scores["time_score"],
        model_name     = "risk-engine-v1",
        model_version  = "1.0.0",
        confidence     = round(random.uniform(0.75, 0.99), 2),
        explanation    = scores["explanation"],
    )
    db.add(prediction)

    if scores["risk_level"] in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        db.add(Alert(
            incident_id = incident.id,
            channel     = "telegram",
            status      = AlertStatus.PENDING,
            message     = (
                f"🚨 {scores['risk_level'].value} ALERT\n"
                f"Type: {incident_type.value}\n"
                f"Area: {area['name']}, Lucknow\n"
                f"Score: {scores['risk_score']}/100\n"
                f"Details: {description}"
            ),
        ))

    db.commit()
    publish_to_redis(redis_client, incident, prediction)

    return {
        "id":         incident.id,
        "type":       incident_type.value,
        "area":       area["name"],
        "risk_level": scores["risk_level"].value,
        "score":      scores["risk_score"],
        "lat":        round(lat, 6),
        "lon":        round(lon, 6),
        "desc":       description,
    }

def generate_single_event(force_event_type=None):
    area = random.choice(LUCKNOW_AREAS)

    lat = area["lat"] + np.random.normal(0, 0.002)
    lon = area["lon"] + np.random.normal(0, 0.002)

    incident_type = force_event_type or random.choice(list(IncidentType))

    description = random.choice(DESCRIPTIONS.get(incident_type, ["Unknown event"]))

    occurred_at = datetime.utcnow()
    hour = occurred_at.hour

    scores = _engine.compute(incident_type, area, hour, description)

    return {
        "id": fake.uuid4(),
        "event_type": incident_type.value,
        "latitude": float(round(lat, 6)),
        "longitude": float(round(lon, 6)),
        "area_name": area["name"],
        "city": "Lucknow",
        "occurred_at": occurred_at.isoformat(),
        "risk_score": scores["risk_score"],
        "risk_level": scores["risk_level"].value,
        "explanation": {
            "dominant_signal": max(
                ["cv", "nlp", "location", "time"],
                key=lambda k: scores[f"{k}_score"]
            ),
            "formula": {
                "cv_score": scores["cv_score"],
                "nlp_score": scores["nlp_score"],
                "location_score": scores["location_score"],
                "time_score": scores["time_score"],
                "final_score": scores["risk_score"],
            }
        },
        "raw_input": {
            "description": description
        }
    }

# ── Batch generator ───────────────────────────────────────────────────────────

def generate_batch(n=20, db=None, redis_client=None) -> list[dict]:
    # Case 1: No DB → used in tests
    if db is None:
        return [generate_single_event() for _ in range(n)]

    # Case 2: With DB → production mode
    results = []
    for _ in range(n):
        try:
            event = generate_one_event(db, redis_client)
            results.append(event)
        except Exception as e:
            logger.error(f"Event generation failed: {e}")
            db.rollback()

    return results


def generate_historical_data(db, days=30, events_per_day=50) -> int:
    """Seed the DB with N days of past data. Returns total events inserted."""
    total = 0
    redis_client = None   # don't publish historical events to stream
    for day_offset in range(days):
        base_date = datetime.utcnow() - timedelta(days=day_offset)
        for _ in range(events_per_day):
            hour   = random.choices(range(24), weights=_hour_weights(), k=1)[0]
            ts     = base_date.replace(hour=hour, minute=random.randint(0,59),
                                       second=random.randint(0,59), microsecond=0)
            try:
                generate_one_event(db, redis_client, occurred_at=ts)
                total += 1
            except Exception as e:
                logger.error(f"Historical event failed: {e}")
                db.rollback()
    return total


def _hour_weights():
    w = [1.0] * 24
    for h in [8, 9, 10, 17, 18, 19, 20]: w[h] = 3.0
    for h in [22, 23, 0, 1]:              w[h] = 2.0
    for h in [3, 4, 5]:                   w[h] = 0.3
    total = sum(w)
    return [x / total for x in w]


# ── Rich terminal table ───────────────────────────────────────────────────────

def print_event_table(events: list[dict]):
    table = Table(title="Generated Events", show_lines=True)
    table.add_column("ID",    style="dim",     width=6)
    table.add_column("Type",  style="cyan",    width=12)
    table.add_column("Area",  style="white",   width=18)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Level", width=10)
    table.add_column("Description", style="dim")

    level_colors = {"LOW":"green","MEDIUM":"yellow","HIGH":"orange3","CRITICAL":"red"}
    for e in events:
        color = level_colors.get(e["risk_level"], "white")
        table.add_row(
            str(e["id"]), e["type"], e["area"], str(e["score"]),
            f"[{color}]{e['risk_level']}[/{color}]", e["desc"][:50],
        )
    console.print(table)


# ── Standalone loop ───────────────────────────────────────────────────────────

def run_simulation_loop():
    from app.database import init_db
    init_db()
    db           = SessionLocal()
    redis_client = get_redis_client()
    seed_hotspots(db)

    console.print(f"\n[bold green]Simulation started[/bold green] — "
                  f"city={settings.simulate_city} batch={settings.simulate_events_per_batch}")

    batch_num = 0
    try:
        while True:
            batch_num += 1
            logger.info(f"Batch #{batch_num}")
            events = generate_batch(settings.simulate_events_per_batch, db, redis_client)
            print_event_table(events)
            time.sleep(settings.simulate_interval_seconds)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")
    finally:
        db.close()


if __name__ == "__main__":
    run_simulation_loop()