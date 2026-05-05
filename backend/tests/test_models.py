"""
test_models.py
──────────────
Unit tests for Phase 1 — data generator and risk scoring engine.
These tests don't need a DB connection (pure logic tests).
Run with: pytest tests/ -v
"""

import pytest
from datetime import datetime

from app.simulator.data_generator import (
    generate_one_event,
    generate_batch,
    RiskScoringEngine,
    LUCKNOW_AREAS,
)
from app.models.events import IncidentType, RiskLevel
from app.database import SessionLocal


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def engine():
    return RiskScoringEngine()


# ── Event generation tests ──────────────────────────────────────────────────

class TestEventGeneration:

    def test_generates_single_event(self, db):
        event = generate_one_event(db)
        assert event is not None
        assert "id" in event
        assert "type" in event
        assert "area" in event
        assert "score" in event
        assert "risk_level" in event

    def test_batch_generates_correct_count(self, db):
        for n in [1, 10, 20]:
            batch = generate_batch(n=n, db=db)
            assert len(batch) == n

    def test_batch_events_have_unique_ids(self, db):
        batch = generate_batch(n=30, db=db)
        ids = [e["id"] for e in batch]
        assert len(ids) == len(set(ids))

    def test_event_area_is_valid(self, db):
        valid_areas = {a["name"] for a in LUCKNOW_AREAS}
        for _ in range(20):
            event = generate_one_event(db)
            assert event["area"] in valid_areas


# ── Risk scoring tests ──────────────────────────────────────────────────────

class TestRiskScoring:

    def test_risk_score_range(self, engine):
        for event_type in IncidentType:
            result = engine.compute(event_type, {"risk": 0.5}, hour=12)
            assert 0 <= result["risk_score"] <= 100

    def test_risk_level_matches_score(self, engine):
        result = engine.compute(IncidentType.ACCIDENT, {"risk": 0.8}, hour=12)
        score = result["risk_score"]
        level = result["risk_level"]

        if score >= 75:
            assert level == RiskLevel.CRITICAL
        elif score >= 50:
            assert level == RiskLevel.HIGH
        elif score >= 25:
            assert level == RiskLevel.MEDIUM
        else:
            assert level == RiskLevel.LOW

    def test_high_risk_area_increases_score(self, engine):
        high = [engine.compute(IncidentType.CRIME, {"risk": 0.9}, 12)["risk_score"] for _ in range(20)]
        low  = [engine.compute(IncidentType.CRIME, {"risk": 0.1}, 12)["risk_score"] for _ in range(20)]

        assert sum(high)/len(high) > sum(low)/len(low)

    def test_rush_hour_effect(self, engine):
        rush = [engine.compute(IncidentType.ACCIDENT, {"risk": 0.5}, 9)["time_score"] for _ in range(20)]
        off  = [engine.compute(IncidentType.ACCIDENT, {"risk": 0.5}, 3)["time_score"] for _ in range(20)]

        assert sum(rush)/len(rush) > sum(off)/len(off)


# ── Helpers ─────────────────────────────────────────────────────────────────

def random_areas(n: int):
    import random
    return random.choices(LUCKNOW_AREAS, k=n)