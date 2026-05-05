"""
test_models.py
──────────────
Unit tests for Phase 1 — data generator and risk scoring engine.
These tests do NOT need a database connection.
All tests use pure functions from data_generator.py only.

Run:
  cd backend
  DATABASE_URL=postgresql://x:x@localhost/x pytest tests/test_models.py -v
"""

import pytest
import random
from datetime import datetime

from app.simulator.data_generator import (
    LUCKNOW_AREAS,
    RiskScoringEngine,
    _hour_weights,
    _engine,
    DESCRIPTIONS,
)
from app.models.events import IncidentType, RiskLevel


# ── Pure helper — generates one event dict WITHOUT touching the DB ────────────
def make_event(incident_type=None, area=None, hour=None):
    """
    Generates a scored event dict using only the simulator internals.
    Does NOT call generate_one_event() (which needs a DB session).
    """
    if area is None:
        area = random.choice(LUCKNOW_AREAS)
    if incident_type is None:
        incident_type = random.choice(list(IncidentType))
    if hour is None:
        hour = random.randint(0, 23)

    description = random.choice(DESCRIPTIONS.get(incident_type, ["Test event"]))
    scores = _engine.compute(incident_type, area, hour, description)

    return {
        "incident_type":  incident_type,
        "area_name":      area["name"],
        "latitude":       area["lat"],
        "longitude":      area["lon"],
        "description":    description,
        "hour":           hour,
        **scores,
    }


def make_batch(n=20):
    return [make_event() for _ in range(n)]


# ════════════════════════════════════════════════════════════════════════════
# EVENT GENERATION TESTS (pure — no DB)
# ════════════════════════════════════════════════════════════════════════════
class TestEventGeneration:

    def test_generates_single_event(self):
        event = make_event()
        assert event is not None
        assert "incident_type" in event
        assert "area_name" in event
        assert "latitude" in event
        assert "longitude" in event

    def test_event_has_all_required_fields(self):
        event = make_event()
        required = [
            "incident_type", "area_name", "latitude", "longitude",
            "description", "risk_score", "risk_level", "explanation",
            "cv_score", "nlp_score", "location_score", "time_score",
        ]
        for field in required:
            assert field in event, f"Missing field: {field}"

    def test_event_area_is_valid(self):
        valid_areas = {a["name"] for a in LUCKNOW_AREAS}
        for _ in range(20):
            event = make_event()
            assert event["area_name"] in valid_areas, \
                f"Invalid area: {event['area_name']}"

    def test_event_coordinates_within_lucknow_bounds(self):
        """Lucknow is roughly lat 26.7–27.1, lon 80.7–81.3"""
        for area in LUCKNOW_AREAS:
            assert 26.7 <= area["lat"] <= 27.1, f"Lat out of range: {area}"
            assert 80.7 <= area["lon"] <= 81.3, f"Lon out of range: {area}"

    def test_event_type_is_valid(self):
        valid_types = set(IncidentType)
        for _ in range(20):
            event = make_event()
            assert event["incident_type"] in valid_types

    def test_force_incident_type(self):
        for incident_type in [IncidentType.FIRE, IncidentType.ACCIDENT, IncidentType.NORMAL]:
            event = make_event(incident_type=incident_type)
            assert event["incident_type"] == incident_type

    def test_batch_generates_correct_count(self):
        for n in [1, 10, 20, 50]:
            batch = make_batch(n)
            assert len(batch) == n, f"Expected {n} events, got {len(batch)}"

    def test_batch_events_all_have_scores(self):
        batch = make_batch(20)
        for e in batch:
            assert e["risk_score"] is not None
            assert e["risk_level"] is not None

    def test_descriptions_exist_for_all_types(self):
        for incident_type in IncidentType:
            descs = DESCRIPTIONS.get(incident_type, [])
            assert len(descs) > 0, f"No descriptions for {incident_type}"

    def test_lucknow_areas_have_required_keys(self):
        for area in LUCKNOW_AREAS:
            for key in ["name", "lat", "lon", "risk"]:
                assert key in area, f"Missing key '{key}' in area: {area}"

    def test_lucknow_areas_risk_in_range(self):
        for area in LUCKNOW_AREAS:
            assert 0.0 <= area["risk"] <= 1.0, \
                f"Risk out of [0,1] for {area['name']}: {area['risk']}"

    def test_lucknow_has_minimum_areas(self):
        assert len(LUCKNOW_AREAS) >= 10, \
            f"Expected at least 10 areas, got {len(LUCKNOW_AREAS)}"


# ════════════════════════════════════════════════════════════════════════════
# RISK SCORING TESTS (pure — no DB)
# ════════════════════════════════════════════════════════════════════════════
class TestRiskScoring:

    def test_risk_score_in_valid_range(self):
        for incident_type in IncidentType:
            for area in random.choices(LUCKNOW_AREAS, k=3):
                result = _engine.compute(incident_type, area, hour=12)
                assert 0 <= result["risk_score"] <= 100, \
                    f"Score {result['risk_score']} out of [0,100] for {incident_type}"

    def test_risk_level_matches_score(self):
        """Risk level labels must match the score thresholds."""
        for _ in range(50):
            event = make_event()
            score = event["risk_score"]
            level = event["risk_level"]

            if score >= 75:
                assert level == RiskLevel.CRITICAL, \
                    f"score={score:.1f} should be CRITICAL, got {level}"
            elif score >= 50:
                assert level == RiskLevel.HIGH, \
                    f"score={score:.1f} should be HIGH, got {level}"
            elif score >= 25:
                assert level == RiskLevel.MEDIUM, \
                    f"score={score:.1f} should be MEDIUM, got {level}"
            else:
                assert level == RiskLevel.LOW, \
                    f"score={score:.1f} should be LOW, got {level}"

    def test_normal_events_have_lower_scores_than_fire(self):
        """NORMAL events should score lower than FIRE events on average."""
        area = LUCKNOW_AREAS[0]
        normal_scores = [
            _engine.compute(IncidentType.NORMAL, area, 12)["risk_score"]
            for _ in range(30)
        ]
        fire_scores = [
            _engine.compute(IncidentType.FIRE, area, 12)["risk_score"]
            for _ in range(30)
        ]
        assert sum(normal_scores) / len(normal_scores) < \
               sum(fire_scores)   / len(fire_scores), \
            "NORMAL average score should be less than FIRE average score"

    def test_explanation_is_string(self):
        for _ in range(10):
            event = make_event()
            assert isinstance(event["explanation"], str)
            assert len(event["explanation"]) > 0

    def test_explanation_contains_score(self):
        event = make_event()
        assert "Score" in event["explanation"] or "score" in event["explanation"]

    def test_all_signal_scores_in_range(self):
        for _ in range(20):
            event = make_event()
            for field in ["cv_score", "nlp_score", "location_score", "time_score"]:
                val = event[field]
                assert 0.0 <= val <= 1.0, \
                    f"{field}={val} out of [0,1]"

    def test_high_risk_area_produces_higher_location_score(self):
        """Charbagh (risk=0.8) should produce higher location_score than Indira Nagar (risk=0.3)"""
        high_area = next(a for a in LUCKNOW_AREAS if a["name"] == "Charbagh")
        low_area  = next(a for a in LUCKNOW_AREAS if a["name"] == "Indira Nagar")

        high_score = _engine.compute(IncidentType.ACCIDENT, high_area, 9)["location_score"]
        low_score  = _engine.compute(IncidentType.ACCIDENT, low_area,  9)["location_score"]

        assert high_score > low_score, \
            f"High-risk area should have higher location_score ({high_score} vs {low_score})"

    def test_weights_sum_to_one(self):
        weights = _engine.WEIGHTS
        total   = sum(weights.values())
        assert abs(total - 1.0) < 0.001, \
            f"Weights must sum to 1.0, got {total}"

    def test_fire_has_highest_cv_base(self):
        """FIRE should have the highest base CV confidence."""
        fire_cv = _engine.CV_BASE[IncidentType.FIRE]
        for itype, cv in _engine.CV_BASE.items():
            if itype != IncidentType.FIRE:
                assert fire_cv >= cv, \
                    f"FIRE CV base ({fire_cv}) should be >= {itype} ({cv})"


# ════════════════════════════════════════════════════════════════════════════
# HOUR WEIGHTS TESTS (pure — no DB)
# ════════════════════════════════════════════════════════════════════════════
class TestHourWeights:

    def test_hour_weights_sum_to_one(self):
        weights = _hour_weights()
        total   = sum(weights)
        assert abs(total - 1.0) < 0.001, \
            f"Hour weights must sum to 1.0, got {total}"

    def test_hour_weights_has_24_entries(self):
        weights = _hour_weights()
        assert len(weights) == 24

    def test_hour_weights_all_positive(self):
        weights = _hour_weights()
        for i, w in enumerate(weights):
            assert w > 0, f"Hour {i} weight must be positive, got {w}"

    def test_rush_hours_have_higher_weight(self):
        """Hours 8-10 and 17-20 should have higher weight than 3-5 AM."""
        weights   = _hour_weights()
        rush_avg  = sum(weights[h] for h in [8, 9, 10, 17, 18]) / 5
        quiet_avg = sum(weights[h] for h in [3, 4, 5]) / 3
        assert rush_avg > quiet_avg, \
            f"Rush hour weight ({rush_avg:.4f}) should exceed quiet hour ({quiet_avg:.4f})"

    def test_risk_scoring_uses_hour_correctly(self):
        """Rush hour (9am) should produce higher or equal time_score than 3am."""
        area = LUCKNOW_AREAS[0]
        rush_score  = _engine.compute(IncidentType.ACCIDENT, area, hour=9)["time_score"]
        quiet_score = _engine.compute(IncidentType.ACCIDENT, area, hour=3)["time_score"]
        # time_score comes from HOURLY_WEIGHTS which peaks at rush hours
        assert rush_score >= quiet_score, \
            f"Rush hour time_score ({rush_score}) should be >= quiet ({quiet_score})"


# ════════════════════════════════════════════════════════════════════════════
# LUCKNOW AREAS CONSISTENCY TESTS
# ════════════════════════════════════════════════════════════════════════════
class TestLucknowAreas:

    def test_no_duplicate_area_names(self):
        names = [a["name"] for a in LUCKNOW_AREAS]
        assert len(names) == len(set(names)), \
            f"Duplicate area names found: {[n for n in names if names.count(n) > 1]}"

    def test_charbagh_is_high_risk(self):
        """Charbagh (railway/bus hub) should have risk >= 0.7"""
        charbagh = next((a for a in LUCKNOW_AREAS if a["name"] == "Charbagh"), None)
        assert charbagh is not None, "Charbagh must be in LUCKNOW_AREAS"
        assert charbagh["risk"] >= 0.7, \
            f"Charbagh risk should be >= 0.7, got {charbagh['risk']}"

    def test_all_areas_have_lucknow_coordinates(self):
        """All areas must be within Lucknow city bounds."""
        for area in LUCKNOW_AREAS:
            assert 26.70 <= area["lat"] <= 27.10, \
                f"{area['name']} lat {area['lat']} outside Lucknow bounds"
            assert 80.70 <= area["lon"] <= 81.30, \
                f"{area['name']} lon {area['lon']} outside Lucknow bounds"