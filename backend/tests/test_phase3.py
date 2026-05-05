"""
test_phase3.py
──────────────
Tests for Phase 3: CV detector, time-series analyzer, and Risk Engine.
All tests run without DB, GPU, or trained models.

Run:
  pytest tests/test_phase3.py -v
"""

import pytest
from datetime import datetime


# ════════════════════════════════════════════════════════════════════════════
# CV DETECTOR TESTS
# ════════════════════════════════════════════════════════════════════════════
class TestCVDetector:

    def test_simulate_returns_cv_result(self):
        from app.cv.detector import _simulate_cv_result, CVResult
        result = _simulate_cv_result()
        assert isinstance(result, CVResult)

    def test_cv_result_score_in_range(self):
        from app.cv.detector import _simulate_cv_result
        for _ in range(20):
            r = _simulate_cv_result()
            assert 0.0 <= r.cv_score <= 1.0, f"cv_score={r.cv_score} out of [0,1]"

    def test_cv_result_label_is_valid(self):
        from app.cv.detector import _simulate_cv_result
        valid = {"ACCIDENT","FIRE","FLOOD","CRIME","CROWD","MEDICAL","NORMAL"}
        for _ in range(20):
            r = _simulate_cv_result()
            assert r.label in valid, f"Invalid label: {r.label}"

    def test_normal_label_not_emergency(self):
        from app.cv.detector import CVResult
        r = CVResult(label="NORMAL", cv_score=0.05, is_emergency=False)
        assert r.is_emergency is False

    def test_emergency_label_is_emergency(self):
        from app.cv.detector import CVResult
        r = CVResult(label="FIRE", cv_score=0.91, is_emergency=True)
        assert r.is_emergency is True

    def test_detect_image_returns_result_without_model(self):
        """detect_image falls back to simulation when ultralytics not available."""
        from app.cv.detector import detect_image, CVResult
        # Pass a dummy bytes object — should not crash, should simulate
        result = detect_image(b"fake_image_bytes")
        assert isinstance(result, CVResult)
        assert result.cv_score >= 0.0

    def test_coco_to_emergency_mapping_complete(self):
        from app.cv.detector import COCO_TO_EMERGENCY
        for cls_name, (emergency, boost) in COCO_TO_EMERGENCY.items():
            assert emergency in {"ACCIDENT","FIRE","FLOOD","CRIME","CROWD","MEDICAL","NORMAL"}
            assert 0.0 <= boost <= 1.0, f"boost={boost} for {cls_name} out of range"

    def test_get_cv_model_info_returns_dict(self):
        from app.cv.detector import get_cv_model_info
        info = get_cv_model_info()
        assert "model_loaded" in info
        assert "method" in info
        assert info["method"] in ["yolov8", "simulated"]

    def test_simulated_detections_have_required_fields(self):
        from app.cv.detector import _simulate_cv_result
        for _ in range(10):
            result = _simulate_cv_result()
            for det in result.detections:
                assert hasattr(det, "class_name")
                assert hasattr(det, "confidence")
                assert hasattr(det, "bbox")
                assert hasattr(det, "emergency")
                assert len(det.bbox) == 4

    def test_frame_count_default_is_one(self):
        from app.cv.detector import _simulate_cv_result
        result = _simulate_cv_result()
        assert result.frame_count == 1


# ════════════════════════════════════════════════════════════════════════════
# RISK ENGINE TESTS
# ════════════════════════════════════════════════════════════════════════════
class TestRiskEngine:

    def _make_ts(self, hour: int = 9) -> datetime:
        return datetime(2024, 6, 10, hour, 30, 0)

    def test_compute_returns_risk_score_object(self):
        from app.risk_engine.scorer import compute_risk_score, RiskScore
        result = compute_risk_score(
            event_type="ACCIDENT", area_name="Hazratganj",
            timestamp=self._make_ts(),
            cv_score=0.8, nlp_score=0.7,
        )
        assert isinstance(result, RiskScore)

    def test_risk_score_in_range(self):
        from app.risk_engine.scorer import compute_risk_score
        for cv in [0.1, 0.5, 0.9]:
            for nlp in [0.1, 0.5, 0.9]:
                r = compute_risk_score(
                    "FIRE", "Kaiserbagh", self._make_ts(),
                    cv_score=cv, nlp_score=nlp,
                )
                assert 0.0 <= r.risk_score <= 100.0

    def test_risk_level_matches_thresholds(self):
        from app.risk_engine.scorer import compute_risk_score
        for cv in [0.05, 0.4, 0.7, 0.95]:
            r = compute_risk_score(
                "ACCIDENT", "Hazratganj", self._make_ts(),
                cv_score=cv, nlp_score=cv,
            )
            s = r.risk_score
            if s >= 75:   assert r.risk_level == "CRITICAL"
            elif s >= 55: assert r.risk_level == "HIGH"
            elif s >= 35: assert r.risk_level == "MEDIUM"
            else:         assert r.risk_level == "LOW"

    def test_high_cv_score_raises_level(self):
        from app.risk_engine.scorer import compute_risk_score
        high = compute_risk_score("FIRE","X",self._make_ts(),cv_score=0.95,nlp_score=0.9)
        low  = compute_risk_score("FIRE","X",self._make_ts(),cv_score=0.10,nlp_score=0.1)
        assert high.risk_score > low.risk_score

    def test_missing_cv_degrades_gracefully(self):
        """Score should still be computed when cv_score is None."""
        from app.risk_engine.scorer import compute_risk_score
        r = compute_risk_score(
            "ACCIDENT","Hazratganj",self._make_ts(),
            cv_score=None, nlp_score=0.8,
        )
        assert r.risk_score > 0

    def test_missing_nlp_degrades_gracefully(self):
        from app.risk_engine.scorer import compute_risk_score
        r = compute_risk_score(
            "FIRE","Kaiserbagh",self._make_ts(),
            cv_score=0.9, nlp_score=None,
        )
        assert r.risk_score > 0

    def test_weights_renormalize_when_signal_missing(self):
        from app.risk_engine.scorer import compute_risk_score
        r = compute_risk_score(
            "ACCIDENT","X",self._make_ts(),
            cv_score=None, nlp_score=0.7,
            location_score=0.5, time_score=0.6,
        )
        # Weights that were used must sum to ~1.0
        used_weights = r.weights_used
        total = sum(used_weights.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"

    def test_explanation_has_required_keys(self):
        from app.risk_engine.scorer import compute_risk_score
        r = compute_risk_score(
            "CRIME","Thakurganj",self._make_ts(22),
            cv_score=0.7, nlp_score=0.6,
        )
        for key in ["dominant_signal","reasons","contributions","raw_signals","formula"]:
            assert key in r.explanation, f"Missing key: {key}"

    def test_dominant_signal_is_valid(self):
        from app.risk_engine.scorer import compute_risk_score
        r = compute_risk_score(
            "FLOOD","Chinhat",self._make_ts(),
            cv_score=0.8, nlp_score=0.3,
        )
        assert r.explanation["dominant_signal"] in ["cv","nlp","location","time"]

    def test_fire_gets_severity_boost(self):
        from app.risk_engine.scorer import compute_risk_score, EVENT_SEVERITY_BOOST
        fire    = compute_risk_score("FIRE",    "X", self._make_ts(),
                                     cv_score=0.6, nlp_score=0.6)
        normal  = compute_risk_score("NORMAL",  "X", self._make_ts(),
                                     cv_score=0.6, nlp_score=0.6)
        assert fire.risk_score > normal.risk_score

    def test_to_dict_contains_all_fields(self):
        from app.risk_engine.scorer import compute_risk_score
        r    = compute_risk_score("ACCIDENT","X",self._make_ts(),cv_score=0.7)
        d    = r.to_dict()
        for key in ["risk_score","risk_level","cv_score","explanation","event_type"]:
            assert key in d, f"Missing key in to_dict: {key}"

    def test_hotspot_cache_load(self):
        from app.risk_engine.scorer import load_hotspot_cache, _get_location_score
        load_hotspot_cache([
            {"area_name": "TestArea", "risk_score": 0.88},
            {"area_name": "SafeArea", "risk_score": 0.12},
        ])
        assert abs(_get_location_score("TestArea") - 0.88) < 0.001
        assert abs(_get_location_score("SafeArea") - 0.12) < 0.001
        assert _get_location_score("NonExistent") == 0.5   # fallback

    def test_area_risk_summary_sorted_by_heat(self):
        from app.risk_engine.scorer import compute_area_risk_summary
        preds = [
            {"area_name":"A","latitude":26.8,"longitude":80.9,"risk_score":90,"event_type":"FIRE"},
            {"area_name":"B","latitude":26.9,"longitude":81.0,"risk_score":20,"event_type":"NORMAL"},
            {"area_name":"A","latitude":26.8,"longitude":80.9,"risk_score":85,"event_type":"FIRE"},
        ]
        summaries = compute_area_risk_summary(preds)
        assert summaries[0]["area_name"] == "A"   # highest risk first
        assert summaries[0]["incident_count"] == 2


# ════════════════════════════════════════════════════════════════════════════
# TIME-SERIES ANALYZER TESTS
# ════════════════════════════════════════════════════════════════════════════
class TestTimeSeriesAnalyzer:

    def test_heuristic_returns_score(self):
        from app.timeseries.analyzer import _score_with_heuristics
        r = _score_with_heuristics(hour=9, weekday=0)
        assert 0.0 <= r["time_score"] <= 1.0

    def test_rush_hour_higher_than_offpeak(self):
        from app.timeseries.analyzer import _score_with_heuristics
        rush    = _score_with_heuristics(9,  0)["time_score"]
        offpeak = _score_with_heuristics(3,  0)["time_score"]
        assert rush > offpeak

    def test_peak_hour_flag_correct(self):
        from app.timeseries.analyzer import _score_with_heuristics
        assert _score_with_heuristics(9,  0)["peak_hour"] is True
        assert _score_with_heuristics(18, 0)["peak_hour"] is True
        assert _score_with_heuristics(3,  0)["peak_hour"] is False
        assert _score_with_heuristics(14, 0)["peak_hour"] is False

    def test_weekend_flag(self):
        from app.timeseries.analyzer import _score_with_heuristics
        weekday = _score_with_heuristics(9, 0)["weekend"]
        weekend = _score_with_heuristics(9, 6)["weekend"]
        assert weekday is False
        assert weekend is True

    def test_get_time_score_returns_dict(self):
        from app.timeseries.analyzer import get_time_score
        r = get_time_score(datetime(2024, 6, 10, 9, 0), "ACCIDENT")
        assert "time_score"  in r
        assert "peak_hour"   in r
        assert "method"      in r
        assert 0.0 <= r["time_score"] <= 1.0

    def test_hourly_risk_profile_has_24_hours(self):
        from app.timeseries.analyzer import get_hourly_risk_profile
        profile = get_hourly_risk_profile("ACCIDENT")
        assert "hourly_profile" in profile
        assert len(profile["hourly_profile"]) == 24

    def test_models_status_returns_dict(self):
        from app.timeseries.analyzer import get_models_status
        status = get_models_status()
        assert "prophet_installed" in status
        assert "scoring_method"    in status


# ════════════════════════════════════════════════════════════════════════════
# PROPHET DATA PREPARATION TESTS (no model needed)
# ════════════════════════════════════════════════════════════════════════════
class TestProphetDataPrep:

    def test_prepare_df_returns_ds_y_columns(self):
        from app.timeseries.prophet_model import prepare_prophet_df
        events = [
            {"occurred_at": "2024-01-01T09:00:00", "event_type": "ACCIDENT"},
            {"occurred_at": "2024-01-01T10:00:00", "event_type": "FIRE"},
            {"occurred_at": "2024-01-01T11:00:00", "event_type": "ACCIDENT"},
        ]
        df = prepare_prophet_df(events)
        assert "ds" in df.columns
        assert "y"  in df.columns

    def test_prepare_df_filter_by_type(self):
        from app.timeseries.prophet_model import prepare_prophet_df
        events = [
            {"occurred_at": "2024-01-01T09:00:00", "event_type": "ACCIDENT"},
            {"occurred_at": "2024-01-01T09:30:00", "event_type": "FIRE"},
            {"occurred_at": "2024-01-01T09:45:00", "event_type": "FIRE"},
        ]
        df = prepare_prophet_df(events, event_type_filter="FIRE")
        assert df["y"].sum() == 2   # only 2 FIRE events

    def test_synthetic_data_generator(self):
        from app.timeseries.prophet_model import generate_synthetic_ts_data
        events = generate_synthetic_ts_data(days=7)
        assert len(events) > 100
        assert all("occurred_at" in e for e in events)
        assert all("event_type"  in e for e in events)

    def test_prepare_df_empty_events(self):
        from app.timeseries.prophet_model import prepare_prophet_df
        df = prepare_prophet_df([])
        assert len(df) == 0