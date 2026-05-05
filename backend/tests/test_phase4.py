"""
test_phase4.py
──────────────
Tests for the alert pipeline — no Telegram token or live WebSocket needed.
Run: pytest tests/test_phase4.py -v
"""

import pytest
import asyncio
from datetime import datetime


# ── WebSocket manager tests ───────────────────────────────────────────────────
class TestWebSocketManager:

    def test_initial_connection_count_zero(self):
        from app.alerts.websocket_manager import ConnectionManager
        mgr = ConnectionManager()
        assert mgr.connection_count == 0

    def test_disconnect_nonexistent_socket_no_crash(self):
        from app.alerts.websocket_manager import ConnectionManager
        mgr = ConnectionManager()
        mgr.disconnect(object())   # should not raise

    def test_broadcast_no_clients_no_crash(self):
        from app.alerts.websocket_manager import ConnectionManager
        mgr = ConnectionManager()
        asyncio.run(mgr.broadcast({"type": "test"}))

    def test_singleton_imported_consistently(self):
        from app.alerts.websocket_manager import ws_manager as m1
        from app.alerts.websocket_manager import ws_manager as m2
        assert m1 is m2   # same object every import


# ── Telegram formatter tests (no API call) ────────────────────────────────────
class TestTelegramFormatter:

    def _make_event(self, level="CRITICAL", etype="FIRE", area="Kaiserbagh",
                    score=88.5) -> dict:
        return {
            "risk_level":  level,
            "event_type":  etype,
            "area_name":   area,
            "risk_score":  score,
            "occurred_at": "2024-06-10T14:32:00",
            "explanation": {
                "dominant_signal": "cv",
                "reasons": ["strong visual detection (91%)", "high-risk area"],
            },
        }

    def test_format_message_contains_level(self):
        from app.alerts.telegram_bot import _format_message
        msg = _format_message(self._make_event("CRITICAL"))
        assert "CRITICAL" in msg

    def test_format_message_contains_area(self):
        from app.alerts.telegram_bot import _format_message
        msg = _format_message(self._make_event(area="Hazratganj"))
        assert "Hazratganj" in msg

    def test_format_message_contains_score(self):
        from app.alerts.telegram_bot import _format_message
        msg = _format_message(self._make_event(score=88.5))
        assert "88.5" in msg

    def test_format_message_contains_reasons(self):
        from app.alerts.telegram_bot import _format_message
        msg = _format_message(self._make_event())
        assert "strong visual detection" in msg

    def test_format_message_all_event_types(self):
        from app.alerts.telegram_bot import _format_message, EVENT_EMOJI
        for etype in ["ACCIDENT", "FIRE", "FLOOD", "CRIME", "CROWD", "MEDICAL"]:
            msg = _format_message(self._make_event(etype=etype))
            assert etype in msg

    def test_should_send_critical_always(self):
        from app.alerts.telegram_bot import _should_send, _last_sent
        _last_sent.clear()
        assert _should_send("CRITICAL", "AnyArea") is True
        # Even if recently sent, CRITICAL always goes through
        _last_sent["AnyArea"] = datetime.utcnow()
        assert _should_send("CRITICAL", "AnyArea") is True

    def test_should_send_high_not_throttled(self):
        from app.alerts.telegram_bot import _should_send, _last_sent
        _last_sent.clear()
        assert _should_send("HIGH", "NewArea") is True

    def test_should_not_send_medium(self):
        from app.alerts.telegram_bot import _should_send
        assert _should_send("MEDIUM", "AnyArea") is False

    def test_should_not_send_low(self):
        from app.alerts.telegram_bot import _should_send
        assert _should_send("LOW", "AnyArea") is False

    def test_no_token_skips_send(self):
        """When token is placeholder, send returns False without API call."""
        from app.alerts.telegram_bot import send_telegram_alert
        event = self._make_event()
        # Settings have placeholder token → should return False
        result = asyncio.run(send_telegram_alert(event))
        assert result is False


# ── Alert engine tests ────────────────────────────────────────────────────────
class TestAlertEngine:

    def _event(self, score=80.0, level="CRITICAL") -> dict:
        return {
            "id":          "test-event-001",
            "event_type":  "FIRE",
            "area_name":   "Kaiserbagh",
            "latitude":    26.853,
            "longitude":   80.935,
            "risk_score":  score,
            "risk_level":  level,
            "occurred_at": datetime.utcnow().isoformat(),
            "raw_input":   "Test fire alert",
            "explanation": {"dominant_signal": "cv", "reasons": ["test"]},
        }

    def test_process_event_returns_actions(self):
        from app.alerts.alert_engine import process_event
        event   = self._event(score=90.0, level="CRITICAL")
        actions = asyncio.run(process_event(event))
        assert "ws_sent"         in actions
        assert "telegram"        in actions
        assert "alert_triggered" in actions
        assert "risk_score"      in actions

    def test_high_score_triggers_alert(self):
        from app.alerts.alert_engine import process_event
        actions = asyncio.run(process_event(self._event(score=80.0, level="CRITICAL")))
        assert actions["alert_triggered"] is True

    def test_low_score_no_alert(self):
        from app.alerts.alert_engine import process_event
        actions = asyncio.run(process_event(self._event(score=20.0, level="LOW")))
        assert actions["alert_triggered"] is False

    def test_ws_always_broadcast(self):
        """WebSocket broadcast happens regardless of risk level."""
        from app.alerts.alert_engine import process_event
        for score, level in [(10.0, "LOW"), (90.0, "CRITICAL")]:
            actions = asyncio.run(process_event(self._event(score, level)))
            assert actions["ws_sent"] is True

    def test_get_alert_stats_structure(self):
        from app.alerts.alert_engine import get_alert_stats
        stats = get_alert_stats()
        for key in ["events_processed", "alerts_sent", "telegram_sent",
                    "ws_broadcasts", "active_ws_connections", "alert_threshold"]:
            assert key in stats, f"Missing key: {key}"

    def test_stats_increment_after_process(self):
        from app.alerts.alert_engine import process_event, _stats
        before = _stats["events_processed"]
        asyncio.run(process_event(self._event()))
        assert _stats["events_processed"] == before + 1

    def test_build_ws_payload_fields(self):
        from app.alerts.alert_engine import _build_ws_payload
        event   = self._event(90.0, "CRITICAL")
        payload = _build_ws_payload(event)
        for field in ["type", "id", "event_type", "area_name",
                      "latitude", "longitude", "risk_score", "risk_level"]:
            assert field in payload, f"Missing WS field: {field}"
        assert payload["type"] == "new_event"

    def test_parse_stream_message_converts_floats(self):
        from app.alerts.alert_engine import _parse_stream_message
        raw = {
            "risk_score": "82.5",
            "latitude":   "26.853",
            "event_type": "FIRE",
            "area_name":  "Kaiserbagh",
        }
        parsed = _parse_stream_message(raw)
        assert isinstance(parsed["risk_score"], float)
        assert isinstance(parsed["latitude"],   float)
        assert isinstance(parsed["event_type"], str)

    def test_parse_stream_message_handles_json_fields(self):
        from app.alerts.alert_engine import _parse_stream_message
        import json
        raw = {
            "explanation": json.dumps({"dominant_signal": "cv", "reasons": ["fire"]}),
            "risk_score":  "75.0",
        }
        parsed = _parse_stream_message(raw)
        assert isinstance(parsed["explanation"], dict)
        assert parsed["explanation"]["dominant_signal"] == "cv"