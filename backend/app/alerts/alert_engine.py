"""
alert_engine.py
───────────────
The alert pipeline:

  New event saved to DB
       ↓
  alert_engine.process_event(event)
       ↓
  ┌─────────────────────────────────┐
  │  1. Check risk threshold        │  score >= ALERT_RISK_THRESHOLD?
  │  2. Broadcast to WebSocket      │  pushes to all dashboard clients
  │  3. Send Telegram (if warranted)│  CRITICAL / HIGH only
  │  4. Mark alert_sent in DB       │  prevents re-alerting
  └─────────────────────────────────┘

Redis stream consumer:
  A background task runs in the FastAPI lifespan,
  continuously reading from the smart_city:events Redis stream.
  Every event published by the simulator is processed here.
  This decouples ingestion from alerting cleanly.
"""

import asyncio
import json
from datetime import datetime
from loguru import logger

from app.config import get_settings
from app.alerts.websocket_manager import ws_manager
from app.alerts.telegram_bot import send_telegram_alert

settings = get_settings()

# ── Alert stats (in-memory, reset on restart)
_stats = {
    "events_processed":  0,
    "alerts_sent":       0,
    "telegram_sent":     0,
    "ws_broadcasts":     0,
    "started_at":        datetime.utcnow().isoformat(),
}


# ── Core event processor 
async def process_event(event: dict, db=None) -> dict:
    """
    Central processing function for every new event.

    Called:
    - After /simulate/batch saves events
    - After /predict/text or /predict/image produces a result
    - By the Redis stream consumer (background task)

    Returns a dict with what actions were taken.
    """
    _stats["events_processed"] += 1

    risk_score = float(event.get("risk_score",  0.0))
    risk_level = event.get("risk_level",         "LOW")
    threshold  = settings.alert_risk_threshold

    actions = {
        "event_id":   event.get("id", "unknown"),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "ws_sent":    False,
        "telegram":   False,
        "threshold":  threshold,
    }

    # ── 1. Always broadcast to WebSocket dashboard 
    # Build a clean, small payload (don't send the full event — dashboard
    # only needs what it renders on the map)
    ws_payload = _build_ws_payload(event)
    await ws_manager.broadcast(ws_payload)
    actions["ws_sent"] = True
    _stats["ws_broadcasts"] += 1

    # ── 2. Check threshold for alert 
    if risk_score >= threshold:
        _stats["alerts_sent"] += 1
        actions["alert_triggered"] = True

        # ── 3. Send Telegram
        telegram_sent = await send_telegram_alert(event)
        actions["telegram"] = telegram_sent
        if telegram_sent:
            _stats["telegram_sent"] += 1

        # ── 4. Mark alert_sent in DB 
        if db:
            _mark_alert_sent_in_db(db, event.get("id"))

        logger.warning(
            f"ALERT [{risk_level}] score={risk_score:.1f} "
            f"@ {event.get('area_name','?')} | "
            f"Telegram={'✓' if telegram_sent else '✗'}"
        )
    else:
        actions["alert_triggered"] = False

    return actions


def _build_ws_payload(event: dict) -> dict:
    """
    Builds the WebSocket message sent to the dashboard.
    Kept small — only fields the map and panel actually render.
    """
    return {
        "type":        "new_event",     # React checks this to know how to handle
        "id":          event.get("id"),
        "event_type":  event.get("event_type"),
        "area_name":   event.get("area_name"),
        "latitude":    event.get("latitude"),
        "longitude":   event.get("longitude"),
        "risk_score":  event.get("risk_score"),
        "risk_level":  event.get("risk_level"),
        "occurred_at": event.get("occurred_at"),
        "raw_input":   event.get("raw_input"),
        "explanation": {
            "dominant_signal": event.get("explanation", {}).get("dominant_signal"),
            "reasons":         event.get("explanation", {}).get("reasons", []),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


def _mark_alert_sent_in_db(db, event_id: str):
    """Updates RiskPrediction.alert_sent = True in DB."""
    try:
        from app.models.events import RiskPrediction
        pred = db.query(RiskPrediction).filter(
            RiskPrediction.event_id == event_id
        ).first()
        if pred:
            pred.alert_sent    = True
            pred.alert_sent_at = datetime.utcnow()
            db.commit()
    except Exception as e:
        logger.error(f"Failed to mark alert_sent in DB: {e}")


# ── Redis stream consumer 
async def redis_stream_consumer(redis_url: str):
    """
    Background asyncio task that reads from the Redis stream
    and processes every event through the alert pipeline.

    This runs forever inside the FastAPI lifespan context.
    Any exception in one iteration is caught and logged — the loop
    continues so a single bad event doesn't kill the consumer.

    Redis XREAD semantics:
    - '$' means "only new messages from now on" (not old ones)
    - block=1000 means "wait up to 1 second before checking again"
      (prevents busy-waiting and burns 0% CPU when idle)
    """
    import redis.asyncio as aioredis

    logger.info("Redis stream consumer starting...")
    r = aioredis.from_url(redis_url, decode_responses=True)

    stream_key = "smart_city:events"
    last_id    = "$"   # only consume NEW messages

    logger.success("Redis stream consumer listening on smart_city:events")

    while True:
        try:
            # block=1000 → wait 1 second max per call (non-busy wait)
            messages = await r.xread(
                {stream_key: last_id}, count=10, block=1000
            )

            if not messages:
                continue   # timeout, loop again

            for stream_name, msg_list in messages:
                for msg_id, msg_data in msg_list:
                    last_id = msg_id   # advance cursor

                    try:
                        event = _parse_stream_message(msg_data)
                        await process_event(event)
                    except Exception as e:
                        logger.error(f"Error processing stream message {msg_id}: {e}")

        except asyncio.CancelledError:
            logger.info("Redis consumer cancelled — shutting down")
            break
        except Exception as e:
            logger.error(f"Redis consumer error: {e}. Retrying in 3s...")
            await asyncio.sleep(3)


def _parse_stream_message(msg_data: dict) -> dict:
    """
    Redis streams store everything as strings.
    Parse JSON fields back to their proper types.
    """
    event = {}
    for k, v in msg_data.items():
        try:
            # Try JSON parse first (handles nested dicts, lists, booleans)
            parsed = json.loads(v)
            event[k] = parsed
        except (json.JSONDecodeError, TypeError):
            # Plain string (e.g. area_name, event_type)
            event[k] = v

    # Ensure numeric types
    for field in ["risk_score", "cv_score", "nlp_score", "location_score", "time_score",
                  "latitude", "longitude"]:
        if field in event and isinstance(event[field], str):
            try:
                event[field] = float(event[field])
            except ValueError:
                pass

    return event


# ── Stats endpoint helper 
def get_alert_stats() -> dict:
    return {
        **_stats,
        "active_ws_connections": ws_manager.connection_count,
        "alert_threshold":       settings.alert_risk_threshold,
    }