"""
telegram_bot.py
───────────────
Sends formatted emergency alerts to a Telegram chat.

Setup (takes 2 minutes):
  1. Open Telegram → search "@BotFather"
  2. Send /newbot → follow prompts → copy the token
  3. Add token to backend/.env as TELEGRAM_BOT_TOKEN=...
  4. Send any message to your bot, then visit:
     https://api.telegram.org/bot<TOKEN>/getUpdates
     Copy the "chat":{"id": ...} value → TELEGRAM_CHAT_ID=...

Why Telegram over email?
  - Instant push notification on your phone
  - Rich formatting (bold, emoji, links)
  - Free, no SMTP config needed
  - Easy to demo live to an interviewer

Throttling:
  We track sent alerts to avoid spamming the same event.
  CRITICAL events: always send
  HIGH events:     send max once per 5 minutes per area
  MEDIUM/LOW:      never send (not worth the noise)
"""

import httpx
import asyncio
from datetime import datetime, timedelta
from loguru import logger
from app.config import get_settings

settings = get_settings()

# ── Throttle tracker ──────────────────────────────────────────────────────────
# area_name → last_sent_at
_last_sent: dict[str, datetime] = {}
HIGH_THROTTLE_MINUTES = 5


def _should_send(risk_level: str, area_name: str) -> bool:
    """Returns True if this alert should be sent (not throttled)."""
    if risk_level == "CRITICAL":
        return True   # always send CRITICAL

    if risk_level == "HIGH":
        last = _last_sent.get(area_name)
        if last is None:
            return True
        return datetime.utcnow() - last > timedelta(minutes=HIGH_THROTTLE_MINUTES)

    return False   # MEDIUM and LOW are never alerted


def _mark_sent(area_name: str):
    _last_sent[area_name] = datetime.utcnow()


# ── Message formatting ────────────────────────────────────────────────────────
LEVEL_EMOJI = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
}

EVENT_EMOJI = {
    "ACCIDENT": "🚗",
    "FIRE":     "🔥",
    "FLOOD":    "🌊",
    "CRIME":    "🚨",
    "CROWD":    "👥",
    "MEDICAL":  "🏥",
    "NORMAL":   "✅",
}


def _format_message(event: dict) -> str:
    """
    Builds a formatted Telegram message (HTML parse mode).

    Example output:
    🔴 CRITICAL ALERT — SmartCity Lucknow

    🔥 Event:    FIRE
    📍 Location: Kaiserbagh
    ⚡ Score:    87.4 / 100
    🕐 Time:     14:32:05

    Why: strong visual detection (91%), high-risk area (Kaiserbagh)

    Dominant signal: cv
    """
    level      = event.get("risk_level",  "UNKNOWN")
    etype      = event.get("event_type",  "UNKNOWN")
    area       = event.get("area_name",   "Unknown")
    score      = event.get("risk_score",  0.0)
    occurred   = event.get("occurred_at", datetime.utcnow().isoformat())
    explanation = event.get("explanation", {})

    level_emoji = LEVEL_EMOJI.get(level, "⚠️")
    event_emoji = EVENT_EMOJI.get(etype,  "⚠️")

    # Parse time
    try:
        ts   = datetime.fromisoformat(occurred)
        time_str = ts.strftime("%H:%M:%S")
        date_str = ts.strftime("%d %b %Y")
    except Exception:
        time_str = "—"
        date_str = "—"

    reasons = explanation.get("reasons", [])
    reason_str = " | ".join(reasons) if reasons else "multiple signals"
    dominant   = explanation.get("dominant_signal", "—")

    msg = (
        f"{level_emoji} <b>{level} ALERT</b> — SmartCity Lucknow\n"
        f"{'─' * 28}\n"
        f"{event_emoji} <b>Event:</b>    {etype}\n"
        f"📍 <b>Location:</b> {area}\n"
        f"⚡ <b>Score:</b>    {score:.1f} / 100\n"
        f"🕐 <b>Time:</b>     {time_str}  ({date_str})\n"
        f"{'─' * 28}\n"
        f"<b>Why:</b> {reason_str}\n"
        f"<b>Dominant signal:</b> {dominant}\n"
    )
    return msg


# ── Sender ────────────────────────────────────────────────────────────────────
async def send_telegram_alert(event: dict) -> bool:
    """
    Sends a Telegram alert for an event if the risk level warrants it.

    Returns True if message was sent, False if throttled or failed.

    Design: uses httpx async client directly (no telegram library needed).
    This keeps the dependency footprint small and avoids bot polling loops.
    """
    risk_level = event.get("risk_level", "LOW")
    area_name  = event.get("area_name",  "Unknown")

    if not _should_send(risk_level, area_name):
        logger.debug(f"Telegram alert throttled: {risk_level} @ {area_name}")
        return False

    token   = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id

    if token == "your_token_here" or not token:
        logger.warning("Telegram token not configured. Skipping alert. "
                       "Set TELEGRAM_BOT_TOKEN in .env to enable.")
        return False

    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id":    chat_id,
        "text":       _format_message(event),
        "parse_mode": "HTML",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()

        _mark_sent(area_name)
        logger.success(f"Telegram alert sent: {risk_level} @ {area_name}")
        return True

    except httpx.HTTPStatusError as e:
        logger.error(f"Telegram API error {e.response.status_code}: {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def send_telegram_alert_sync(event: dict) -> bool:
    """
    Synchronous wrapper for use in non-async contexts.
    Uses asyncio.run() — don't call from inside an already-running event loop.
    """
    try:
        return asyncio.run(send_telegram_alert(event))
    except RuntimeError:
        # Already inside event loop (FastAPI context) — create a task instead
        loop = asyncio.get_event_loop()
        loop.create_task(send_telegram_alert(event))
        return True