"""
websocket_manager.py
────────────────────
Manages all active WebSocket connections from the React dashboard.

How it works:
  - Dashboard opens ws://localhost:8000/ws/live on page load
  - ConnectionManager stores that socket in a set
  - When any new event is processed, broadcast() sends it to ALL open sockets
  - If a client disconnects (tab closed, network drop) it's removed cleanly

Why not use Redis pub/sub directly from the browser?
  Browsers can't connect to Redis. The FastAPI WebSocket endpoint
  acts as the bridge: Redis stream → Python → WebSocket → React.

Thread safety:
  FastAPI runs on asyncio. All broadcast calls are awaited,
  so no threading primitives needed.
"""

import json
from typing import Any
from fastapi import WebSocket
from loguru import logger


class ConnectionManager:
    """
    Holds all active WebSocket connections.
    One instance shared across the whole app (module-level singleton below).
    """

    def __init__(self):
        # Set of currently connected WebSocket objects
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        """Accept and register a new WebSocket connection."""
        await ws.accept()
        self.active.add(ws)
        logger.info(f"WS connected — total active: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        """Remove a disconnected socket (called in finally blocks)."""
        self.active.discard(ws)
        logger.info(f"WS disconnected — total active: {len(self.active)}")

    async def broadcast(self, data: dict | str):
        """
        Sends data to every connected client.
        Silently drops dead connections (client closed tab etc.).

        Args:
            data: dict (will be JSON-serialised) or raw string
        """
        if not self.active:
            return   # nobody listening — skip serialisation

        payload = json.dumps(data) if isinstance(data, dict) else data
        dead    = set()

        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                # Socket is dead — collect for cleanup
                dead.add(ws)

        # Clean up dead sockets
        for ws in dead:
            self.active.discard(ws)

        if dead:
            logger.debug(f"Removed {len(dead)} dead WebSocket(s)")

    async def send_to(self, ws: WebSocket, data: dict):
        """Send to a single specific client (used for initial state on connect)."""
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            self.active.discard(ws)

    @property
    def connection_count(self) -> int:
        return len(self.active)


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this everywhere: from app.alerts.websocket_manager import ws_manager
ws_manager = ConnectionManager()