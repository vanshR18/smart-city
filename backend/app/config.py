from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ─────────────────────────────────────────────────────────────────
    app_name:    str = "SmartCityAI"
    app_version: str = "1.0.0"
    app_env:     str = "development"

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Simulator ─────────────────────────────────────────────────────────────
    simulate_city:               str = "Lucknow"
    simulate_events_per_batch:   int = 20
    simulate_interval_seconds:   int = 5

    # ── Alerts ─────────────────────────────────────────────────────────────────
    telegram_bot_token:  str   = "your_token_here"
    telegram_chat_id:    str   = "your_chat_id_here"
    alert_risk_threshold: float = 55.0   # score >= this triggers an alert

    class Config:
        env_file            = ".env"
        env_file_encoding   = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()