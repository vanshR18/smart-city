"""
train_timeseries.py
───────────────────
Trains Prophet time-series models using either:
  A) Data from the PostgreSQL database (if running)
  B) Synthetic data (if no DB available)

Usage:
  cd backend
  python scripts/train_timeseries.py              # uses DB if available
  python scripts/train_timeseries.py --synthetic  # always use synthetic data
  python scripts/train_timeseries.py --days 60    # use last 60 days from DB
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Train time-series models")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of DB")
    parser.add_argument("--days",      type=int, default=60,
                        help="Days of history to use from DB")
    parser.add_argument("--events-per-day", type=int, default=50,
                        help="Events per day for synthetic mode")
    return parser.parse_args()


def fetch_from_db(days: int) -> list[dict]:
    """Pulls recent events from PostgreSQL."""
    try:
        from app.database import SessionLocal
        from app.models.events import IncidentEvent
        from datetime import datetime, timedelta

        db    = SessionLocal()
        since = datetime.utcnow() - timedelta(days=days)

        rows = db.query(IncidentEvent).filter(
            IncidentEvent.occurred_at >= since
        ).all()
        db.close()

        events = [
            {
                "occurred_at": r.occurred_at.isoformat(),
                "event_type":  r.event_type.value,
            }
            for r in rows
        ]
        logger.info(f"Fetched {len(events)} events from DB (last {days} days)")
        return events

    except Exception as e:
        logger.warning(f"DB fetch failed ({e}). Falling back to synthetic data.")
        return []


def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold]SmartCityAI — Time-Series Training[/bold]\n"
        f"Mode: {'Synthetic' if args.synthetic else 'DB + fallback to synthetic'}\n"
        f"Days: {args.days}",
        border_style="blue",
    ))

    # Get training data
    events = []
    if not args.synthetic:
        events = fetch_from_db(args.days)

    if len(events) < 100:
        logger.info(f"Only {len(events)} DB events — augmenting with synthetic data")
        from app.timeseries.prophet_model import generate_synthetic_ts_data
        synthetic = generate_synthetic_ts_data(days=args.days)
        events    = events + synthetic
        logger.info(f"Total events for training: {len(events)}")

    # Train all models
    from app.timeseries.prophet_model import train_all_models
    results = train_all_models(events)

    # Print results table
    table = Table(title="Training Results", border_style="dim")
    table.add_column("Model",    style="cyan")
    table.add_column("Status",   style="green")
    table.add_column("MAE",      justify="right")
    table.add_column("RMSE",     justify="right")
    table.add_column("Peak Hours")

    for name, result in results.items():
        status    = result.get("status", "?")
        mae       = str(round(result.get("test_mae", 0), 3))  if result.get("test_mae")  else "—"
        rmse      = str(round(result.get("test_rmse", 0), 3)) if result.get("test_rmse") else "—"
        insights  = result.get("peak_insights", {})
        peak_h    = str(insights.get("peak_hours", "—")) if insights else "—"
        table.add_row(name, status, mae, rmse, peak_h)

    console.print(table)

    console.print(Panel.fit(
        "[bold green]Time-series training complete![/bold green]\n\n"
        "View in MLflow:\n"
        "  [bold]mlflow ui --port 5000[/bold]\n"
        "  Open: http://localhost:5000  →  experiment: emergency-timeseries\n\n"
        "Test the time profile:\n"
        "  Start API: [bold]uvicorn main:app --reload[/bold]\n"
        "  GET http://localhost:8000/risk/time-profile?event_type=ACCIDENT",
        border_style="green",
    ))


if __name__ == "__main__":
    main()