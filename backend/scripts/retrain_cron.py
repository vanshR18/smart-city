"""
retrain_cron.py
───────────────
Command-line entrypoint for the retraining pipeline.
Can be called manually OR scheduled via system cron / Docker cron.

Usage:
  cd backend
  python scripts/retrain_cron.py                    # normal run
  python scripts/retrain_cron.py --force            # always promote
  python scripts/retrain_cron.py --epochs 5         # more epochs
  python scripts/retrain_cron.py --dry-run          # check data only, no training

System cron example (retrain every day at 2 AM):
  0 2 * * * cd /app && python scripts/retrain_cron.py >> /var/log/retrain.log 2>&1
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
    p = argparse.ArgumentParser(description="SmartCityAI NLP retraining pipeline")
    p.add_argument("--force",    action="store_true", help="Always promote (skip F1 gate)")
    p.add_argument("--epochs",   type=int, default=3, help="Training epochs")
    p.add_argument("--min-f1",   type=float, default=0.01, help="Min F1 improvement to promote")
    p.add_argument("--dry-run",  action="store_true", help="Check data only, skip training")
    return p.parse_args()


def main():
    args = parse_args()

    console.print(Panel.fit(
        "[bold]SmartCityAI — MLOps Retraining Pipeline[/bold]\n"
        f"Force:     {args.force}\n"
        f"Epochs:    {args.epochs}\n"
        f"Min F1 Δ:  +{args.min_f1}\n"
        f"Dry run:   {args.dry_run}",
        border_style="blue",
    ))

    if args.dry_run:
        from app.mlops.retrain import fetch_recent_events_from_db, build_retrain_dataset
        events = fetch_recent_events_from_db()
        df     = build_retrain_dataset(events)
        if df is not None:
            console.print(f"\n[green]Dataset ready: {len(df)} samples[/green]")
            console.print(df["label_name"].value_counts().to_string())
        else:
            console.print("[yellow]Insufficient data for retraining[/yellow]")
        return

    from app.mlops.retrain import run_retraining_pipeline
    result = run_retraining_pipeline(
        force=args.force,
        min_f1_improvement=args.min_f1,
        num_epochs=args.epochs,
    )

    # Print result table
    table = Table(title="Retraining Result", border_style="dim")
    table.add_column("Field",  style="cyan")
    table.add_column("Value",  style="white")

    table.add_row("Status",       result["status"])
    table.add_row("Promoted",     str(result["promoted"]))
    table.add_row("Current F1",   f"{result['current_f1']:.4f}" if result["current_f1"] else "—")
    table.add_row("New F1",       f"{result['new_f1']:.4f}"     if result["new_f1"]     else "—")
    table.add_row("F1 Delta",     f"{result['f1_delta']:+.4f}"  if result["f1_delta"] is not None else "—")
    table.add_row("Duration",     f"{result['duration_sec']}s")
    table.add_row("MLflow Run",   result["run_id"] or "—")
    table.add_row("Reason",       result["reason"][:80] if result["reason"] else "—")

    console.print(table)

    color = {"promoted": "green", "rejected": "yellow",
             "skipped": "yellow", "failed": "red"}.get(result["status"], "white")
    console.print(f"\n[{color}]{result['reason']}[/{color}]")

    if result["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()