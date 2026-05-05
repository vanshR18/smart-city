"""
train_nlp.py
────────────
One-command script to train the NLP classifier.

Usage:
  cd backend
  python scripts/train_nlp.py                          # default config
  python scripts/train_nlp.py --epochs 3 --lr 3e-5    # custom hyperparams
  python scripts/train_nlp.py --fast                   # quick run for testing

What it does:
  1. Checks dataset exists (prompts you to run download_dataset.py if not)
  2. Trains DistilBERT with full MLflow tracking
  3. Saves best model to app/nlp/models_store/best_model/
  4. Prints MLflow UI command so you can view results in browser
"""

import sys
import argparse
from pathlib import Path

# Make app/ importable from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.panel import Panel

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the emergency NLP classifier")
    parser.add_argument("--epochs",      type=int,   default=5,     help="Number of epochs")
    parser.add_argument("--lr",          type=float, default=2e-5,  help="Learning rate")
    parser.add_argument("--batch-size",  type=int,   default=16,    help="Training batch size")
    parser.add_argument("--freeze",      type=int,   default=4,     help="Number of DistilBERT layers to freeze")
    parser.add_argument("--fast",        action="store_true",        help="Quick run: 1 epoch, small batch (for testing)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Check dataset exists
    data_dir = Path(__file__).parent.parent / "app" / "nlp" / "data"
    if not (data_dir / "train.csv").exists():
        console.print("[red]Dataset not found![/red]")
        console.print("Run this first:  [bold]python scripts/download_dataset.py[/bold]")
        sys.exit(1)

    # Build config
    config = {
        "num_epochs":       1 if args.fast else args.epochs,
        "learning_rate":    args.lr,
        "train_batch_size": 8 if args.fast else args.batch_size,
        "freeze_layers":    args.freeze,
    }

    console.print(Panel.fit(
        f"[bold]SmartCityAI — NLP Classifier Training[/bold]\n"
        f"Model    : distilbert-base-uncased\n"
        f"Epochs   : {config['num_epochs']}\n"
        f"LR       : {config['learning_rate']}\n"
        f"Batch    : {config['train_batch_size']}\n"
        f"Frozen   : {config['freeze_layers']} layers\n"
        f"Mode     : {'FAST (testing)' if args.fast else 'FULL'}",
        border_style="green",
    ))

    # Import here (after sys.path is set)
    from app.nlp.trainer import train

    logger.info("Starting training...")
    run_id = train(config)

    console.print(Panel.fit(
        f"[bold green]Training complete![/bold green]\n"
        f"MLflow Run ID : {run_id}\n\n"
        f"View results:\n"
        f"  [bold]cd backend && mlflow ui --port 5000[/bold]\n"
        f"  Then open: http://localhost:5000\n\n"
        f"Test the model:\n"
        f"  [bold]uvicorn main:app --reload[/bold]\n"
        f"  POST http://localhost:8000/predict/text",
        border_style="green",
    ))


if __name__ == "__main__":
    main()