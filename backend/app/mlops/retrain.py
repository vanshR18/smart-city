"""
retrain.py
──────────
Automated MLOps retraining pipeline.

What this does:
  1. Pulls recent events from PostgreSQL as new training signal
  2. Rebuilds the dataset (recent data mixed with synthetic baseline)
  3. Retrains DistilBERT NLP model with MLflow experiment tracking
  4. Compares new model F1 against current production model F1
  5. Only promotes new model if it beats production by MIN_F1_IMPROVEMENT
  6. Reloads the live inference engine without restarting the server
  7. Logs a full audit trail to MLflow

Why automated retraining matters (what to say in interviews):
  Real emergency data distributions shift over time — Lucknow gets new
  neighbourhoods, new incident patterns emerge (e.g. post-flood crime spike).
  A model trained on last year's data degrades silently. This pipeline detects
  that drift and self-heals without human intervention.

Promotion logic:
  new_f1 >= current_f1 + MIN_F1_IMPROVEMENT  →  promote, reload inference
  new_f1 < current_f1 + MIN_F1_IMPROVEMENT   →  reject, keep current model
  No current model exists                     →  always promote

MLflow tracks every run, so you can see the full model lineage
in the MLflow UI (http://localhost:5000).
"""

import json
import shutil
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from typing import Optional

import mlflow
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR  = Path(__file__).parent.parent.parent   # backend/
MLOPS_DIR    = Path(__file__).parent
MODELS_DIR   = BACKEND_DIR / "app" / "nlp" / "models_store"
BEST_MODEL   = MODELS_DIR / "best_model"
BACKUP_MODEL = MODELS_DIR / "previous_model"
METRICS_FILE = MODELS_DIR / "latest_metrics.json"
MLFLOW_DIR   = BACKEND_DIR / "mlruns"

# ── Config ────────────────────────────────────────────────────────────────────
MIN_F1_IMPROVEMENT   = 0.01    # new model must beat current by at least 1%
MIN_TRAIN_SAMPLES    = 100     # don't retrain if dataset is too small
RECENT_DAYS_WINDOW   = 30      # pull events from last N days
SYNTHETIC_AUGMENT    = True    # always mix synthetic data with real


def get_current_model_f1() -> Optional[float]:
    """
    Reads the F1 score of the currently deployed model from disk.
    Returns None if no model has been trained yet.
    """
    if not METRICS_FILE.exists():
        return None
    try:
        metrics = json.loads(METRICS_FILE.read_text())
        f1 = metrics.get("test_f1_macro")
        logger.info(f"Current model F1: {f1:.4f}" if f1 else "No current model F1 found")
        return float(f1) if f1 else None
    except Exception as e:
        logger.warning(f"Could not read current model metrics: {e}")
        return None


def fetch_recent_events_from_db(days: int = RECENT_DAYS_WINDOW) -> list[dict]:
    """
    Pulls recent incident events from PostgreSQL.
    These become the "new signal" that updates the model's understanding.
    Falls back to empty list gracefully if DB is not available.
    """
    try:
        from app.database import SessionLocal
        from app.models.events import Incident

        db    = SessionLocal()
        since = datetime.utcnow() - timedelta(days=days)

        rows = db.query(Incident).filter(
            Incident.occurred_at >= since
        ).all()
        db.close()

        events = [
            {
                "text":       row.raw_input or f"{row.event_type.value} incident at {row.area_name}",
                "label_name": row.event_type.value,
            }
            for row in rows
            if row.raw_input  # only use events that have text (NLP signal)
        ]

        logger.info(f"Fetched {len(events)} recent events from DB (last {days} days)")
        return events

    except Exception as e:
        logger.warning(f"DB fetch failed: {e}. Will use synthetic data only.")
        return []


def build_retrain_dataset(db_events: list[dict]) -> Optional[pd.DataFrame]:
    """
    Builds the training dataset by combining:
      1. DB events (real, recent)
      2. Synthetic Lucknow sentences (stable baseline)

    Returns None if total samples below MIN_TRAIN_SAMPLES.
    """
    import sys
    sys.path.insert(0, str(BACKEND_DIR))

    from app.nlp.dataset import LABEL2ID

    rows = []

    # Real DB events
    for event in db_events:
        label_name = event.get("label_name", "NORMAL")
        if label_name in LABEL2ID:
            rows.append({
                "text":       event["text"],
                "label":      LABEL2ID[label_name],
                "label_name": label_name,
                "source":     "db",
            })

    # Synthetic baseline (always include for stability)
    if SYNTHETIC_AUGMENT:
        try:
            from scripts.download_dataset import SYNTHETIC_DATA, LABEL2ID as DL_LABEL2ID
            for label_name, sentences in SYNTHETIC_DATA.items():
                lid = LABEL2ID.get(label_name)
                if lid is None:
                    continue
                for text in sentences:
                    rows.append({
                        "text":       text,
                        "label":      lid,
                        "label_name": label_name,
                        "source":     "synthetic",
                    })
            logger.info(f"Added synthetic baseline: {sum(len(v) for v in SYNTHETIC_DATA.values())} sentences")
        except Exception as e:
            logger.warning(f"Could not load synthetic data: {e}")

    if len(rows) < MIN_TRAIN_SAMPLES:
        logger.warning(
            f"Only {len(rows)} samples — minimum is {MIN_TRAIN_SAMPLES}. "
            "Skipping retraining."
        )
        return None

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Retrain dataset: {len(df)} samples "
                f"({sum(df.source=='db')} real, {sum(df.source=='synthetic')} synthetic)")
    return df


def save_dataset_splits(df: pd.DataFrame) -> bool:
    """Saves train/val/test CSVs to app/nlp/data/ for the trainer to read."""
    data_dir = BACKEND_DIR / "app" / "nlp" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n         = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    df.iloc[:train_end].to_csv(data_dir / "train.csv", index=False)
    df.iloc[train_end:val_end].to_csv(data_dir / "val.csv",   index=False)
    df.iloc[val_end:].to_csv(data_dir / "test.csv",  index=False)

    logger.info(f"Saved splits → train:{train_end} val:{val_end-train_end} "
                f"test:{n-val_end}")
    return True


def backup_current_model():
    """Backs up current best model before overwriting with new one."""
    if BEST_MODEL.exists():
        if BACKUP_MODEL.exists():
            shutil.rmtree(BACKUP_MODEL)
        shutil.copytree(BEST_MODEL, BACKUP_MODEL)
        logger.info(f"Current model backed up to {BACKUP_MODEL}")


def restore_backup_model():
    """Restores backup model if new model was rejected or failed."""
    if BACKUP_MODEL.exists():
        if BEST_MODEL.exists():
            shutil.rmtree(BEST_MODEL)
        shutil.copytree(BACKUP_MODEL, BEST_MODEL)
        logger.info("Backup model restored after rejection")


def reload_inference_engine():
    """
    Hot-reloads the NLP inference engine with the new model weights.
    No server restart required — model is swapped in memory.
    """
    try:
        from app.nlp import inference
        # Reset cached model so next request triggers fresh load
        inference._model     = None
        inference._tokenizer = None
        inference._version   = "reloading"
        # Trigger immediate reload
        success = inference.load_model()
        if success:
            logger.success("Inference engine hot-reloaded with new model")
        else:
            logger.error("Hot-reload failed — inference engine will use rule-based fallback")
        return success
    except Exception as e:
        logger.error(f"Hot-reload error: {e}")
        return False


def run_retraining_pipeline(
    force:             bool  = False,
    min_f1_improvement: float = MIN_F1_IMPROVEMENT,
    num_epochs:        int   = 3,
) -> dict:
    """
    Full retraining pipeline. Called by:
      - The APScheduler cron job (every RETRAIN_INTERVAL_HOURS)
      - The /mlops/retrain endpoint (manual trigger via API)
      - scripts/retrain_cron.py (command line)

    Args:
        force:              skip F1 comparison and always promote
        min_f1_improvement: minimum delta to promote new model
        num_epochs:         training epochs (fewer = faster, less accurate)

    Returns:
        dict with outcome, metrics, and what action was taken
    """
    started_at = datetime.utcnow()
    logger.info("=" * 60)
    logger.info("MLOps Retraining Pipeline — starting")
    logger.info(f"  force={force} | epochs={num_epochs} | min_delta={min_f1_improvement}")
    logger.info("=" * 60)

    result = {
        "started_at":     started_at.isoformat(),
        "status":         "unknown",
        "promoted":       False,
        "current_f1":     None,
        "new_f1":         None,
        "f1_delta":       None,
        "reason":         "",
        "run_id":         None,
        "completed_at":   None,
        "duration_sec":   None,
    }

    try:
        # ── 1. Get current model baseline ─────────────────────────────────────
        current_f1 = get_current_model_f1()
        result["current_f1"] = current_f1

        # ── 2. Fetch recent data ───────────────────────────────────────────────
        db_events = fetch_recent_events_from_db()
        df        = build_retrain_dataset(db_events)

        if df is None:
            result["status"] = "skipped"
            result["reason"] = f"Insufficient data (<{MIN_TRAIN_SAMPLES} samples)"
            return result

        # ── 3. Save dataset splits ─────────────────────────────────────────────
        save_dataset_splits(df)

        # ── 4. Train new model ─────────────────────────────────────────────────
        logger.info("Training new model...")
        backup_current_model()

        from app.nlp.trainer import train
        run_id = train(config={
            "num_epochs":       num_epochs,
            "learning_rate":    2e-5,
            "train_batch_size": 16,
            "freeze_layers":    4,
        })
        result["run_id"] = run_id

        # ── 5. Read new model metrics ──────────────────────────────────────────
        new_metrics = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}
        new_f1      = new_metrics.get("test_f1_macro")

        if new_f1 is None:
            raise ValueError("Training completed but no F1 metric found in metrics file")

        result["new_f1"] = float(new_f1)

        # ── 6. Compare and decide ──────────────────────────────────────────────
        if current_f1 is None:
            # No existing model — always promote
            promote  = True
            f1_delta = None
            reason   = "No existing model — first deployment"

        elif force:
            promote  = True
            f1_delta = float(new_f1) - float(current_f1)
            reason   = f"Force flag set (delta={f1_delta:+.4f})"

        else:
            f1_delta = float(new_f1) - float(current_f1)
            promote  = f1_delta >= min_f1_improvement
            reason   = (
                f"New model F1 {new_f1:.4f} beats current {current_f1:.4f} "
                f"by {f1_delta:+.4f} (threshold: +{min_f1_improvement})"
                if promote else
                f"New model F1 {new_f1:.4f} does NOT beat current {current_f1:.4f} "
                f"by required {min_f1_improvement} (delta={f1_delta:+.4f}) — rejected"
            )

        result["f1_delta"] = f1_delta
        result["promoted"] = promote
        result["reason"]   = reason

        if promote:
            logger.success(f"Model promoted: {reason}")
            reload_inference_engine()
            result["status"] = "promoted"
        else:
            logger.warning(f"Model rejected: {reason}")
            restore_backup_model()
            result["status"] = "rejected"

        # ── 7. Log promotion decision to MLflow ────────────────────────────────
        mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
        mlflow.set_experiment("emergency-nlp-classifier")

        with mlflow.start_run(run_name=f"retrain-decision-{datetime.now().strftime('%Y%m%d-%H%M')}"):
            mlflow.log_params({
                "trigger":           "scheduled" if not force else "manual",
                "force":             force,
                "min_f1_improvement": min_f1_improvement,
                "num_epochs":        num_epochs,
                "db_events_used":    len(db_events),
                "dataset_size":      len(df),
            })
            mlflow.log_metrics({
                "current_f1": current_f1 or 0.0,
                "new_f1":     float(new_f1),
                "f1_delta":   f1_delta or 0.0,
                "promoted":   1.0 if promote else 0.0,
            })
            mlflow.set_tag("decision", "PROMOTED" if promote else "REJECTED")
            mlflow.set_tag("reason",   reason[:250])

    except Exception as e:
        logger.error(f"Retraining pipeline failed: {e}")
        logger.error(traceback.format_exc())
        result["status"] = "failed"
        result["reason"] = str(e)
        # Restore backup if training crashed mid-way
        restore_backup_model()

    finally:
        completed_at          = datetime.utcnow()
        result["completed_at"] = completed_at.isoformat()
        result["duration_sec"] = round((completed_at - started_at).total_seconds(), 1)

        logger.info(f"Pipeline finished in {result['duration_sec']}s — status: {result['status']}")

    return result


# ── APScheduler setup (used in main.py lifespan) ──────────────────────────────
def start_retraining_scheduler(interval_hours: int = 24):
    """
    Starts a background APScheduler that calls run_retraining_pipeline()
    every `interval_hours` hours.

    Called from main.py lifespan on startup.
    Returns the scheduler object (keep a reference to prevent GC).

    APScheduler is non-blocking — it runs in a background thread and
    does not interfere with FastAPI's async event loop.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
        scheduler.add_job(
            func=run_retraining_pipeline,
            trigger="interval",
            hours=interval_hours,
            id="nlp_retrain",
            name="NLP Model Retraining",
            replace_existing=True,
            max_instances=1,        # never run two retraining jobs simultaneously
        )
        scheduler.start()
        logger.success(f"Retraining scheduler started — fires every {interval_hours}h")
        return scheduler

    except ImportError:
        logger.warning(
            "APScheduler not installed — auto-retraining disabled. "
            "pip install apscheduler to enable. "
            "Manual retraining still works via POST /mlops/retrain"
        )
        return None