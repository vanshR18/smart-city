"""
mlops_router.py
───────────────
API endpoints for MLOps operations.
Lets you trigger retraining, inspect model registry, and view experiments
without touching the command line — important for demo situations.

Endpoints:
  POST /mlops/retrain          ← trigger retraining manually
  GET  /mlops/model-registry   ← current model versions and metrics
  GET  /mlops/experiments      ← recent MLflow runs
  GET  /mlops/scheduler-status ← when next scheduled retrain fires
  POST /mlops/promote          ← force-promote a specific MLflow run
"""

import json
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from loguru import logger

router = APIRouter(prefix="/mlops", tags=["MLOps"])

BACKEND_DIR  = Path(__file__).parent.parent.parent
MODELS_DIR   = BACKEND_DIR / "app" / "nlp" / "models_store"
MLFLOW_DIR   = BACKEND_DIR / "mlruns"


# ── Schemas ───────────────────────────────────────────────────────────────────
class RetrainRequest(BaseModel):
    force:              bool  = False   # skip F1 comparison gate
    num_epochs:         int   = 3
    min_f1_improvement: float = 0.01


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/retrain")
async def trigger_retrain(
    body: RetrainRequest,
    background_tasks: BackgroundTasks,
):
    """
    Manually triggers the NLP retraining pipeline in a background task.
    Returns immediately — check /mlops/model-registry for results.

    Set force=true to skip the F1 comparison gate and always promote.
    """
    from app.mlops.retrain import run_retraining_pipeline

    logger.info(f"Manual retrain triggered via API (force={body.force})")

    # Run in background so the HTTP response returns immediately
    background_tasks.add_task(
        run_retraining_pipeline,
        force=body.force,
        min_f1_improvement=body.min_f1_improvement,
        num_epochs=body.num_epochs,
    )

    return {
        "status":  "started",
        "message": "Retraining pipeline started in background",
        "config":  body.model_dump(),
        "check":   "GET /mlops/model-registry for results",
    }


@router.get("/model-registry")
def model_registry():
    """
    Returns current model version, metrics, and training history.
    This is the single source of truth for what's deployed right now.
    """
    from app.nlp.inference import get_model_info

    registry = {
        "inference_status": get_model_info(),
        "best_model_path":  str(MODELS_DIR / "best_model"),
        "best_model_exists": (MODELS_DIR / "best_model").exists(),
        "backup_exists":     (MODELS_DIR / "previous_model").exists(),
        "latest_metrics":    None,
        "training_config":   None,
        "timestamp":         datetime.utcnow().isoformat(),
    }

    # Read latest metrics
    metrics_file = MODELS_DIR / "latest_metrics.json"
    if metrics_file.exists():
        try:
            registry["latest_metrics"] = json.loads(metrics_file.read_text())
        except Exception:
            pass

    # Read training config
    config_file = MODELS_DIR / "training_config.json"
    if config_file.exists():
        try:
            registry["training_config"] = json.loads(config_file.read_text())
        except Exception:
            pass

    return registry


@router.get("/experiments")
def list_experiments(limit: int = Query(default=10, le=50)):
    """
    Returns recent MLflow experiment runs.
    Provides a lightweight view without the full MLflow UI.
    Open http://localhost:5000 for the full interactive dashboard.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
        client = mlflow.tracking.MlflowClient()

        experiments = client.search_experiments()
        all_runs    = []

        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=limit,
                order_by=["start_time DESC"],
            )
            for run in runs:
                all_runs.append({
                    "run_id":      run.info.run_id[:8],
                    "experiment":  exp.name,
                    "status":      run.info.status,
                    "started_at":  datetime.fromtimestamp(
                        run.info.start_time / 1000
                    ).isoformat() if run.info.start_time else None,
                    "metrics":     {
                        k: round(v, 4)
                        for k, v in run.data.metrics.items()
                    },
                    "params": {
                        k: v for k, v in run.data.params.items()
                        if k in ["num_epochs", "learning_rate", "model_name",
                                 "freeze_layers", "trigger", "decision"]
                    },
                    "tags": dict(run.data.tags),
                })

        # Sort by start time, newest first
        all_runs.sort(key=lambda r: r["started_at"] or "", reverse=True)

        return {
            "experiments": [e.name for e in experiments],
            "runs":        all_runs[:limit],
            "mlflow_ui":   "http://localhost:5000",
        }

    except Exception as e:
        return {
            "error":     str(e),
            "message":   "MLflow not available or no runs yet",
            "mlflow_ui": "http://localhost:5000",
        }


@router.get("/scheduler-status")
def scheduler_status():
    """Returns when the next scheduled retraining run will fire."""
    try:
        from app.mlops.retrain import start_retraining_scheduler
        # Access the module-level scheduler if it was started
        import app.mlops.retrain as retrain_module
        sched = getattr(retrain_module, "_scheduler", None)

        if sched is None:
            return {
                "scheduler_running": False,
                "message": "Scheduler not started (APScheduler not installed or disabled)",
            }

        jobs = []
        for job in sched.get_jobs():
            jobs.append({
                "id":       job.id,
                "name":     job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            })

        return {
            "scheduler_running": sched.running,
            "jobs": jobs,
        }

    except Exception as e:
        return {"scheduler_running": False, "error": str(e)}


@router.get("/data-drift")
def data_drift_report():
    """
    Compares recent DB event type distribution vs training data distribution.
    If they diverge significantly, it signals the model needs retraining.
    A simplified drift detector — real production would use Evidently AI.
    """
    try:
        from app.database import SessionLocal
        from app.models.events import Incident
        from collections import Counter
        from datetime import timedelta

        db     = SessionLocal()
        since  = datetime.utcnow() - timedelta(days=7)
        recent = db.query(Incident).filter(
            Incident.occurred_at >= since
        ).all()
        db.close()

        if not recent:
            return {"status": "no_data", "message": "No recent events in DB"}

        recent_dist = Counter(r.event_type.value for r in recent)
        total       = sum(recent_dist.values())
        recent_pct  = {k: round(v / total * 100, 1) for k, v in recent_dist.items()}

        # Expected distribution from training data (approximate)
        expected_pct = {
            "ACCIDENT": 30, "FIRE": 15, "FLOOD": 8,
            "CRIME": 20, "CROWD": 12, "MEDICAL": 10, "NORMAL": 5,
        }

        drift = {}
        for etype, expected in expected_pct.items():
            actual = recent_pct.get(etype, 0)
            delta  = actual - expected
            drift[etype] = {
                "expected_pct": expected,
                "actual_pct":   actual,
                "delta":        round(delta, 1),
                "drifted":      abs(delta) > 10,  # >10% shift = significant
            }

        any_drift = any(d["drifted"] for d in drift.values())

        return {
            "period":           "last 7 days",
            "total_events":     total,
            "drift_detected":   any_drift,
            "recommendation":   "Consider retraining" if any_drift else "Model distribution looks healthy",
            "distribution":     drift,
        }

    except Exception as e:
        return {"error": str(e), "status": "check_failed"}