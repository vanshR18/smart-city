"""
trainer.py
──────────
Fine-tunes DistilBERT for emergency text classification.
Every experiment is tracked in MLflow: hyperparams, metrics, model artifact.

Architecture:
  DistilBertForSequenceClassification
  └─ DistilBERT encoder (6 transformer layers, frozen or partially frozen)
  └─ Classification head (linear: 768 → NUM_LABELS)

Training strategy:
  - Freeze bottom 4 transformer layers → only top 2 layers + head are updated
  - This prevents catastrophic forgetting of pre-trained knowledge
  - Cuts training time by ~40% vs training all layers

MLflow tracks:
  - All hyperparameters (lr, epochs, batch_size, etc.)
  - Per-epoch: train_loss, val_loss, val_accuracy, val_f1
  - Best model saved to MLflow model registry as "emergency-nlp-classifier"
  - Confusion matrix + classification report as artifacts
"""

import os
import json
import time
import numpy as np
import mlflow
import mlflow.pytorch
from pathlib import Path
from datetime import datetime
from loguru import logger
from rich.console import Console
from rich.table import Table

import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from app.nlp.dataset import (
    get_tokenizer, get_data_collator, load_datasets,
    LABEL2ID, ID2LABEL, NUM_LABELS, MODEL_CHECKPOINT,
)

console = Console()

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR   = Path(__file__).parent / "models_store"
MLFLOW_DIR   = Path(__file__).parent.parent.parent / "mlruns"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Default hyperparameters ───────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "model_checkpoint":  MODEL_CHECKPOINT,
    "num_labels":        NUM_LABELS,
    "learning_rate":     2e-5,      # sweet spot for BERT fine-tuning (1e-5 to 5e-5)
    "num_epochs":        5,
    "train_batch_size":  16,
    "eval_batch_size":   32,
    "warmup_ratio":      0.1,       # 10% of steps used for LR warmup
    "weight_decay":      0.01,      # L2 regularisation (prevents overfitting)
    "freeze_layers":     4,         # freeze bottom N transformer layers
    "max_grad_norm":     1.0,       # gradient clipping (prevents exploding gradients)
    "seed":              42,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")       # Apple Silicon
        logger.info("Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (training will be slow — consider Google Colab for GPU)")
    return device


def freeze_layers(model: DistilBertForSequenceClassification, n: int):
    """
    Freezes the bottom N transformer layers of DistilBERT.
    DistilBERT has 6 layers total (indexed 0–5).
    Freezing bottom layers = preserve general language knowledge,
    only fine-tune top layers for our specific task.
    """
    frozen = 0
    for i, layer in enumerate(model.distilbert.transformer.layer):
        if i < n:
            for param in layer.parameters():
                param.requires_grad = False
            frozen += 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Frozen {frozen} layers | Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


def evaluate(model, dataloader, device) -> dict:
    """
    Runs model on a DataLoader and returns accuracy, macro-F1, and per-class F1.
    Sets model to eval mode (disables dropout) before inference.
    """
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    return {
        "loss":        total_loss / len(dataloader),
        "accuracy":    accuracy_score(all_labels, all_preds),
        "f1_macro":    f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "report":      classification_report(all_labels, all_preds, target_names=label_names),
        "confusion":   confusion_matrix(all_labels, all_preds).tolist(),
        "preds":       all_preds,
        "labels":      all_labels,
    }


# ── Main training function ────────────────────────────────────────────────────
def train(config: dict = None) -> str:
    """
    Fine-tunes DistilBERT and logs everything to MLflow.

    Args:
        config: Override any DEFAULT_CONFIG keys (for hyperparameter search).

    Returns:
        The MLflow run_id (use this to load the model later).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg["seed"])
    device = get_device()

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("emergency-nlp-classifier")

    with mlflow.start_run(run_name=f"distilbert-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # Log every hyperparameter
        mlflow.log_params(cfg)

        # ── Data ──────────────────────────────────────────────────────────────
        logger.info("Loading datasets...")
        train_ds, val_ds, test_ds = load_datasets()
        collator = get_data_collator()

        train_loader = DataLoader(
            train_ds, batch_size=cfg["train_batch_size"],
            shuffle=True, collate_fn=collator, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["eval_batch_size"],
            shuffle=False, collate_fn=collator, num_workers=0,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg["eval_batch_size"],
            shuffle=False, collate_fn=collator, num_workers=0,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        logger.info(f"Loading model: {cfg['model_checkpoint']}")
        model = DistilBertForSequenceClassification.from_pretrained(
            cfg["model_checkpoint"],
            num_labels=cfg["num_labels"],
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        freeze_layers(model, cfg["freeze_layers"])
        model.to(device)

        # ── Optimizer + scheduler ─────────────────────────────────────────────
        # AdamW with weight decay on all params except biases and LayerNorm
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_groups = [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": cfg["weight_decay"]},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_groups, lr=cfg["learning_rate"])

        total_steps  = len(train_loader) * cfg["num_epochs"]
        warmup_steps = int(total_steps * cfg["warmup_ratio"])
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        mlflow.log_params({"total_steps": total_steps, "warmup_steps": warmup_steps})
        logger.info(f"Training: {total_steps} steps | {warmup_steps} warmup steps")

        # ── Training loop ─────────────────────────────────────────────────────
        best_val_f1     = 0.0
        best_model_path = MODELS_DIR / "best_model"

        for epoch in range(cfg["num_epochs"]):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0

            for step, batch in enumerate(train_loader):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # Backprop
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                train_loss += loss.item()

                # Print progress every 10 steps
                if (step + 1) % 10 == 0:
                    avg = train_loss / (step + 1)
                    lr  = scheduler.get_last_lr()[0]
                    console.print(
                        f"  Epoch {epoch+1}/{cfg['num_epochs']} "
                        f"step {step+1}/{len(train_loader)} "
                        f"loss={avg:.4f} lr={lr:.2e}",
                        style="dim",
                    )

            # ── Validation ────────────────────────────────────────────────────
            avg_train_loss = train_loss / len(train_loader)
            val_metrics    = evaluate(model, val_loader, device)
            epoch_time     = time.time() - epoch_start

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss":   avg_train_loss,
                "val_loss":     val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

            # Pretty print epoch summary
            table = Table(title=f"Epoch {epoch+1} Results", border_style="dim")
            table.add_column("Metric");  table.add_column("Value", justify="right")
            table.add_row("Train Loss",     f"{avg_train_loss:.4f}")
            table.add_row("Val Loss",       f"{val_metrics['loss']:.4f}")
            table.add_row("Val Accuracy",   f"{val_metrics['accuracy']*100:.2f}%")
            table.add_row("Val F1 (macro)", f"{val_metrics['f1_macro']:.4f}")
            table.add_row("Time",           f"{epoch_time:.0f}s")
            console.print(table)

            # Save best model (based on val F1, more robust than accuracy)
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                model.save_pretrained(best_model_path)
                get_tokenizer().save_pretrained(best_model_path)
                logger.success(f"New best model saved (F1={best_val_f1:.4f})")
                mlflow.log_metric("best_val_f1", best_val_f1, step=epoch)

        # ── Final test evaluation ─────────────────────────────────────────────
        logger.info("Running final evaluation on test set...")
        best_model = DistilBertForSequenceClassification.from_pretrained(best_model_path)
        best_model.to(device)
        test_metrics = evaluate(best_model, test_loader, device)

        mlflow.log_metrics({
            "test_accuracy":   test_metrics["accuracy"],
            "test_f1_macro":   test_metrics["f1_macro"],
            "test_f1_weighted": test_metrics["f1_weighted"],
        })

        # Log classification report as artifact
        report_path = MODELS_DIR / "classification_report.txt"
        report_path.write_text(test_metrics["report"])
        mlflow.log_artifact(str(report_path))

        # Log confusion matrix as JSON artifact
        confusion_path = MODELS_DIR / "confusion_matrix.json"
        confusion_path.write_text(json.dumps({
            "matrix": test_metrics["confusion"],
            "labels": [ID2LABEL[i] for i in range(NUM_LABELS)],
        }, indent=2))
        mlflow.log_artifact(str(confusion_path))

        # Log model config
        config_path = MODELS_DIR / "training_config.json"
        config_path.write_text(json.dumps(cfg, indent=2))
        mlflow.log_artifact(str(config_path))

        # ── Register model in MLflow Model Registry ───────────────────────────
        model_uri = f"runs:/{run_id}/model"
        mlflow.pytorch.log_model(
            best_model,
            artifact_path="model",
            registered_model_name="emergency-nlp-classifier",
        )

        console.print(f"\n[bold green]Training complete![/bold green]")
        console.print(f"  Test Accuracy : {test_metrics['accuracy']*100:.2f}%")
        console.print(f"  Test F1 Macro : {test_metrics['f1_macro']:.4f}")
        console.print(f"  MLflow Run ID : {run_id}")
        console.print(f"\n[dim]Classification Report:[/dim]")
        console.print(test_metrics["report"])

        # Save run_id to disk so inference.py can load the right model
        (MODELS_DIR / "latest_run_id.txt").write_text(run_id)
        (MODELS_DIR / "latest_metrics.json").write_text(json.dumps({
            "run_id":        run_id,
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "trained_at":    datetime.utcnow().isoformat(),
        }, indent=2))

        return run_id