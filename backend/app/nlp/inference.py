"""
inference.py
────────────
Production inference wrapper for the trained DistilBERT model.

Design decisions:
- Model loaded once on first call, then cached in memory (no reload per request)
- Returns structured NLPResult with class label + per-class probabilities + urgency score
- Urgency score (0–1 float) is the probability of the predicted class → used by Risk Engine
- Falls back to rule-based classifier if no trained model exists yet (Phase 1 compat)
- Batch inference supported for processing multiple texts at once efficiently
"""

import json
import re
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from app.nlp.dataset import LABEL2ID, ID2LABEL, NUM_LABELS, MAX_LENGTH, MODEL_CHECKPOINT

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = Path(__file__).parent / "models_store"
BEST_MODEL  = MODELS_DIR / "best_model"


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class NLPResult:
    """
    Structured output from the NLP classifier.
    This is what gets stored in RiskPrediction.explanation["nlp_result"].
    """
    text:          str
    label:         str               # e.g. "FIRE"
    label_id:      int               # e.g. 1
    confidence:    float             # probability of predicted class (0–1)
    urgency_score: float             # same as confidence; alias for Risk Engine
    probabilities: dict              # {label_name: probability} for all 7 classes
    is_emergency:  bool              # False only for NORMAL
    model_version: str = "unknown"
    method:        str = "distilbert"  # "distilbert" or "rule_based"


# ── Model cache ───────────────────────────────────────────────────────────────
# Stored at module level — survives across FastAPI requests within one process
_model:     Optional[DistilBertForSequenceClassification] = None
_tokenizer: Optional[DistilBertTokenizerFast]             = None
_device:    Optional[torch.device]                        = None
_version:   str                                           = "not_loaded"


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: Optional[Path] = None) -> bool:
    """
    Loads the trained model into memory.
    Returns True if successful, False if no model exists yet.
    """
    global _model, _tokenizer, _device, _version

    path = model_path or BEST_MODEL

    if not path.exists():
        logger.warning(
            f"Trained model not found at {path}. "
            "Falling back to rule-based classifier. "
            "Run: python scripts/train_nlp.py"
        )
        return False

    logger.info(f"Loading NLP model from {path}...")
    _device    = _get_device()
    _tokenizer = DistilBertTokenizerFast.from_pretrained(str(path))
    _model     = DistilBertForSequenceClassification.from_pretrained(str(path))
    _model.to(_device)
    _model.eval()   # disable dropout for inference

    # Read version from metrics file
    metrics_file = MODELS_DIR / "latest_metrics.json"
    if metrics_file.exists():
        metrics  = json.loads(metrics_file.read_text())
        _version = f"run:{metrics.get('run_id','?')[:8]} | f1:{metrics.get('test_f1_macro',0):.3f}"
    else:
        _version = "trained"

    logger.success(f"NLP model loaded on {_device} | version: {_version}")
    return True


def is_model_loaded() -> bool:
    return _model is not None


# ── Rule-based fallback ───────────────────────────────────────────────────────
# Used when model hasn't been trained yet.
# Keyword matching — simple but interpretable.

_RULES = {
    "FIRE":     r"\b(fire|aag|burn|flame|blaze|smoke|inferno|explosion|blast)\b",
    "FLOOD":    r"\b(flood|paani|baarish|waterlog|drown|overflow|rain|submerge)\b",
    "ACCIDENT": r"\b(accident|crash|collision|hurt|injured|ambulance|emergency|overturned|hit)\b",
    "CRIME":    r"\b(crime|robbery|theft|chori|loot|murder|attack|kidnap|shoot|stab)\b",
    "CROWD":    r"\b(crowd|stampede|bheed|protest|rally|riot|mob)\b",
    "MEDICAL":  r"\b(medical|hospital|ambulance|collapsed|heart|faint|overdose|sick|injured)\b",
}

def _rule_based_classify(text: str) -> NLPResult:
    """Keyword-based fallback classifier. Used before model is trained."""
    text_lower = text.lower()
    scores     = {}
    for label, pattern in _RULES.items():
        matches = len(re.findall(pattern, text_lower))
        scores[label] = min(matches * 0.25, 0.9)   # cap at 0.9

    if not any(scores.values()):
        label, confidence = "NORMAL", 0.85
    else:
        label      = max(scores, key=scores.get)
        confidence = scores[label]

    # Build uniform probability dict (rule-based can't give true probs)
    probs = {l: 0.02 for l in ID2LABEL.values()}
    probs[label] = confidence
    # Normalize
    total = sum(probs.values())
    probs = {k: round(v / total, 4) for k, v in probs.items()}

    return NLPResult(
        text=text, label=label, label_id=LABEL2ID[label],
        confidence=confidence, urgency_score=confidence,
        probabilities=probs,
        is_emergency=(label != "NORMAL"),
        model_version="rule_based_v1",
        method="rule_based",
    )


# ── Main inference function ───────────────────────────────────────────────────
def predict(text: str) -> NLPResult:
    """
    Classifies a single text and returns a structured NLPResult.

    Auto-loads model on first call. Falls back to rule-based if not trained.
    """
    global _model

    # Try to load model if not already loaded
    if _model is None:
        loaded = load_model()
        if not loaded:
            return _rule_based_classify(text)

    # Tokenize
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = _model(**inputs)
        logits  = outputs.logits                          # shape: [1, NUM_LABELS]
        probs   = torch.softmax(logits, dim=-1)[0]        # shape: [NUM_LABELS]

    probs_np    = probs.cpu().numpy()
    pred_id     = int(np.argmax(probs_np))
    pred_label  = ID2LABEL[pred_id]
    confidence  = float(probs_np[pred_id])

    prob_dict = {
        ID2LABEL[i]: round(float(probs_np[i]), 4)
        for i in range(NUM_LABELS)
    }

    return NLPResult(
        text=text,
        label=pred_label,
        label_id=pred_id,
        confidence=round(confidence, 4),
        urgency_score=round(confidence, 4),
        probabilities=prob_dict,
        is_emergency=(pred_label != "NORMAL"),
        model_version=_version,
        method="distilbert",
    )


def predict_batch(texts: list[str]) -> list[NLPResult]:
    """
    Classifies multiple texts in one forward pass (more efficient than looping).
    Falls back to looped rule-based if model not loaded.
    """
    global _model

    if _model is None:
        loaded = load_model()
        if not loaded:
            return [_rule_based_classify(t) for t in texts]

    # Batch tokenize
    inputs = _tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,       # pad all to same length within batch
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()  # [batch, NUM_LABELS]

    results = []
    for i, text in enumerate(texts):
        pred_id    = int(np.argmax(probs[i]))
        pred_label = ID2LABEL[pred_id]
        confidence = float(probs[i][pred_id])
        prob_dict  = {ID2LABEL[j]: round(float(probs[i][j]), 4) for j in range(NUM_LABELS)}

        results.append(NLPResult(
            text=text,
            label=pred_label,
            label_id=pred_id,
            confidence=round(confidence, 4),
            urgency_score=round(confidence, 4),
            probabilities=prob_dict,
            is_emergency=(pred_label != "NORMAL"),
            model_version=_version,
            method="distilbert",
        ))

    return results


def get_model_info() -> dict:
    """Returns current model status — used by /health endpoint."""
    return {
        "model_loaded":  _model is not None,
        "model_version": _version,
        "device":        str(_device) if _device else "none",
        "model_path":    str(BEST_MODEL),
        "model_exists":  BEST_MODEL.exists(),
        "method":        "distilbert" if _model else "rule_based",
    }