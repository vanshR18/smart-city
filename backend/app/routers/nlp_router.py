"""
nlp_router.py
─────────────
FastAPI router for NLP inference endpoints.
Mounted at /predict in main.py.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

from app.nlp.inference import predict, predict_batch, get_model_info, load_model

router = APIRouter(prefix="/predict", tags=["NLP Inference"])


# ── Request / Response schemas ────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        example="Bada accident hua hai Hazratganj crossing pe, ambulance bulao",
    )
    include_probabilities: bool = Field(
        default=True,
        description="Include per-class probability scores in response",
    )


class BatchTextInput(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        example=[
            "Fire broke out in Kaiserbagh market",
            "Traffic moving normally on Hazratganj",
        ],
    )


class NLPResponse(BaseModel):
    text:          str
    label:         str
    confidence:    float
    urgency_score: float
    is_emergency:  bool
    probabilities: Optional[dict] = None
    model_version: str
    method:        str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/text", response_model=NLPResponse)
def predict_text(body: TextInput):
    """
    Classify a single emergency text.

    Returns the event type label, confidence score, and urgency score.
    The urgency_score (0–1) feeds directly into the Risk Scoring Engine.

    Example inputs:
    - "Bada accident hua hai Hazratganj crossing pe" → ACCIDENT
    - "Building mein aag lag gayi Kaiserbagh mein" → FIRE
    - "Traffic moving normally on Hazratganj road" → NORMAL
    """
    try:
        result = predict(body.text)
    except Exception as e:
        logger.error(f"NLP inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return NLPResponse(
        text=result.text,
        label=result.label,
        confidence=result.confidence,
        urgency_score=result.urgency_score,
        is_emergency=result.is_emergency,
        probabilities=result.probabilities if body.include_probabilities else None,
        model_version=result.model_version,
        method=result.method,
    )


@router.post("/text/batch")
def predict_text_batch(body: BatchTextInput):
    """
    Classify multiple texts in a single request.
    More efficient than calling /predict/text N times.
    Maximum 50 texts per request.
    """
    if len(body.texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch")

    try:
        results = predict_batch(body.texts)
    except Exception as e:
        logger.error(f"Batch NLP inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")

    return {
        "count": len(results),
        "results": [
            {
                "text":          r.text,
                "label":         r.label,
                "urgency_score": r.urgency_score,
                "is_emergency":  r.is_emergency,
                "confidence":    r.confidence,
            }
            for r in results
        ],
        "emergency_count": sum(1 for r in results if r.is_emergency),
    }


@router.get("/text/model-info")
def model_info():
    """
    Returns the current NLP model status.
    Useful for checking if the trained model is loaded or if
    the system is using the rule-based fallback.
    """
    return get_model_info()


@router.post("/text/reload-model")
def reload_model():
    """
    Forces a reload of the model from disk.
    Call this after retraining to pick up the new weights without restarting.
    """
    success = load_model()
    info    = get_model_info()
    return {
        "reloaded": success,
        "message":  "Model loaded successfully" if success else "No trained model found, using rule-based fallback",
        **info,
    }