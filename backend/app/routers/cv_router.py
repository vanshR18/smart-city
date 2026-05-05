"""
cv_router.py
────────────
Endpoints for computer vision emergency detection.
Supports image upload (bytes) and base64 encoded images.
Video endpoint accepts a server-side file path (for demo mode).
"""

import base64
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger

from app.cv.detector import (
    detect_image, detect_image_bytes_b64, detect_video_frames,
    get_cv_model_info, CVResult,
)
from app.risk_engine.scorer import score_from_cv_only, compute_risk_score

router = APIRouter(prefix="/predict", tags=["Computer Vision"])


# ── Response schema ───────────────────────────────────────────────────────────
class CVResponse(BaseModel):
    label:          str
    cv_score:       float
    is_emergency:   bool
    risk_score:     float
    risk_level:     str
    frame_count:    int
    processing_ms:  float
    model_version:  str
    method:         str
    detections:     list
    explanation:    dict


class B64ImageInput(BaseModel):
    image_b64:  str  = Field(..., description="Base64 encoded image")
    area_name:  str  = Field(default="Unknown", description="Location name")
    filename:   str  = Field(default="upload.jpg")


class VideoPathInput(BaseModel):
    video_path:     str   = Field(..., description="Server-side path to video file")
    area_name:      str   = Field(default="Unknown")
    sample_every_n: int   = Field(default=30, ge=1, le=120)
    max_frames:     int   = Field(default=10, ge=1, le=50)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/image", response_model=CVResponse)
async def predict_image(
    file:      UploadFile = File(..., description="Image file (JPEG/PNG)"),
    area_name: str        = Form(default="Unknown"),
):
    """
    Upload an image and get emergency detection results + risk score.

    Accepts: JPEG, PNG, WebP
    Returns: detected emergency type, bounding boxes, and risk score.

    Example use: upload a CCTV frame to check for accidents or fire.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected image, got {file.content_type}")

    try:
        image_bytes = await file.read()
        cv_result   = detect_image(image_bytes)
    except Exception as e:
        logger.error(f"Image detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    risk = score_from_cv_only(cv_result, area_name, datetime.utcnow())

    return _build_response(cv_result, risk)


@router.post("/image/base64", response_model=CVResponse)
def predict_image_b64(body: B64ImageInput):
    """
    Detect emergency from a base64-encoded image.
    Useful for frontend clients sending images without multipart form.
    """
    try:
        cv_result = detect_image_bytes_b64(body.image_b64)
    except Exception as e:
        logger.error(f"B64 image detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    risk = score_from_cv_only(cv_result, body.area_name, datetime.utcnow())
    return _build_response(cv_result, risk)


@router.post("/video", response_model=CVResponse)
def predict_video(body: VideoPathInput):
    """
    Analyze a video file and return aggregated emergency detection.

    Samples frames at the specified rate, runs YOLO on each,
    then aggregates using peak+average fusion.

    The video file must exist on the server. In a real deployment
    you'd upload it first to a storage bucket and pass the path.
    """
    try:
        cv_result = detect_video_frames(
            body.video_path,
            sample_every_n=body.sample_every_n,
            max_frames=body.max_frames,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Video not found: {body.video_path}")
    except Exception as e:
        logger.error(f"Video detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    risk = score_from_cv_only(cv_result, body.area_name, datetime.utcnow())
    return _build_response(cv_result, risk)


@router.get("/image/model-info")
def cv_model_info():
    """Returns YOLOv8 model status and what mode (real vs simulated) is active."""
    return get_cv_model_info()


# ── Helper ────────────────────────────────────────────────────────────────────
def _build_response(cv_result: CVResult, risk) -> CVResponse:
    return CVResponse(
        label=cv_result.label,
        cv_score=cv_result.cv_score,
        is_emergency=cv_result.is_emergency,
        risk_score=risk.risk_score,
        risk_level=risk.risk_level,
        frame_count=cv_result.frame_count,
        processing_ms=cv_result.processing_ms,
        model_version=cv_result.model_version,
        method=cv_result.method,
        detections=[
            {
                "class_name": d.class_name,
                "confidence": d.confidence,
                "emergency":  d.emergency,
                "bbox":       d.bbox,
            }
            for d in cv_result.detections
        ],
        explanation={**cv_result.explanation, **risk.explanation},
    )