"""
detector.py
───────────
YOLOv8-based emergency detector for images and video frames.

Architecture decision:
  We use YOLOv8n (nano) — smallest variant, ~3.2MB, runs on CPU in ~50ms/frame.
  For production with GPU you'd switch to YOLOv8m or YOLOv8l.

Two detection modes:
  1. OBJECT DETECTION — detects accident/fire/crowd bounding boxes in image
     Uses:  yolov8n.pt (pretrained COCO) for objects like cars, fire, people
     Maps COCO classes → our emergency categories

  2. IMAGE CLASSIFICATION — classifies whole image into emergency category
     Uses:  yolov8n-cls.pt (pretrained ImageNet classification)
     More reliable when detection boxes are not the goal

Why not fine-tune from scratch?
  Fine-tuning requires labeled video datasets (UCF-Crime, FireNet) which are
  large (50GB+) and need GPU. For this project we:
  a) Use pretrained weights + COCO class mapping (works immediately)
  b) Provide the full fine-tuning code so you can run it with a GPU later
  c) Add a simulation fallback so the pipeline works end-to-end now

The COCO → Emergency mapping logic is what makes this interesting in interviews.
"""

import os
import io
import base64
import random
import time
import numpy as np
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field
from loguru import logger

# ── lazy imports — only load when model is actually used 
_ultralytics_available = False
_cv2_available         = False

try:
    from ultralytics import YOLO
    _ultralytics_available = True
except ImportError:
    logger.warning("ultralytics not installed. CV will use simulation mode.")

try:
    import cv2
    _cv2_available = True
except ImportError:
    logger.warning("opencv not installed. Video frame extraction disabled.")

try:
    from PIL import Image
    _pil_available = True
except ImportError:
    _pil_available = False

# ── Paths 
CV_DIR     = Path(__file__).parent
MODELS_DIR = CV_DIR / "models_store"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── COCO class → Emergency type mapping 
# YOLOv8 pretrained on COCO detects 80 classes.
# We map relevant ones to our 7 emergency categories.
# Unmapped classes → NORMAL (background / no emergency).

COCO_TO_EMERGENCY = {
    # ACCIDENT indicators
    "car":          ("ACCIDENT", 0.45),   # (label, base_confidence_boost)
    "truck":        ("ACCIDENT", 0.40),
    "bus":          ("ACCIDENT", 0.38),
    "motorcycle":   ("ACCIDENT", 0.42),
    "bicycle":      ("ACCIDENT", 0.30),
    "traffic light":("ACCIDENT", 0.20),
    "stop sign":    ("ACCIDENT", 0.15),

    # FIRE indicators
    "fire":         ("FIRE",     0.90),   # not in COCO but added for fine-tuned model
    "smoke":        ("FIRE",     0.80),

    # CROWD / STAMPEDE indicators
    "person":       ("CROWD",    0.25),   # single person = low; many persons = high

    # MEDICAL indicators
    "bed":          ("MEDICAL",  0.35),
    "scissors":     ("MEDICAL",  0.25),

    # No direct COCO class for FLOOD/CRIME/NORMAL — handled by image-level analysis
}

# How many "person" detections trigger CROWD vs MEDICAL vs NORMAL
CROWD_PERSON_THRESHOLD = 5   # ≥5 persons in frame → flag as CROWD

# Emergency classes our fine-tuned model will predict (Phase 3 extension)
FINE_TUNED_CLASSES = {
    0: "ACCIDENT",
    1: "FIRE",
    2: "FLOOD",
    3: "CRIME",
    4: "CROWD",
    5: "MEDICAL",
    6: "NORMAL",
}


# ── Result dataclass 
@dataclass
class Detection:
    """Single object detected in an image."""
    class_name:  str
    confidence:  float
    bbox:        list[float]   # [x1, y1, x2, y2] normalized 0–1
    emergency:   str           # mapped emergency category
    area_ratio:  float = 0.0   # fraction of image the bbox covers


@dataclass
class CVResult:
    """
    Structured output from the CV detector.
    cv_score (0–1) feeds into the Risk Scoring Engine as the dominant signal.
    """
    label:          str             # PRIMARY emergency label for this image
    cv_score:       float           # 0–1 confidence (used by Risk Engine)
    is_emergency:   bool
    detections:     list[Detection] = field(default_factory=list)
    frame_count:    int  = 1        # >1 for video
    processing_ms:  float = 0.0
    model_version:  str  = "yolov8n-pretrained"
    method:         str  = "yolov8"  # "yolov8" | "simulated"
    explanation:    dict = field(default_factory=dict)


# ── Model cache 
_detect_model:  Optional["YOLO"] = None
_loaded_path:   str = ""


def _load_yolo(model_name: str = "yolov8n.pt") -> Optional["YOLO"]:
    """
    Loads YOLOv8 model. Downloads weights automatically on first call
    (cached in models_store/ after first download).
    Returns None if ultralytics is not installed.
    """
    global _detect_model, _loaded_path

    if not _ultralytics_available:
        return None

    model_path = MODELS_DIR / model_name

    if _detect_model is not None and _loaded_path == str(model_path):
        return _detect_model   # already loaded

    try:
        logger.info(f"Loading YOLOv8 model: {model_name} ...")
        # YOLO auto-downloads from ultralytics hub if not found locally
        _detect_model = YOLO(str(model_path) if model_path.exists() else model_name)
        # Copy weights to our models_store for reproducibility
        if not model_path.exists():
            _detect_model.save(str(model_path))
        _loaded_path = str(model_path)
        logger.success(f"YOLOv8 model loaded: {model_name}")
        return _detect_model
    except Exception as e:
        logger.error(f"Failed to load YOLOv8: {e}. Falling back to simulation.")
        return None


# ── Core detection logic 
def _map_detections_to_emergency(raw_detections: list) -> tuple[str, float, list[Detection]]:
    """
    Converts raw YOLO detections to our emergency schema.

    Logic:
      1. Count persons → if ≥ CROWD_PERSON_THRESHOLD → CROWD
      2. Check for direct emergency class hits (fire, smoke)
      3. Check for vehicle detections → ACCIDENT
      4. Default → NORMAL

    Returns: (label, cv_score, detections_list)
    """
    detections   = []
    class_counts = {}
    emergency_votes = {}  # emergency_type → [confidence_scores]

    for det in raw_detections:
        # det is a YOLO Results box object
        try:
            cls_id     = int(det.cls[0])
            cls_name   = det.names[cls_id] if hasattr(det, "names") else str(cls_id)
            conf       = float(det.conf[0])
            box        = det.xyxyn[0].tolist()   # normalized [x1,y1,x2,y2]
            area_ratio = (box[2] - box[0]) * (box[3] - box[1])
        except Exception:
            continue

        emergency, boost = COCO_TO_EMERGENCY.get(cls_name, ("NORMAL", 0.0))
        adjusted_conf    = min(conf + boost, 1.0)

        detection = Detection(
            class_name=cls_name,
            confidence=round(conf, 3),
            bbox=box,
            emergency=emergency,
            area_ratio=round(area_ratio, 4),
        )
        detections.append(detection)

        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        if emergency not in emergency_votes:
            emergency_votes[emergency] = []
        emergency_votes[emergency].append(adjusted_conf)

    if not detections:
        return "NORMAL", 0.05, []

    # Crowd override: many persons in frame
    person_count = class_counts.get("person", 0)
    if person_count >= CROWD_PERSON_THRESHOLD:
        crowd_score = min(0.3 + (person_count / 20), 0.95)
        return "CROWD", round(crowd_score, 3), detections

    # Pick emergency with highest mean confidence
    if emergency_votes:
        best_emergency = max(
            emergency_votes,
            key=lambda k: sum(emergency_votes[k]) / len(emergency_votes[k])
        )
        best_score = sum(emergency_votes[best_emergency]) / len(emergency_votes[best_emergency])

        if best_emergency != "NORMAL" and best_score > 0.25:
            return best_emergency, round(best_score, 3), detections

    return "NORMAL", 0.10, detections


# ── Simulation fallback 
# Used when ultralytics is not installed or model fails to load.
# Produces realistic-looking CV results for pipeline testing.

_SIM_SCENARIOS = [
    ("ACCIDENT", 0.82, [{"class_name": "car", "conf": 0.91},
                        {"class_name": "truck", "conf": 0.76}]),
    ("FIRE",     0.91, [{"class_name": "fire", "conf": 0.94}]),
    ("CROWD",    0.74, [{"class_name": "person", "conf": 0.88}] * 8),
    ("NORMAL",   0.08, []),
    ("MEDICAL",  0.65, [{"class_name": "person", "conf": 0.82}]),
    ("FLOOD",    0.70, []),   # COCO has no water class → simulation only
]

def _simulate_cv_result(filename: str = "") -> CVResult:
    """
    Generates a realistic simulated CV result.
    Biased toward emergency scenarios (more interesting for demos).
    """
    weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
    label, base_score, raw_dets = random.choices(_SIM_SCENARIOS, weights=weights, k=1)[0]

    # Add noise to score
    score = round(min(base_score + random.uniform(-0.08, 0.08), 1.0), 3)

    detections = [
        Detection(
            class_name=d["class_name"],
            confidence=round(d["conf"] + random.uniform(-0.05, 0.05), 3),
            bbox=[random.uniform(0.1, 0.4), random.uniform(0.1, 0.4),
                  random.uniform(0.5, 0.9), random.uniform(0.5, 0.9)],
            emergency=label,
            area_ratio=round(random.uniform(0.05, 0.35), 3),
        )
        for d in raw_dets
    ]

    return CVResult(
        label=label,
        cv_score=score,
        is_emergency=(label != "NORMAL"),
        detections=detections,
        processing_ms=round(random.uniform(30, 120), 1),
        model_version="simulated-v0",
        method="simulated",
        explanation={
            "note":       "Simulated — ultralytics not installed or model not loaded",
            "detections": len(detections),
            "top_class":  label,
        },
    )


# ── Public API 
def detect_image(
    image_input: Union[str, bytes, "np.ndarray"],
    confidence_threshold: float = 0.25,
) -> CVResult:
    """
    Runs emergency detection on a single image.

    Args:
        image_input: file path (str), raw bytes, or numpy array (BGR from OpenCV)
        confidence_threshold: minimum YOLO confidence to keep a detection

    Returns:
        CVResult with label, cv_score, and per-object detections
    """
    t_start = time.time()

    model = _load_yolo()
    if model is None:
        result        = _simulate_cv_result()
        result.processing_ms = round((time.time() - t_start) * 1000, 1)
        return result

    try:
        # Handle different input types
        if isinstance(image_input, bytes):
            # Convert bytes → numpy array via PIL
            from PIL import Image as PILImage
            import io
            pil_img    = PILImage.open(io.BytesIO(image_input)).convert("RGB")
            image_np   = np.array(pil_img)
        elif isinstance(image_input, str):
            image_np   = image_input   # YOLO accepts file paths
        else:
            image_np   = image_input   # numpy array

        results = model(image_np, conf=confidence_threshold, verbose=False)

        # results[0].boxes contains all detections
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            label, score, dets = "NORMAL", 0.05, []
        else:
            # Attach class names to boxes object (needed by mapper)
            boxes.names = results[0].names
            label, score, dets = _map_detections_to_emergency([boxes])

        processing_ms = round((time.time() - t_start) * 1000, 1)

        return CVResult(
            label=label,
            cv_score=score,
            is_emergency=(label != "NORMAL"),
            detections=dets,
            processing_ms=processing_ms,
            model_version="yolov8n-coco",
            method="yolov8",
            explanation={
                "total_detections":  len(dets),
                "confidence_threshold": confidence_threshold,
                "top_detections": [
                    {"class": d.class_name, "conf": d.confidence, "emergency": d.emergency}
                    for d in sorted(dets, key=lambda x: x.confidence, reverse=True)[:3]
                ],
            },
        )

    except Exception as e:
        logger.error(f"YOLO inference failed: {e}. Using simulation.")
        result = _simulate_cv_result()
        result.processing_ms = round((time.time() - t_start) * 1000, 1)
        return result


def detect_image_bytes_b64(b64_string: str) -> CVResult:
    """Convenience wrapper: accepts base64-encoded image string."""
    raw_bytes = base64.b64decode(b64_string)
    return detect_image(raw_bytes)


def detect_video_frames(
    video_path: str,
    sample_every_n: int = 30,       # analyze every Nth frame (30 = ~1/sec at 30fps)
    max_frames: int = 10,           # safety cap — don't process 10k frame videos
    confidence_threshold: float = 0.25,
) -> CVResult:
    """
    Extracts frames from a video and runs detection on each.
    Aggregates results by voting across frames.

    Args:
        video_path:      path to video file
        sample_every_n:  frame sampling rate (higher = faster, less accurate)
        max_frames:      maximum frames to analyze
        confidence_threshold: YOLO minimum confidence

    Returns:
        CVResult aggregated across all sampled frames
    """
    t_start = time.time()

    if not _cv2_available:
        logger.warning("OpenCV not available — simulating video result")
        result = _simulate_cv_result()
        result.frame_count = 5
        result.method = "simulated-video"
        return result

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_results = []
    frame_idx     = 0
    analyzed      = 0

    try:
        while analyzed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n == 0:
                # frame is BGR numpy array — detect_image handles numpy
                result = detect_image(frame, confidence_threshold)
                frame_results.append(result)
                analyzed += 1

            frame_idx += 1
    finally:
        cap.release()

    if not frame_results:
        return CVResult(label="NORMAL", cv_score=0.05, is_emergency=False,
                        frame_count=0, method="yolov8-video")

    # Aggregate: take the worst-case (highest risk) across frames
    # Reasoning: if ANY frame shows CRITICAL, report CRITICAL
    best = max(frame_results, key=lambda r: r.cv_score)

    # But average the score to avoid single-frame noise
    avg_score   = sum(r.cv_score for r in frame_results) / len(frame_results)
    final_score = round((best.cv_score * 0.6) + (avg_score * 0.4), 3)

    processing_ms = round((time.time() - t_start) * 1000, 1)

    return CVResult(
        label=best.label,
        cv_score=final_score,
        is_emergency=(best.label != "NORMAL"),
        detections=best.detections,
        frame_count=len(frame_results),
        processing_ms=processing_ms,
        model_version=best.model_version,
        method="yolov8-video",
        explanation={
            "frames_analyzed":  len(frame_results),
            "frames_total":     frame_idx,
            "sample_every_n":   sample_every_n,
            "peak_frame_score": best.cv_score,
            "avg_frame_score":  round(avg_score, 3),
            "final_score":      final_score,
        },
    )


# ── Fine-tuning code (run with GPU) 
FINETUNE_YAML = """
# dataset.yaml — for YOLOv8 fine-tuning on emergency detection
# 
# To fine-tune:
#   1. Download UCF-Crime dataset or FireNet dataset from Kaggle
#   2. Convert to YOLO format (each image gets a .txt with bboxes)
#   3. Fill in paths below
#   4. Run: python scripts/finetune_yolo.py
#
# YOLO format for each label file:
#   <class_id> <x_center> <y_center> <width> <height>   (all normalized 0-1)

path: app/cv/data            # root dir
train: images/train          # relative to path
val:   images/val

nc: 7                        # number of classes
names:
  0: ACCIDENT
  1: FIRE
  2: FLOOD
  3: CRIME
  4: CROWD
  5: MEDICAL
  6: NORMAL
"""

def generate_finetune_yaml():
    """Writes the dataset YAML for fine-tuning. Call once when you have data."""
    yaml_path = CV_DIR / "data" / "dataset.yaml"
    yaml_path.write_text(FINETUNE_YAML)
    logger.info(f"Fine-tune YAML written to {yaml_path}")
    return yaml_path


def get_cv_model_info() -> dict:
    """Returns current CV model status."""
    model_files = list(MODELS_DIR.glob("*.pt"))
    return {
        "ultralytics_available": _ultralytics_available,
        "opencv_available":      _cv2_available,
        "model_loaded":          _detect_model is not None,
        "model_files":           [f.name for f in model_files],
        "method":                "yolov8" if _ultralytics_available else "simulated",
    }