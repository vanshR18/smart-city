"""
frame_extractor.py
──────────────────
Extracts frames from video files for YOLO inference.

Why a separate module?
  - Frame extraction is I/O heavy; detection is CPU/GPU heavy.
    Keeping them separate lets you parallelize later.
  - You can extract all frames once, cache them, and re-run detection
    with different models/thresholds without re-reading the video.
"""

import os
import time
from pathlib import Path
from typing import Generator
from dataclasses import dataclass
from loguru import logger

try:
    import cv2
    _cv2_available = True
except ImportError:
    _cv2_available = False

try:
    import numpy as np
    _np_available = True
except ImportError:
    _np_available = False


@dataclass
class VideoMetadata:
    """Basic info about a video file."""
    path:        str
    fps:         float
    width:       int
    height:      int
    total_frames:int
    duration_sec:float
    codec:       str


@dataclass
class ExtractedFrame:
    """A single extracted frame with its metadata."""
    frame_idx:   int         # index in original video
    timestamp_sec: float     # time in video
    image:       "np.ndarray"  # BGR numpy array (what OpenCV gives us)


def get_video_metadata(video_path: str) -> VideoMetadata:
    """
    Reads video metadata without extracting frames.
    Fast — only opens the file header.
    """
    if not _cv2_available:
        raise ImportError("opencv-python-headless required for video processing")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec_int    = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec        = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])
        duration     = total_frames / fps if fps > 0 else 0
    finally:
        cap.release()

    return VideoMetadata(
        path=video_path, fps=fps, width=width, height=height,
        total_frames=total_frames, duration_sec=round(duration, 2),
        codec=codec,
    )


def extract_frames(
    video_path: str,
    sample_every_n: int = 30,
    max_frames: int = 20,
    resize_to: tuple[int, int] | None = (640, 640),
) -> Generator[ExtractedFrame, None, None]:
    """
    Generator that yields frames from a video file.

    Args:
        video_path:    path to video
        sample_every_n: yield every Nth frame (30 ≈ 1 frame/sec at 30fps)
        max_frames:    stop after this many frames (prevents OOM on long videos)
        resize_to:     (width, height) to resize frame. None = keep original.
                       640x640 is YOLOv8's native input size.

    Yields:
        ExtractedFrame objects (one at a time — no memory accumulation)
    """
    if not _cv2_available:
        raise ImportError("opencv-python-headless required")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    yielded   = 0

    try:
        while yielded < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n == 0:
                if resize_to:
                    frame = cv2.resize(frame, resize_to)

                yield ExtractedFrame(
                    frame_idx=frame_idx,
                    timestamp_sec=round(frame_idx / fps, 2),
                    image=frame,
                )
                yielded += 1

            frame_idx += 1

        logger.info(f"Extracted {yielded} frames from {Path(video_path).name} "
                    f"(total {frame_idx} frames processed)")
    finally:
        cap.release()


def save_frames_to_disk(
    video_path: str,
    output_dir: str,
    sample_every_n: int = 30,
    max_frames: int = 50,
) -> list[str]:
    """
    Saves extracted frames as JPEG files to disk.
    Returns list of saved file paths.

    Use when you want to:
    - Inspect frames visually during debugging
    - Build a labeled dataset for fine-tuning
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    video_name  = Path(video_path).stem

    for frame in extract_frames(video_path, sample_every_n, max_frames):
        filename = f"{video_name}_frame{frame.frame_idx:06d}_t{frame.timestamp_sec:.1f}s.jpg"
        filepath = output / filename
        cv2.imwrite(str(filepath), frame.image)
        saved_paths.append(str(filepath))

    logger.success(f"Saved {len(saved_paths)} frames to {output_dir}")
    return saved_paths