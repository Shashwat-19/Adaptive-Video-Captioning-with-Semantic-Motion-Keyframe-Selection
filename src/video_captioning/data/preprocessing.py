"""Video metadata inspection and candidate-frame extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def inspect_video(video_path: str | Path) -> dict[str, float | int | str | bool]:
    """Inspect video decodeability and return basic media metadata."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not decode video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Invalid fps/frame_count ({fps}, {frame_count})")
        return {
            "readable": True,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_s": frame_count / fps,
            "error": "",
        }
    except Exception as exc:
        return {
            "readable": False,
            "fps": float("nan"),
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "duration_s": float("nan"),
            "error": str(exc),
        }
    finally:
        capture.release()


def extract_candidate_frames(video_path: str | Path, num_frames: int):
    """Decode temporally-spaced RGB PIL candidate frames and their timestamps."""
    if num_frames < 1:
        raise ValueError("num_frames must be at least one.")
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total <= 0 or fps <= 0:
            raise RuntimeError(f"Invalid video metadata for {video_path}")
        requested = np.unique(np.linspace(0, total - 1, min(num_frames, total), dtype=int))
        frames, actual = [], []
        for index in requested:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, bgr = capture.read()
            if success:
                frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                actual.append(int(index))
        return frames, actual, [index / fps for index in actual]
    finally:
        capture.release()
