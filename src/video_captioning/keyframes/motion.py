"""Optical-flow motion scoring used by the research workflow."""

from __future__ import annotations

import numpy as np


def minmax_normalize(scores: np.ndarray | list[float]) -> np.ndarray:
    """Scale a score vector to [0, 1], retaining all-zero degenerate vectors."""
    values = np.asarray(scores, dtype=float)
    if len(values) == 0 or np.ptp(values) < 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / np.ptp(values)


def motion_scores(frames: list) -> np.ndarray:
    """Compute normalized candidate-level Farnebäck optical-flow magnitude."""
    if not frames:
        return np.array([])
    import cv2

    grayscale = [
        cv2.resize(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY), (160, 120))
        for frame in frames
    ]
    scores = [0.0]
    for previous, current in zip(grayscale[:-1], grayscale[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        scores.append(float(np.linalg.norm(flow, axis=2).mean()))
    return minmax_normalize(scores)
