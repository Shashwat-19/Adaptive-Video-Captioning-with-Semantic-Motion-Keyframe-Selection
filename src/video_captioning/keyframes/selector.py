"""Sampling baselines and adaptive semantic-motion keyframe selection."""

from __future__ import annotations

import numpy as np

from video_captioning.keyframes.motion import minmax_normalize


VALID_METHODS = {
    "uniform",
    "random",
    "motion",
    "semantic",
    "semantic_motion",
    "semantic_motion_diverse",
}


def uniform_selection(indices: list[int], k: int) -> list[int]:
    """Choose a temporally uniform baseline subset of candidate frame indices."""
    if k < 1:
        raise ValueError("k must be at least one.")
    if not indices:
        return []
    positions = np.unique(np.linspace(0, len(indices) - 1, min(k, len(indices)), dtype=int))
    return np.asarray(indices)[positions].tolist()


def random_selection(indices: list[int], k: int, seed: int = 42) -> list[int]:
    """Choose a deterministic random baseline subset."""
    if k < 1:
        raise ValueError("k must be at least one.")
    if not indices:
        return []
    generator = np.random.default_rng(seed)
    return sorted(generator.choice(np.asarray(indices), size=min(k, len(indices)), replace=False).tolist())


def fused_scores(
    semantic: np.ndarray | list[float], motion: np.ndarray | list[float], alpha: float
) -> np.ndarray:
    """Fuse normalized semantic novelty and motion relevance with semantic weight ``alpha``."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one.")
    semantic_values = minmax_normalize(semantic)
    motion_values = minmax_normalize(motion)
    if len(semantic_values) != len(motion_values):
        raise ValueError("semantic and motion scores must have the same length.")
    return alpha * semantic_values + (1 - alpha) * motion_values


def select_diverse_keyframes(
    indices: list[int],
    scores: np.ndarray | list[float],
    timestamps: list[float],
    k: int,
    min_temporal_distance: float,
) -> tuple[list[int], list[float], list[float]]:
    """Select high-scoring frames using greedy temporal NMS and deterministic backfill."""
    if not (len(indices) == len(scores) == len(timestamps)):
        raise ValueError("indices, scores, and timestamps must have equal length.")
    if k < 1 or not indices:
        return [], [], []
    chosen_positions: list[int] = []
    for position in np.argsort(scores)[::-1]:
        if all(abs(timestamps[position] - timestamps[chosen]) >= min_temporal_distance for chosen in chosen_positions):
            chosen_positions.append(int(position))
        if len(chosen_positions) == min(k, len(indices)):
            break
    for position in np.argsort(scores)[::-1]:
        if len(chosen_positions) == min(k, len(indices)):
            break
        if int(position) not in chosen_positions:
            chosen_positions.append(int(position))
    chosen_positions.sort()
    return (
        [indices[position] for position in chosen_positions],
        [float(scores[position]) for position in chosen_positions],
        [float(timestamps[position]) for position in chosen_positions],
    )


def select_keyframe_positions(
    method: str,
    indices: list[int],
    timestamps: list[float],
    k: int,
    motion: np.ndarray | list[float] | None = None,
    semantic: np.ndarray | list[float] | None = None,
    alpha: float = 0.5,
    min_temporal_distance: float = 0.5,
    seed: int = 42,
) -> list[int]:
    """Return candidate-frame positions for a supported baseline or adaptive method."""
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(VALID_METHODS)}")
    if method == "uniform":
        selected = uniform_selection(indices, k)
        return [indices.index(frame) for frame in selected]
    if method == "random":
        selected = random_selection(indices, k, seed)
        return [indices.index(frame) for frame in selected]
    if motion is None or semantic is None:
        raise ValueError(f"{method} selection requires motion and semantic scores.")
    score_map = {
        "motion": minmax_normalize(motion),
        "semantic": minmax_normalize(semantic),
        "semantic_motion": fused_scores(semantic, motion, alpha),
        "semantic_motion_diverse": fused_scores(semantic, motion, alpha),
    }
    scores = score_map[method]
    if method == "semantic_motion_diverse":
        selected, _, _ = select_diverse_keyframes(
            indices, scores, timestamps, k, min_temporal_distance
        )
        return [indices.index(frame) for frame in selected]
    return sorted(np.argsort(scores)[::-1][: min(k, len(indices))].tolist())
