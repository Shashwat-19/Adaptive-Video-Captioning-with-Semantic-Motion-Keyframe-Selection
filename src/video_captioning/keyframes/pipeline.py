"""End-to-end candidate extraction and keyframe selection for a single video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from video_captioning.data.preprocessing import extract_candidate_frames
from video_captioning.keyframes.motion import motion_scores
from video_captioning.keyframes.selector import select_keyframe_positions
from video_captioning.keyframes.semantic import load_semantic_encoder, semantic_scores
from video_captioning.utils.config import ExperimentConfig


@dataclass
class KeyframeSelection:
    """Selected frame data and metadata from one video."""

    frames: list
    frame_indices: list[int]
    timestamps: list[float]
    candidate_count: int


def select_frames_from_video(
    video_path: str | Path,
    config: ExperimentConfig,
    method: str | None = None,
    semantic_runtime: tuple | None = None,
) -> tuple[KeyframeSelection, tuple | None]:
    """Extract candidates and apply one configured baseline or adaptive method.

    A previously loaded ``semantic_runtime`` can be supplied to reuse DINOv2 over
    many videos, avoiding repeated checkpoint loads during batch work.
    """
    method = method or config.keyframe_method
    frames, indices, timestamps = extract_candidate_frames(video_path, config.num_candidate_frames)
    kwargs = {"seed": config.seed}
    if method not in {"uniform", "random"}:
        if semantic_runtime is None:
            semantic_runtime = load_semantic_encoder()
        processor, model, device = semantic_runtime
        semantic, _ = semantic_scores(frames, processor, model, device)
        kwargs.update(
            motion=motion_scores(frames),
            semantic=semantic,
            alpha=config.alpha,
            min_temporal_distance=config.min_temporal_distance,
        )
    positions = select_keyframe_positions(
        method, indices, timestamps, config.num_selected_frames, **kwargs
    )
    return (
        KeyframeSelection(
            frames=[frames[position] for position in positions],
            frame_indices=[indices[position] for position in positions],
            timestamps=[timestamps[position] for position in positions],
            candidate_count=len(frames),
        ),
        semantic_runtime,
    )
