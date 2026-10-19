import ast
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

import video_captioning.keyframes.pipeline as pipeline
from video_captioning.utils.config import ExperimentConfig

FINETUNING_SOURCE = Path(__file__).resolve().parents[1] / "src" / "video_captioning" / "training" / "finetuning.py"


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    loads = []
    monkeypatch.setattr(
        pipeline,
        "extract_candidate_frames",
        lambda path, num_frames: (list(range(6)), [0, 10, 20, 30, 40, 50], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    monkeypatch.setattr(pipeline, "load_semantic_encoder", lambda: loads.append(1) or ("processor", "model", "cpu"))
    monkeypatch.setattr(pipeline, "semantic_scores", lambda frames, *args: (np.linspace(0, 1, len(frames)), None))
    monkeypatch.setattr(pipeline, "motion_scores", lambda frames: np.linspace(1, 0, len(frames)))
    return loads


def test_semantic_runtime_is_reused_across_videos(stubbed_pipeline):
    config = ExperimentConfig(num_selected_frames=2, keyframe_method="semantic_motion_diverse")

    _, runtime = pipeline.select_frames_from_video("video_a", config)
    selection, reused = pipeline.select_frames_from_video("video_b", config, semantic_runtime=runtime)

    assert reused is runtime
    assert len(stubbed_pipeline) == 1
    assert len(selection.frame_indices) == 2


def test_finetuning_passes_semantic_runtime_by_keyword():
    calls = [
        node
        for node in ast.walk(ast.parse(FINETUNING_SOURCE.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "select_frames_from_video"
    ]

    assert calls
    for call in calls:
        assert len(call.args) <= 2, "the third positional parameter is `method`, not the semantic runtime"
        assert "semantic_runtime" in {keyword.arg for keyword in call.keywords}


def test_random_baseline_varies_by_video_and_repeats_per_video(stubbed_pipeline):
    config = ExperimentConfig(num_selected_frames=2, keyframe_method="random")
    names = [f"video_{i}" for i in range(6)]
    picks = {name: pipeline.select_frames_from_video(name, config)[0].frame_indices for name in names}

    assert len({tuple(p) for p in picks.values()}) > 1
    assert pipeline.select_frames_from_video("video_0", config)[0].frame_indices == picks["video_0"]
