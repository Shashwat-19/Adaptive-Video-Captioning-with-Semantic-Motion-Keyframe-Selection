import pytest

np = pytest.importorskip("numpy")

from video_captioning.keyframes.selector import (
    fused_scores,
    select_diverse_keyframes,
    select_keyframe_positions,
    uniform_selection,
)


def test_uniform_selection_spans_candidate_timeline():
    assert uniform_selection([0, 10, 20, 30, 40], 3) == [0, 20, 40]


def test_fused_score_respects_alpha_endpoints():
    semantic = np.array([0.0, 0.5, 1.0])
    motion = np.array([1.0, 0.5, 0.0])

    assert np.allclose(fused_scores(semantic, motion, alpha=1.0), semantic)
    assert np.allclose(fused_scores(semantic, motion, alpha=0.0), motion)


def test_diverse_selection_prefers_separated_high_score_frames():
    indices = [0, 10, 20, 30]
    selected, _, _ = select_diverse_keyframes(indices, [0.9, 1.0, 0.8, 0.7], [0.0, 0.1, 2.0, 4.0], 2, 1.0)

    assert selected == [10, 20]


def test_adaptive_selection_returns_chronological_candidate_positions():
    positions = select_keyframe_positions(
        "semantic_motion",
        indices=[0, 10, 20, 30],
        timestamps=[0.0, 1.0, 2.0, 3.0],
        k=2,
        semantic=[0.1, 0.9, 0.8, 0.2],
        motion=[0.1, 0.8, 0.9, 0.2],
    )

    assert positions == [1, 2]
