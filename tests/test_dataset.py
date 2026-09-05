import pytest

pd = pytest.importorskip("pandas")

from video_captioning.data.dataset import normalize_captions, split_by_video_id


def test_normalize_captions_accepts_common_msvd_column_names():
    raw = pd.DataFrame(
        {
            "Video Name": ["clip_a.avi", "clip_b.mp4", ""],
            "Description": ["A person walks", "A dog runs", "Ignored"],
        }
    )

    normalized = normalize_captions(raw)

    assert normalized[["video_id", "caption"]].to_dict("records") == [
        {"video_id": "clip_a", "caption": "A person walks"},
        {"video_id": "clip_b", "caption": "A dog runs"},
    ]


def test_video_level_splits_are_disjoint():
    captions = pd.DataFrame(
        {"video_id": [f"video_{index}" for index in range(20)], "caption": ["caption"] * 20}
    )

    splits = split_by_video_id(captions, seed=7)

    assert not (splits["train"] & splits["validation"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["validation"] & splits["test"])
    assert set().union(*splits.values()) == set(captions["video_id"])
