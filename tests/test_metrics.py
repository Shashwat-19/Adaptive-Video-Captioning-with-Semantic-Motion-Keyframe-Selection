import math

import pytest

pytest.importorskip("numpy")

from video_captioning.evaluation.metrics import METRIC_NAMES, evaluation_pairs


def test_evaluation_pairs_excludes_missing_references():
    pairs = evaluation_pairs(
        [
            {"video_id": "video_a", "caption": "a dog runs"},
            {"video_id": "video_b", "caption": "ignored"},
        ],
        {"video_a": ["a dog runs"]},
    )

    assert pairs == [("a dog runs", ["a dog runs"])]


def test_metric_names_are_complete_and_stable():
    assert METRIC_NAMES == ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr")
    assert not math.isnan(float(len(METRIC_NAMES)))
