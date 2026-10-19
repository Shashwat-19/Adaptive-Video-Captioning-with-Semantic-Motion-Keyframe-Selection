import importlib.util
import math
import sys
import types

import pytest

pytest.importorskip("numpy")

from video_captioning.evaluation.metrics import (
    METRIC_NAMES,
    ensure_nltk_data,
    evaluate_captions,
    evaluation_pairs,
)


def fake_nltk(installed: set[str]) -> tuple[types.ModuleType, list[str]]:
    downloads: list[str] = []

    def find(path: str) -> str:
        if path.removeprefix("corpora/") not in installed:
            raise LookupError(path)
        return path

    module = types.ModuleType("nltk")
    module.data = types.SimpleNamespace(find=find)
    module.download = lambda resource, quiet=False: downloads.append(resource)
    return module, downloads


@pytest.mark.parametrize(
    ("installed", "expected"),
    [(set(), ["wordnet"]), ({"wordnet"}, []), ({"wordnet.zip"}, [])],
)
def test_ensure_nltk_data_downloads_only_missing_wordnet(monkeypatch, installed, expected):
    module, downloads = fake_nltk(installed)
    monkeypatch.setitem(sys.modules, "nltk", module)

    ensure_nltk_data()

    assert downloads == expected


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


def test_evaluate_captions_returns_nan_for_every_metric_without_references():
    scores = evaluate_captions([{"video_id": "a", "caption": "a dog runs"}], {})

    assert list(scores) == list(METRIC_NAMES)
    assert all(math.isnan(value) for value in scores.values())


def test_evaluate_captions_end_to_end_on_a_fixed_input():
    # Needs nltk and rouge-score; on first run ensure_nltk_data downloads the wordnet corpus.
    pytest.importorskip("nltk")
    pytest.importorskip("rouge_score")
    predictions = [
        {"video_id": "a", "caption": "a man is running"},
        {"video_id": "b", "caption": "a dog plays"},
        {"video_id": "c", "caption": "has no references"},
    ]
    references = {"a": ["a man runs", "a person is running"], "b": ["a dog is playing", "the puppy plays"], "c": []}

    scores = evaluate_captions(predictions, references)

    # Expected values from the pinned nltk 3.9.1, rouge-score 0.1.2 and pycocoevalcap 1.2.
    # BLEU-2 = sqrt(4/4 unigram x 3/5 bigram precision); ROUGE-L is unstemmed: per-pair best F is 0.75 and 4/7, so the mean is 37/56.
    expected = {
        "BLEU-1": 1.0,
        "BLEU-2": 0.7745966692414834,
        "BLEU-3": 0.2714417616594907,
        "BLEU-4": 0.1778279410038923,
        "METEOR": 0.7398217075636431,
        "ROUGE-L": 0.6607142857142857,
    }
    assert {name: scores[name] for name in expected} == pytest.approx(expected, rel=1e-9)
    if importlib.util.find_spec("pycocoevalcap") is None:
        assert math.isnan(scores["CIDEr"])
    else:
        assert scores["CIDEr"] == pytest.approx(1.8956396746556945, rel=1e-9)


def test_metric_protocol_ignores_case_and_punctuation():
    pytest.importorskip("nltk")
    pytest.importorskip("rouge_score")
    plain = evaluate_captions([{"video_id": "a", "caption": "a man runs"}], {"a": ["a man runs"]})
    noisy = evaluate_captions([{"video_id": "a", "caption": "A MAN, runs!"}], {"a": ["a man runs."]})

    names = [name for name in METRIC_NAMES if name != "CIDEr"]
    assert {name: noisy[name] for name in names} == pytest.approx({name: plain[name] for name in names})


def test_rouge_l_is_unstemmed():
    pytest.importorskip("nltk")
    pytest.importorskip("rouge_score")

    assert evaluate_captions([{"video_id": "a", "caption": "runs"}], {"a": ["running"]})["ROUGE-L"] == 0.0


def test_punctuation_only_caption_is_excluded():
    scores = evaluate_captions([{"video_id": "a", "caption": "!!!"}], {"a": ["a dog"]})

    assert all(math.isnan(value) for value in scores.values())
