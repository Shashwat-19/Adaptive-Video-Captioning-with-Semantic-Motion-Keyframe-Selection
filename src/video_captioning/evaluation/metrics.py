"""Automatic caption metrics implemented by the research notebook."""

from __future__ import annotations

from typing import Iterable

import numpy as np


METRIC_NAMES = ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr")


def evaluation_pairs(
    predictions: Iterable[dict], references: dict[str, list[str]]
) -> list[tuple[str, list[str]]]:
    """Keep only predictions that have at least one reference caption."""
    return [
        (str(row["caption"]), references[str(row["video_id"])])
        for row in predictions
        if str(row.get("video_id")) in references and references[str(row["video_id"])]
    ]


def evaluate_captions(
    predictions: Iterable[dict], references: dict[str, list[str]]
) -> dict[str, float]:
    """Calculate BLEU, METEOR, ROUGE-L, and optional CIDEr for caption predictions.

    ``predictions`` accepts dataframe records or ordinary dictionaries with ``video_id``
    and ``caption`` keys. CIDEr remains ``NaN`` if ``pycocoevalcap`` is unavailable.
    """
    pairs = evaluation_pairs(predictions, references)
    if not pairs:
        return {metric: float("nan") for metric in METRIC_NAMES}

    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    hypotheses = [caption.split() for caption, _ in pairs]
    reference_tokens = [[caption.split() for caption in captions] for _, captions in pairs]
    smoothing = SmoothingFunction().method1
    scores = {
        f"BLEU-{size}": corpus_bleu(
            reference_tokens,
            hypotheses,
            weights=tuple([1 / size] * size),
            smoothing_function=smoothing,
        )
        for size in range(1, 5)
    }
    scores["METEOR"] = float(
        np.mean(
            [meteor_score([reference.split() for reference in refs], caption.split()) for caption, refs in pairs]
        )
    )
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores["ROUGE-L"] = float(
        np.mean(
            [
                max(scorer.score(reference, caption)["rougeL"].fmeasure for reference in refs)
                for caption, refs in pairs
            ]
        )
    )
    scores["CIDEr"] = float("nan")
    try:
        from pycocoevalcap.cider.cider import Cider

        ground_truth = {str(index): refs for index, (_, refs) in enumerate(pairs)}
        generated = {str(index): [caption] for index, (caption, _) in enumerate(pairs)}
        scores["CIDEr"] = float(Cider().compute_score(ground_truth, generated)[0])
    except ImportError:
        pass
    return scores
