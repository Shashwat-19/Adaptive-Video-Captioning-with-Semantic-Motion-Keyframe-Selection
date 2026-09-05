"""Evaluate a generated-caption CSV against the configured MSVD caption file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from video_captioning.data.dataset import normalize_captions, read_caption_table
from video_captioning.evaluation.metrics import evaluate_captions
from video_captioning.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="CSV with video_id and caption columns.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Experiment YAML path.")
    parser.add_argument("--output", type=Path, help="Optional metrics JSON output path.")
    args = parser.parse_args()

    config = load_config(args.config)
    predictions = pd.read_csv(args.predictions)
    required = {"video_id", "caption"}
    if missing := required - set(predictions.columns):
        raise ValueError(f"Predictions CSV is missing columns: {sorted(missing)}")
    captions = normalize_captions(read_caption_table(config.caption_path))
    references = captions.groupby("video_id")["caption"].apply(list).to_dict()
    metrics = evaluate_captions(predictions.to_dict("records"), references)
    rendered = json.dumps(metrics, indent=2, allow_nan=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
