"""Run the optional BLIP-2/LoRA fine-tuning workflow from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from video_captioning.data.dataset import (
    index_video_paths,
    normalize_captions,
    read_caption_table,
    split_by_video_id,
)
from video_captioning.training.finetuning import run_finetuning
from video_captioning.utils.config import load_config
from video_captioning.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml", help="Experiment YAML path.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.seed)
    captions = normalize_captions(read_caption_table(config.caption_path))
    video_paths = index_video_paths(config.video_root)
    splits = split_by_video_id(
        captions, config.train_ratio, config.val_ratio, config.test_ratio, config.seed
    )
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_text(
        json.dumps(config.as_dict(), indent=2), encoding="utf-8"
    )
    history = run_finetuning(config, captions, video_paths, splits["train"], splits["validation"])
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
