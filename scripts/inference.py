"""Generate a caption for a local video using the implemented selection workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from video_captioning.keyframes.pipeline import select_frames_from_video
from video_captioning.keyframes.selector import VALID_METHODS
from video_captioning.models.captioner import BLIP2VideoCaptioner
from video_captioning.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Path to an input video.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Experiment YAML path.")
    parser.add_argument("--method", choices=sorted(VALID_METHODS), help="Override keyframe method.")
    parser.add_argument("--checkpoint", type=Path, help="Optional local fine-tuned checkpoint.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    config = load_config(args.config)
    method = args.method or config.keyframe_method
    selection, _ = select_frames_from_video(args.video, config, method)
    captioner = BLIP2VideoCaptioner(
        config.model_name, config.image_size, config.max_caption_length, config.use_4bit
    )
    if args.checkpoint:
        captioner.load_checkpoint(args.checkpoint)
    caption = captioner.generate_caption(selection.frames)
    payload = {
        "video": str(args.video),
        "method": method,
        "selected_frame_indices": selection.frame_indices,
        "caption": caption,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
