"""Select adaptive keyframes from one video and print the selection as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from video_captioning.keyframes.pipeline import select_frames_from_video
from video_captioning.keyframes.selector import VALID_METHODS
from video_captioning.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Path to an input video.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Experiment YAML path.")
    parser.add_argument("--method", choices=sorted(VALID_METHODS), help="Override keyframe method.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    config = load_config(args.config)
    method = args.method or config.keyframe_method
    selection, _ = select_frames_from_video(args.video, config, method)
    payload = {
        "video": str(args.video),
        "method": method,
        "candidate_count": selection.candidate_count,
        "selected_frames": [
            {"frame_index": frame_index, "timestamp_s": timestamp}
            for frame_index, timestamp in zip(selection.frame_indices, selection.timestamps)
        ],
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
