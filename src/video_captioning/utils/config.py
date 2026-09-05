"""Configuration loading for reproducible experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """Runtime settings shared by the notebook and command-line tools."""

    data_root: str = "data/msvd"
    video_dir: str = "YouTubeClips"
    caption_file: str = "captions.csv"
    output_dir: str = "outputs/default"
    seed: int = 42
    num_candidate_frames: int = 30
    num_selected_frames: int = 8
    alpha: float = 0.5
    min_temporal_distance: float = 0.5
    image_size: int = 224
    model_name: str = "Salesforce/blip2-opt-2.7b"
    max_caption_length: int = 32
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 3
    num_workers: int = 2
    use_lora: bool = True
    use_4bit: bool = False
    keyframe_method: str = "semantic_motion_diverse"
    max_train_videos: int | None = None

    @property
    def video_root(self) -> Path:
        return Path(self.data_root) / self.video_dir

    @property
    def caption_path(self) -> Path:
        return Path(self.data_root) / self.caption_file

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of this configuration."""
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML configuration, resolving a single optional parent config."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML to read experiment configuration files.") from exc

    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("inherits", None)
    if parent:
        parent_data = load_config(path.parent / parent).as_dict()
        parent_data.update(data)
        data = parent_data
    known = {name for name in ExperimentConfig.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config fields in {path}: {sorted(unknown)}")
    return ExperimentConfig(**data)
