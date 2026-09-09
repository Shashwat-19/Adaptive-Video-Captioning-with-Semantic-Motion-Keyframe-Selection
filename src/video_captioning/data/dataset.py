"""Caption-table parsing and leakage-safe video-level splitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


VIDEO_EXTENSIONS = {".avi", ".mp4", ".webm", ".mov", ".mkv"}


def discover_caption_file(root: Path, configured: Path) -> Path:
    """Return the configured caption file or a caption-like table below ``root``."""
    if configured.exists():
        return configured
    choices = [*root.rglob("*.csv"), *root.rglob("*.json"), *root.rglob("*.tsv")]
    captionish = [
        path
        for path in choices
        if any(token in path.name.lower() for token in ("caption", "description", "annotation"))
    ]
    if captionish:
        return captionish[0]
    raise FileNotFoundError(
        f"Caption file absent: {configured}. Set caption_file in the configuration."
    )


def read_caption_table(path: str | Path) -> pd.DataFrame:
    """Read CSV, TSV, or JSON captions into a dataframe without schema assumptions."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        return pd.json_normalize(raw if isinstance(raw, list) else raw.get("data", raw))
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else None, engine="python")


def normalize_captions(raw: pd.DataFrame) -> pd.DataFrame:
    """Map common MSVD column names to non-empty ``video_id`` and ``caption`` fields."""
    lookup = {
        str(column).strip().lower().replace(" ", "_").replace("-", "_"): column
        for column in raw.columns
    }

    def pick(options: list[str]) -> str | None:
        return next((lookup[option] for option in options if option in lookup), None)

    id_column = pick(["video_id", "video", "video_name", "videoid", "clip_id", "filename", "file_name"])
    caption_column = pick(["caption", "description", "sentence", "text", "caption_text"])
    start_column = pick(["start", "start_time", "start_frame"])
    end_column = pick(["end", "end_time", "end_frame"])
    if not id_column or not caption_column:
        raise ValueError(
            "Cannot map video/caption columns. "
            f"Found {list(raw.columns)}; update normalize_captions() for this schema."
        )
    normalized = raw.copy().rename(columns={id_column: "video_id", caption_column: "caption"})
    if start_column and end_column:
        normalized["video_id"] = (
            normalized["video_id"].astype(str).str.strip()
            + "_"
            + normalized[start_column].astype(str).str.strip()
            + "_"
            + normalized[end_column].astype(str).str.strip()
        )
    else:
        normalized["video_id"] = (
            normalized["video_id"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.(avi|mp4|webm|mov|mkv)$", "", regex=True, case=False)
        )
    normalized["caption"] = normalized["caption"].astype(str).str.strip()
    return normalized.loc[
        normalized["video_id"].ne("") & normalized["caption"].ne("")
    ].reset_index(drop=True)


def index_video_paths(video_root: str | Path) -> dict[str, Path]:
    """Index supported video files by filename stem."""
    root = Path(video_root)
    if not root.exists():
        return {}
    return {
        path.stem: path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    }


def split_by_video_id(
    captions: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, set[str]]:
    """Create disjoint train/validation/test sets from unique video identifiers."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.")
    ids = np.array(sorted(captions["video_id"].unique()))
    if len(ids) < 3:
        raise ValueError("At least three unique videos are required for train/validation/test splits.")
    train_ids, holdout = train_test_split(
        ids, test_size=val_ratio + test_ratio, random_state=seed, shuffle=True
    )
    validation_ids, test_ids = train_test_split(
        holdout,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        shuffle=True,
    )
    splits = {"train": set(train_ids), "validation": set(validation_ids), "test": set(test_ids)}
    if splits["train"] & splits["validation"] or splits["train"] & splits["test"]:
        raise RuntimeError("Video split overlap detected.")
    return splits
