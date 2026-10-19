"""Build the Colab-ready MSVD video-captioning research notebook."""
import json
from pathlib import Path
from textwrap import dedent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


cells = [
md(r'''
# Efficient Video Captioning using Adaptive Semantic-Motion Keyframe Selection

**MSVD + BLIP-2 research workflow.** This Colab-compatible notebook implements reproducible source-video-grouped splits, frame-selection ablations, T4-conscious LoRA fine-tuning, a validation-only alpha sensitivity report, automatic evaluation, and artifact tracking. Records the active runtime and installs missing dependencies; model compatibility is checked by the validation smoke test. Set the paths in Section 0. A default run now performs a validation smoke test; full training and test studies remain opt-in. Enable `RUN_TEST_ABLATIONS` for the zero-shot study, optionally `RUN_FINETUNING` for both trained arms, and `RUN_VALIDATION_ALPHA_SWEEP` to report alpha sensitivity (alpha stays at the prespecified `ALPHA` for every arm). The notebook freezes the configuration before requested test studies. Use a fresh output directory for a new experiment. Saved outputs are cleared because the revised protocol requires a new run.
'''),
md(r'''
# 0. Research Configuration
'''),
code(r'''
from dataclasses import dataclass, asdict
from pathlib import Path
import os, json, random, time, gc, math, warnings

@dataclass
class ResearchConfig:
    # Configurable MSVD paths; can be set directly or overridden by environment variables.
    DATA_ROOT: str = os.environ.get("MSVD_DATA_ROOT", "/content/msvd")
    VIDEO_DIR: str = os.environ.get("MSVD_VIDEO_DIR", "YouTubeClips")
    CAPTION_FILE: str = os.environ.get("MSVD_CAPTION_FILE", "data/extracted/annotations/AllVideoDescriptions.txt")
    OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "/content/video_captioning_outputs")
    SEED: int = 42
    NUM_CANDIDATE_FRAMES: int = 30
    NUM_SELECTED_FRAMES: int = 8
    ALPHA: float = 0.50
    MIN_TEMPORAL_DISTANCE: float = 0.50  # seconds
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 1
    GRADIENT_ACCUMULATION_STEPS: int = 8
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 3
    MAX_CAPTION_LENGTH: int = 32
    MAX_TRAIN_CAPTION_TOKENS: int = 32  # caption targets including EOS, excluding BOS/image tokens
    NUM_BEAMS: int = 3
    CAPTION_PROMPT: str = ""  # unconditional captioning in both training and inference
    SEMANTIC_MODEL_NAME: str = "facebook/dinov2-small"
    FRAME_BUDGETS: tuple = (4, 8, 12, 16)
    NUM_WORKERS: int = 0  # GPU preprocessing runs in the main process
    MODEL_NAME: str = "Salesforce/blip2-opt-2.7b"
    TRAIN_RATIO: float = 0.80
    VAL_RATIO: float = 0.10
    TEST_RATIO: float = 0.10
    USE_LORA: bool = True
    USE_4BIT: bool = False  # enable only if bitsandbytes is available and necessary
    SMOKE_TEST_VIDEOS: int = 4
    MAX_TRAIN_VIDEOS: int | None = None  # e.g. 100 for a fast pilot
    RUN_SMOKE_TEST: bool = True
    RUN_FULL_METADATA: bool = False
    RUN_VALIDATION_ALPHA_SWEEP: bool = False
    RUN_EFFICIENCY_ANALYSIS: bool = False
    RUN_QUALITATIVE_ANALYSIS: bool = False
    RUN_FINETUNING: bool = False
    RUN_TEST_ABLATIONS: bool = False
    RUN_FRAME_BUDGET_STUDY: bool = False

# Clear prior in-memory experiment tables when rerunning Section 0.
for variable in ["ablation","budget_results","alpha_validation","efficiency","qualitative","smoke_predictions"]:
    globals().pop(variable,None)
CFG = ResearchConfig()
if CFG.CAPTION_PROMPT:
    raise ValueError("This notebook trains unconditional captions; CAPTION_PROMPT must be empty.")
if CFG.NUM_WORKERS != 0:
    raise ValueError("GPU frame preprocessing requires NUM_WORKERS=0.")
if CFG.USE_4BIT and not CFG.USE_LORA:
    raise ValueError("Quantized fine-tuning requires LoRA.")
RUN_ID = os.environ.get("EXPERIMENT_RUN_ID", time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
MODE = "research"
ROOT = Path(CFG.DATA_ROOT)
VIDEO_ROOT = ROOT / CFG.VIDEO_DIR
CAPTION_PATH = ROOT / CFG.CAPTION_FILE
OUT = Path(CFG.OUTPUT_DIR)
ARTIFACT_DIRS = ["results/figures", "results/metrics", "results/generated_captions",
                 "results/frame_scores", "results/qualitative", "results/logs",
                 "checkpoints/uniform_baseline", "checkpoints/proposed", "configs", "cache"]
for relative in ARTIFACT_DIRS:
    (OUT / relative).mkdir(parents=True, exist_ok=True)
with open(OUT / "configs/experiment_config.json", "w") as f:
    json.dump(asdict(CFG), f, indent=2)
print(json.dumps(asdict(CFG), indent=2))
'''),
md(r'''
# 1. Environment and GPU Setup
'''),
code(r'''
# Optional Drive mount: it is never forced, and local /content data works unchanged.
MOUNT_GOOGLE_DRIVE = False
if MOUNT_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
'''),
code(r'''
import importlib.util
import importlib.metadata as importlib_metadata
import subprocess
import sys
import platform
import json
import random
from pathlib import Path

# Keep Colab's installed Python, PyTorch, and CUDA stack.
# Install missing dependencies instead of enforcing older, incompatible pins.
required = {
    "transformers": "transformers",
    "accelerate": "accelerate",
    "peft": "peft",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "tqdm": "tqdm",
    "yaml": "PyYAML",
    "rouge_score": "rouge-score==0.1.2",
    "wordcloud": "wordcloud",
    "sentencepiece": "sentencepiece",
    "nltk": "nltk",
    "pycocoevalcap": "pycocoevalcap==1.2",
}
if CFG.USE_4BIT:
    required["bitsandbytes"] = "bitsandbytes"

if importlib.util.find_spec("torch") is None:
    raise RuntimeError(
        "PyTorch is missing. Select a Colab GPU runtime before continuing."
    )

missing = [
    package
    for module, package in required.items()
    if importlib.util.find_spec(module) is None
]
print("Missing packages:", missing or "none")

if missing:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--upgrade-strategy", "only-if-needed",
        *missing,
    ])
    importlib.invalidate_caches()

import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from collections import Counter
from IPython.display import display

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG.SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

if CFG.USE_4BIT and device != "cuda":
    raise RuntimeError(
        "USE_4BIT=True requires a supported CUDA GPU runtime."
    )

if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {props.total_memory / 2**30:.2f} GB")
else:
    print("GPU name: CPU (CUDA unavailable)")
    print("GPU memory: n/a")

print(f"Python: {platform.python_version()}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch: {torch.__version__}")
print(f"OpenCV: {cv2.__version__}")
# Several OpenCV wheels all ship the cv2 package, so distribution metadata alone can name a version that is not the one imported.
cv2_distributions = sorted(
    {
        name
        for d in importlib_metadata.distributions()
        for name in [d.metadata.get("Name")]  # .get avoids the KeyError newer Pythons raise for a missing field
        if name and any(str(f).split("/")[0] == "cv2" for f in (d.files or []))
    }
)
if len(cv2_distributions) > 1:
    print(f"WARNING: several packages provide cv2 {cv2_distributions}; trust the imported version above, not the package table.")

# Record the actual environment rather than claiming compatibility
# with the project's older pinned stack.
versions = {}
packages = {
    spec.split("==")[0] for spec in required.values()
} | {"torch", "opencv-python", "opencv-contrib-python"}

for package in sorted(packages):
    try:
        versions[package] = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        versions[package] = None

version_report = {
    "environment_policy": "preserve_runtime_install_missing",
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": versions,
    "cv2_imported_version": cv2.__version__,
    "cv2_distributions": cv2_distributions,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "device": device,
    "gpu_name": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else None
    ),
    "model_workflow_verified": False,
}

OUT = Path(OUT)
(OUT / "configs").mkdir(parents=True, exist_ok=True)
(OUT / "configs/environment_versions.json").write_text(
    json.dumps(version_report, indent=2),
    encoding="utf-8",
)

print(json.dumps(version_report, indent=2))
print(
    "Environment recorded. Run the validation smoke test before training; "
    "this setup check does not verify BLIP-2/LoRA compatibility."
)
'''),
md(r'''
# 0b. Architecture Diagram

Renders the proposed pipeline (candidate sampling -> semantic/motion scoring -> fusion -> temporal diversity -> BLIP-2 -> evaluation) and saves it to `results/figures/architecture.png` for the paper.
'''),
code(r'''
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

boxes = [
    "MSVD videos + captions",
    "Dataset verification & preprocessing",
    "Split by source YouTube video",
    "Candidate frame sampling",
    "Semantic scoring + Motion scoring",
    "Score fusion (alpha × semantic + (1 − alpha) × motion)",
    "Temporal diversity filtering",
    "Top-K adaptive keyframes",
    "BLIP-2 (ViT → Q-Former → LLM)",
    "Generated caption",
    "Evaluation (BLEU / METEOR / ROUGE-L / CIDEr)",
]

fig, ax = plt.subplots(figsize=(7.5, 12))

n = len(boxes)
box_height = 0.7
gap = 0.35
y = n * (box_height + gap)
positions = []

for index, label in enumerate(boxes):
    y -= box_height + gap
    positions.append(y)

    face_color = "#dbe9ff"

    if "fusion" in label.lower() or "diversity" in label.lower():
        face_color = "#ffe8b3"
    elif "BLIP-2" in label:
        face_color = "#d9f2d9"
    elif "Evaluation" in label:
        face_color = "#f0d9f2"

    rectangle = mpatches.FancyBboxPatch(
        (0.05, y),
        0.9,
        box_height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor=face_color,
    )
    ax.add_patch(rectangle)

    ax.text(
        0.5,
        y + box_height / 2,
        label,
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
    )

    if index > 0:
        previous_y = positions[index - 1]

        ax.annotate(
            "",
            xy=(0.5, y + box_height),
            xytext=(0.5, previous_y),
            arrowprops={
                "arrowstyle": "->",
                "color": "#555555",
                "lw": 1.2,
            },
        )

ax.set_xlim(0, 1)
ax.set_ylim(
    min(positions) - 0.2,
    max(positions) + box_height + 0.5,
)
ax.axis("off")
ax.set_title(
    "Adaptive Semantic-Motion Keyframe Selection Pipeline",
    fontsize=12,
    fontweight="bold",
    pad=16,
)

figure_directory = OUT / "results/figures"

figure_directory.mkdir(parents=True, exist_ok=True)
figure_path = figure_directory / "architecture.png"

plt.tight_layout()
fig.savefig(
    figure_path,
    dpi=220,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()
plt.close(fig)

print(f"Saved architecture figure: {figure_path}")
'''),
md(r'''
# 2. Dataset Loading and Verification
'''),
code(r'''
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import tarfile

ROOT = Path(ROOT)
CONTENT_ROOT = ROOT

MSVD_VIDEO_URL = (
    "https://www.cs.utexas.edu/~ml/clamp/"
    "videoDescription/YouTubeClips.tar"
)
MSVD_VIDEO_FILENAME = "YouTubeClips.tar"
MSVD_VIDEO_BYTES = 1_856_460_800
MSVD_VIDEO_MD5 = "9bdb20fcf14d59524a6febca9f6a8d89"
MSVD_EXPECTED_VIDEOS = 1970

RAW_DIRECTORY = ROOT / "data/raw"
VIDEO_ROOT = ROOT / CFG.VIDEO_DIR
VIDEO_DIR = VIDEO_ROOT

RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
VIDEO_ROOT.mkdir(parents=True, exist_ok=True)

VIDEO_SUFFIXES = {
    ".avi",
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
    ".mpg",
    ".mpeg",
}


def streaming_md5(path: Path) -> str:
    digest = hashlib.md5()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(16 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def quarantine_invalid_file(path: Path, reason: str) -> Path:
    timestamp = dt.datetime.now(
        dt.timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    quarantine_path = path.with_name(
        f"{path.name}.invalid-{timestamp}"
    )
    path.replace(quarantine_path)

    print(
        f"Moved invalid file to {quarantine_path} "
        f"because: {reason}"
    )
    return quarantine_path


def locate_valid_existing_archive() -> Path | None:
    candidates = sorted(
        path
        for path in CONTENT_ROOT.rglob(MSVD_VIDEO_FILENAME)
        if path.is_file()
    )

    for candidate in candidates:
        size = candidate.stat().st_size

        if size != MSVD_VIDEO_BYTES:
            print(
                f"Ignoring incomplete archive {candidate}: "
                f"{size:,}/{MSVD_VIDEO_BYTES:,} bytes"
            )
            continue

        print(f"Checking existing archive: {candidate}")
        actual_md5 = streaming_md5(candidate)

        if actual_md5 == MSVD_VIDEO_MD5:
            print(f"Reusing verified archive: {candidate}")
            return candidate

        print(
            f"Ignoring archive with incorrect MD5: "
            f"{candidate} ({actual_md5})"
        )

    return None


def download_video_archive() -> Path:
    destination = RAW_DIRECTORY / MSVD_VIDEO_FILENAME
    partial = destination.with_name(
        destination.name + ".part"
    )

    if destination.exists():
        if (
            destination.stat().st_size == MSVD_VIDEO_BYTES
            and streaming_md5(destination) == MSVD_VIDEO_MD5
        ):
            print(f"Reusing verified archive: {destination}")
            return destination

        quarantine_invalid_file(
            destination,
            "size or MD5 mismatch",
        )

    if partial.exists():
        partial_size = partial.stat().st_size

        if partial_size == MSVD_VIDEO_BYTES:
            print("Complete partial file found; verifying it.")

            if streaming_md5(partial) == MSVD_VIDEO_MD5:
                partial.replace(destination)
                return destination

            quarantine_invalid_file(
                partial,
                "complete partial file has incorrect MD5",
            )

        elif partial_size > MSVD_VIDEO_BYTES:
            quarantine_invalid_file(
                partial,
                "partial file exceeds expected archive size",
            )

        else:
            print(
                f"Resuming download at "
                f"{partial_size:,}/{MSVD_VIDEO_BYTES:,} bytes"
            )

    else:
        print(
            f"Downloading the complete MSVD video archive "
            f"({MSVD_VIDEO_BYTES / 2**30:.2f} GiB)."
        )

    command = [
        "curl",
        "--location",
        "--fail",
        "--show-error",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        MSVD_VIDEO_URL,
    ]

    completed = subprocess.run(command)

    if completed.returncode != 0:
        raise RuntimeError(
            "The automatic MSVD video download failed. "
            f"The resumable partial file remains at {partial}. "
            "Rerun this cell to continue downloading."
        )

    actual_size = partial.stat().st_size

    if actual_size != MSVD_VIDEO_BYTES:
        raise ValueError(
            "Downloaded archive has an unexpected size: "
            f"{actual_size:,}/{MSVD_VIDEO_BYTES:,} bytes. "
            "Rerun the cell to resume."
        )

    print("Download completed. Verifying the published MD5 checksum.")
    actual_md5 = streaming_md5(partial)

    if actual_md5 != MSVD_VIDEO_MD5:
        quarantine_invalid_file(
            partial,
            (
                f"MD5 mismatch: expected {MSVD_VIDEO_MD5}, "
                f"found {actual_md5}"
            ),
        )
        raise ValueError(
            "The downloaded MSVD archive failed checksum validation."
        )

    partial.replace(destination)
    print(f"Verified archive: {destination}")

    return destination


def safely_extract_msvd(
    archive_path: Path,
    destination: Path,
) -> None:
    marker_path = destination / ".extraction.json"

    if marker_path.exists():
        try:
            marker = json.loads(
                marker_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            marker = {}

        existing_videos = [
            path
            for path in destination.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_SUFFIXES
        ]

        if (
            marker.get("archive_md5") == MSVD_VIDEO_MD5
            and marker.get("video_count") == MSVD_EXPECTED_VIDEOS
            and len(existing_videos) == MSVD_EXPECTED_VIDEOS
        ):
            print(
                f"Reusing {len(existing_videos):,} extracted videos "
                f"from {destination}"
            )
            return

    extracted_count = 0
    seen_filenames = set()

    print(f"Extracting MSVD videos to {destination}")

    with tarfile.open(archive_path, mode="r:*") as archive:
        members = [
            member
            for member in archive
            if member.isfile()
            and Path(member.name).suffix.lower()
            in VIDEO_SUFFIXES
        ]

        if len(members) != MSVD_EXPECTED_VIDEOS:
            raise ValueError(
                f"Expected {MSVD_EXPECTED_VIDEOS} video members in "
                f"the TAR archive, found {len(members)}."
            )

        for member in members:
            # Flatten the historical YouTubeClips/ directory safely.
            filename = Path(member.name).name

            if not filename:
                raise ValueError(
                    f"Unsafe or empty TAR member name: {member.name!r}"
                )

            if filename in seen_filenames:
                raise ValueError(
                    f"Duplicate filename inside archive: {filename}"
                )

            seen_filenames.add(filename)
            destination_path = destination / filename

            # Reuse a correctly sized file from an interrupted extraction.
            if (
                destination_path.exists()
                and destination_path.is_file()
                and destination_path.stat().st_size == member.size
            ):
                extracted_count += 1
                continue

            temporary_path = destination_path.with_name(
                destination_path.name + ".part"
            )

            source = archive.extractfile(member)

            if source is None:
                raise RuntimeError(
                    f"Could not read TAR member: {member.name}"
                )

            with source:
                with temporary_path.open("wb") as output:
                    shutil.copyfileobj(
                        source,
                        output,
                        length=8 * 1024 * 1024,
                    )

            if temporary_path.stat().st_size != member.size:
                raise ValueError(
                    f"Incomplete extraction for {member.name}: "
                    f"{temporary_path.stat().st_size}/{member.size} bytes"
                )

            temporary_path.replace(destination_path)
            extracted_count += 1

            if (
                extracted_count % 100 == 0
                or extracted_count == len(members)
            ):
                print(
                    f"Extracted/verified "
                    f"{extracted_count:,}/{len(members):,} videos"
                )

    extracted_videos = sorted(
        path
        for path in destination.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
    )

    unique_ids = {path.stem for path in extracted_videos}

    if (
        len(extracted_videos) != MSVD_EXPECTED_VIDEOS
        or len(unique_ids) != MSVD_EXPECTED_VIDEOS
    ):
        raise ValueError(
            f"Expected {MSVD_EXPECTED_VIDEOS} unique extracted videos; "
            f"found {len(extracted_videos)} files and "
            f"{len(unique_ids)} unique IDs."
        )

    marker = {
        "archive": str(archive_path),
        "archive_md5": MSVD_VIDEO_MD5,
        "video_count": len(extracted_videos),
        "destination": str(destination),
        "completed_at_utc": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
    }

    temporary_marker = marker_path.with_suffix(".json.part")
    temporary_marker.write_text(
        json.dumps(marker, indent=2),
        encoding="utf-8",
    )
    temporary_marker.replace(marker_path)

    print(
        f"Successfully extracted and verified "
        f"{len(extracted_videos):,} MSVD videos."
    )


existing_videos = [p for p in VIDEO_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
if len(existing_videos) < MSVD_EXPECTED_VIDEOS:  # also resumes an interrupted extraction
    video_archive = locate_valid_existing_archive()
    if video_archive is None:
        video_archive = download_video_archive()
    safely_extract_msvd(video_archive, VIDEO_ROOT)
else:
    print(f"Using configured video directory: {VIDEO_ROOT} ({len(existing_videos)} files)")

video_paths = {}

for path in sorted(VIDEO_ROOT.rglob("*")):
    if (
        not path.is_file()
        or path.suffix.lower() not in VIDEO_SUFFIXES
    ):
        continue

    video_id = path.stem

    if video_id in video_paths:
        raise ValueError(
            f"Duplicate extracted video ID {video_id!r}: "
            f"{video_paths[video_id]} and {path}"
        )

    video_paths[video_id] = path

if len(video_paths) != MSVD_EXPECTED_VIDEOS:
    raise ValueError(
        f"Expected {MSVD_EXPECTED_VIDEOS} usable video paths, "
        f"found {len(video_paths)}."
    )

print(f"VIDEO_ROOT = {VIDEO_ROOT}")
print(f"Complete MSVD video set ready: {len(video_paths):,} videos")
'''),
md(r'''
## 2b. Video–Caption Coverage Audit

Only matched video IDs enter the split. Inspect the saved unmatched-ID lists before treating this dataset as a representative MSVD sample.
'''),
code(r'''
from pathlib import Path
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path(CFG.DATA_ROOT)
OUT = Path(CFG.OUTPUT_DIR)
METRICS_DIR = OUT / "results/metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

if "video_paths" not in globals() or not video_paths:
    raise RuntimeError("Run Section 2 first to index the video files.")

# Reuse the uploaded TXT, including Colab's renamed copies.
candidates = []
for directory in [Path.cwd(), Path("/content")]:
    if directory.is_dir():
        candidates.extend(directory.glob("AllVideoDescriptions*.txt"))

candidates.extend(ROOT.rglob("AllVideoDescriptions*.txt"))
candidates = sorted({
    path.resolve() for path in candidates if path.is_file()
})

configured_path = ROOT / CFG.CAPTION_FILE
if configured_path.is_file() and configured_path.suffix.lower() == ".txt":
    CAPTION_PATH = configured_path.resolve()
elif candidates:
    # Prefer the exact filename over renamed upload copies.
    exact = [p for p in candidates if p.name == "AllVideoDescriptions.txt"]
    choices = exact or candidates
    if len(choices) != 1:
        raise ValueError(
            "Multiple annotation files found. Set CFG.CAPTION_FILE "
            f"to the intended absolute path: {list(map(str, choices))}"
        )
    CAPTION_PATH = choices[0]
else:
    from google.colab import files

    print("Upload AllVideoDescriptions.txt.")
    uploaded = files.upload()
    names = [
        name for name in uploaded
        if Path(name).suffix.lower() == ".txt"
    ]
    if len(names) != 1:
        raise ValueError("Upload exactly one annotation TXT file.")

    CAPTION_PATH = ROOT / "data/extracted/annotations/AllVideoDescriptions.txt"
    CAPTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAPTION_PATH.write_bytes(uploaded[names[0]])
    CAPTION_PATH = CAPTION_PATH.resolve()

print(f"Using annotations: {CAPTION_PATH}")

# Official reconstructed format:
# video_id_start_end caption text
# Header/comment lines begin with '#'.
records = []
invalid_rows = []

with CAPTION_PATH.open(encoding="utf-8-sig") as handle:
    for line_number, line in enumerate(handle, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            video_id, caption_raw = line.split(maxsplit=1)

            # Split from the right because source YouTube IDs may contain '_'.
            source_video_id, start_text, end_text = video_id.rsplit("_", 2)
            start, end = float(start_text), float(end_text)

            if (
                not source_video_id
                or not np.isfinite(start)
                or not np.isfinite(end)
                or start < 0
                or end <= start
            ):
                raise ValueError("Invalid source ID or segment bounds")

            caption_clean = " ".join(caption_raw.split())
            if not caption_clean:
                raise ValueError("Empty caption")

            records.append({
                "video_id": video_id,
                "source_video_id": source_video_id,
                "Start": start,
                "End": end,
                "Language": "English",
                "caption_raw": caption_raw,
                "caption_clean": caption_clean,
                "caption": caption_clean,
            })
        except ValueError as exc:
            invalid_rows.append({
                "line_number": line_number,
                "text": line,
                "error": str(exc),
            })

if invalid_rows:
    invalid_path = METRICS_DIR / "invalid_annotation_rows.csv"
    pd.DataFrame(invalid_rows).to_csv(invalid_path, index=False)
    raise ValueError(
        f"{len(invalid_rows)} malformed annotation lines. "
        f"Inspect {invalid_path}."
    )

if not records:
    raise ValueError("No captions were parsed from the annotation file.")

raw_captions = pd.DataFrame(records)
captions = raw_captions.copy()

caption_id_set = set(captions["video_id"])
video_id_set = set(video_paths)

available_captions = captions.loc[
    captions["video_id"].isin(video_id_set)
].copy()

missing_video_ids = sorted(caption_id_set - video_id_set)
missing_caption_ids = sorted(video_id_set - caption_id_set)

pd.DataFrame({"video_id": missing_video_ids}).to_csv(
    METRICS_DIR / "missing_video_ids.csv", index=False
)
pd.DataFrame({"video_id": missing_caption_ids}).to_csv(
    METRICS_DIR / "missing_caption_ids.csv", index=False
)

coverage = pd.DataFrame([{
    "annotation_format": "reconstructed_english_txt",
    "raw_annotation_rows": len(raw_captions),
    "english_caption_rows": len(captions),
    "captioned_video_ids": captions["video_id"].nunique(),
    "source_video_ids": captions["source_video_id"].nunique(),
    "video_files": len(video_paths),
    "matched_video_ids": available_captions["video_id"].nunique(),
    "unmatched_caption_ids": len(missing_video_ids),
    "uncaptioned_video_files": len(missing_caption_ids),
    "matched_caption_rows": len(available_captions),
}])
coverage.to_csv(METRICS_DIR / "dataset_coverage.csv", index=False)
display(coverage)

if available_captions.empty:
    raise ValueError(
        "No annotation IDs match the video filenames. "
        "Inspect missing_video_ids.csv and missing_caption_ids.csv."
    )

# Record the annotation path actually used.
CFG.CAPTION_FILE = str(CAPTION_PATH)
(OUT / "configs").mkdir(parents=True, exist_ok=True)
(OUT / "configs/experiment_config.json").write_text(
    json.dumps(asdict(CFG), indent=2),
    encoding="utf-8",
)

captions = available_captions.reset_index(drop=True)
print(
    f"Ready: {len(captions):,} English captions for "
    f"{captions['video_id'].nunique():,} matched clips from "
    f"{captions['source_video_id'].nunique():,} source videos."
)
'''),
md(r'''
# 3. Dataset Splitting (source-video groups)
'''),
code(r'''
def split_by_video_id(frame, train_ratio=.8, val_ratio=.1, test_ratio=.1, seed=42):
    """Split source YouTube IDs, then expand to clips. Ratios apply to source groups."""
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    if not np.isfinite(ratios).all() or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1):
        raise ValueError("Split ratios must be positive and sum to 1.")
    mapping = frame[["video_id", "source_video_id"]].drop_duplicates()
    if mapping.video_id.duplicated().any(): raise ValueError("A clip maps to multiple source videos")
    sources = np.array(sorted(mapping.source_video_id.unique()))
    rng = np.random.default_rng(seed); rng.shuffle(sources)
    n_train = int(len(sources) * train_ratio); n_val = int(len(sources) * val_ratio)
    if min(n_train, n_val, len(sources)-n_train-n_val) < 1:
        raise ValueError("Too few source videos for nonempty splits with these ratios.")
    groups = {"train":set(sources[:n_train]), "validation":set(sources[n_train:n_train+n_val]),
              "test":set(sources[n_train+n_val:])}
    sets = {name:set(mapping.loc[mapping.source_video_id.isin(ids), "video_id"]) for name,ids in groups.items()}
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
    assert set.union(*sets.values()) == set(frame.video_id)
    return sets

# Only videos that can actually be evaluated enter the experiment split.
splits = split_by_video_id(available_captions, CFG.TRAIN_RATIO, CFG.VAL_RATIO, CFG.TEST_RATIO, CFG.SEED)
split_names = {"train":"train_video_ids.csv", "validation":"val_video_ids.csv", "test":"test_video_ids.csv"}
rows=[]
for name, ids in splits.items():
    pd.DataFrame({"video_id":sorted(ids)}).to_csv(OUT / split_names[name], index=False)
    rows.append({"Split":name, "Unique Videos":len(ids), "Source Videos":available_captions.loc[available_captions.video_id.isin(ids), "source_video_id"].nunique(), "Caption Records":available_captions.video_id.isin(ids).sum()})
split_table=pd.DataFrame(rows); display(split_table)
source_counts = available_captions[["video_id", "source_video_id"]].drop_duplicates().groupby("source_video_id").size()
print(f"Sources with multiple clips: {(source_counts > 1).sum()} of {len(source_counts)}")
split_metadata={"protocol":"source_video_grouped_v1", "seed":CFG.SEED,
                "ratios":[CFG.TRAIN_RATIO, CFG.VAL_RATIO, CFG.TEST_RATIO],
                "clip_ids":{k:sorted(v) for k,v in splits.items()},
                "source_ids":{k:sorted(available_captions.loc[available_captions.video_id.isin(v), "source_video_id"].unique()) for k,v in splits.items()}}
available_captions[["video_id", "source_video_id"]].drop_duplicates().to_csv(OUT / "configs/clip_source_mapping.csv", index=False)
with open(OUT / "configs/split_metadata.json", "w") as f: json.dump(split_metadata, f, indent=2)
print("Verified: train ∩ validation = train ∩ test = validation ∩ test = ∅")
'''),
md(r'''
# 4. Exploratory Data Analysis (train only)
'''),
code(r'''
eda_captions = captions.loc[captions.video_id.isin(splits["train"])].copy()
print("EDA scope: training captions only")
import re
from wordcloud import WordCloud
FIG = OUT / "results/figures"
def savefig(name):
    plt.tight_layout(); plt.savefig(FIG / name, dpi=220, bbox_inches="tight"); plt.show()
eda_captions["word_count"] = eda_captions.caption.str.findall(r"[A-Za-z']+").str.len()
words = [w.lower() for text in eda_captions.caption for w in re.findall(r"[A-Za-z']+", text)]
freq = Counter(words)
fig, ax = plt.subplots(figsize=(8,4)); sns.histplot(eda_captions.word_count, bins=30, ax=ax); ax.set(title="Caption length distribution", xlabel="Words per caption", ylabel="Caption records"); ax.grid(alpha=.25); savefig("caption_length_histogram.png")
fig, ax = plt.subplots(figsize=(7,3)); sns.boxplot(x=eda_captions.word_count, ax=ax); ax.set(title="Caption word-count boxplot", xlabel="Words per caption", ylabel="Count"); ax.grid(alpha=.25); savefig("caption_wordcount_boxplot.png")
top = pd.DataFrame(freq.most_common(20), columns=["word","count"]); fig, ax = plt.subplots(figsize=(10,5)); sns.barplot(data=top, x="count", y="word", ax=ax); ax.set(title="Top 20 words", xlabel="Frequency", ylabel="Word"); ax.grid(axis="x", alpha=.25); savefig("top_words.png")
ranked = sorted(freq.values(), reverse=True); fig, ax = plt.subplots(figsize=(7,4)); ax.loglog(range(1,len(ranked)+1), ranked); ax.set(title="Word frequency distribution", xlabel="Word rank (log)", ylabel="Frequency (log)"); ax.grid(alpha=.25); savefig("word_frequency_distribution.png")
fig, ax = plt.subplots(figsize=(9,5)); ax.imshow(WordCloud(width=1200,height=600,background_color="white").generate_from_frequencies(freq)); ax.axis("off"); ax.set_title("MSVD caption word cloud"); savefig("wordcloud.png")
counts = eda_captions.groupby("video_id").size(); fig, ax = plt.subplots(figsize=(7,4)); sns.histplot(counts, bins=30, ax=ax); ax.set(title="Captions per video", xlabel="Captions", ylabel="Videos"); ax.grid(alpha=.25); savefig("captions_per_video.png")
'''),
md(r'''
# 5. Video Preprocessing
'''),
code(r'''
def inspect_video(video_path):
    cap=cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened(): raise RuntimeError("OpenCV could not decode video")
        fps=float(cap.get(cv2.CAP_PROP_FPS)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or n <= 0: raise RuntimeError(f"invalid fps/frame_count ({fps}, {n})")
        return {"readable":True,"fps":fps,"frame_count":n,"width":w,"height":h,"duration_s":n/fps,"error":""}
    except Exception as exc: return {"readable":False,"fps":np.nan,"frame_count":np.nan,"width":np.nan,"height":np.nan,"duration_s":np.nan,"error":str(exc)}
    finally: cap.release()

def build_video_metadata(paths, run_all=False):
    iterable=list(paths.items()) if run_all else list(paths.items())[:min(25,len(paths))]
    records=[]
    for vid, path in tqdm(iterable, desc="Inspecting videos"):
        records.append({"video_id":vid,"path":str(path), **inspect_video(path)})
    meta=pd.DataFrame(records); meta.to_csv(OUT/"results/video_metadata.csv", index=False); return meta

video_metadata=build_video_metadata(video_paths, CFG.RUN_FULL_METADATA)
display(video_metadata.head()); metadata_scope=f"inspected {len(video_metadata)} of {len(video_paths)} videos"
print(f"{metadata_scope}; unreadable within inspected set: {(~video_metadata.readable).sum()}")
if not video_metadata.empty:
    for col, title, xlab in [("duration_s","Video duration distribution","Duration (seconds)"),("fps","FPS distribution","FPS"),("frame_count","Frame-count distribution","Frames")]:
        fig,ax=plt.subplots(figsize=(7,4)); sns.histplot(video_metadata.loc[video_metadata.readable,col].dropna(), bins=25, ax=ax); ax.set(title=f"{title} ({metadata_scope})",xlabel=xlab,ylabel="Videos"); ax.grid(alpha=.25); savefig(f"{col}_distribution.png")
    fig,ax=plt.subplots(figsize=(7,4)); sns.scatterplot(data=video_metadata[video_metadata.readable],x="width",y="height",ax=ax); ax.set(title=f"Resolution distribution ({metadata_scope})",xlabel="Width (pixels)",ylabel="Height (pixels)"); ax.grid(alpha=.25); savefig("resolution_distribution.png")
'''),
md(r'''
# 6. Candidate Frame Extraction
'''),
code(r'''
def extract_candidate_frames(video_path, num_frames):
    """Decode only temporally-spaced candidates; returns RGB PIL frames, indices, timestamps."""
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {video_path}")
    try:
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS))
        if total <= 0 or fps <= 0 or num_frames <= 0:
            raise ValueError(f"Invalid video metadata or candidate count: frames={total}, fps={fps}, candidates={num_frames}")
        indices=np.unique(np.linspace(0, total-1, min(num_frames,total), dtype=int))
        frames=[]; actual=[]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, bgr=cap.read()
            if ok: frames.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))); actual.append(int(idx))
        return frames, actual, [i/fps for i in actual]
    finally: cap.release()

def cached_candidates(video_id):
    # Invalidate thumbnails if a source video is replaced at the same path.
    import hashlib
    source=video_paths[video_id]; stat=source.stat()
    identity=hashlib.sha256(f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    cache=OUT / "cache" / f"{video_id}_{identity}_n{CFG.NUM_CANDIDATE_FRAMES}_s{CFG.IMAGE_SIZE}_candidates.npz"
    # Cache compact thumbnails, not thousands of source images.
    if cache.exists():
        z=np.load(cache, allow_pickle=False); return [Image.fromarray(x) for x in z["frames"]], z["indices"].tolist(), z["times"].tolist()
    frames,indices,times=extract_candidate_frames(video_paths[video_id], CFG.NUM_CANDIDATE_FRAMES)
    if not frames: raise ValueError(f"No decodable candidate frames for {video_id}")
    thumbs=np.stack([np.asarray(f.resize((CFG.IMAGE_SIZE,CFG.IMAGE_SIZE))) for f in frames])
    np.savez_compressed(cache,frames=thumbs,indices=np.array(indices),times=np.array(times))
    return [Image.fromarray(x) for x in thumbs],indices,times

def plot_timeline(indices, selected=None, title="Candidate frame locations"):
    fig,ax=plt.subplots(figsize=(11,2)); ax.scatter(indices,np.ones(len(indices)),label="Candidates",s=35)
    if selected is not None: ax.scatter(selected,np.ones(len(selected))*1.04,label="Selected",s=60,marker="*")
    ax.set(title=title,xlabel="Frame index",ylabel="Timeline"); ax.set_yticks([]); ax.grid(axis="x",alpha=.25); ax.legend(); savefig(title.lower().replace(" ","_")+".png")

sample_video_id=next(iter(sorted(splits["validation"] & set(video_paths))), None)
if sample_video_id:
    sample_frames,sample_indices,sample_times=cached_candidates(sample_video_id); plot_timeline(sample_indices, title="Candidate frame locations")
    print(sample_video_id, len(sample_frames), "candidate frames")
'''),
md(r'''
# 7. Baseline Frame Sampling
'''),
code(r'''
def uniform_selection(indices, k):
    return np.asarray(indices)[np.unique(np.linspace(0,len(indices)-1,min(k,len(indices)),dtype=int))].tolist()
def video_seed(video_id):
    import zlib  # crc32 is stable across processes; hash() is salted per run
    return CFG.SEED+zlib.crc32(str(video_id).encode())
def random_selection(indices, k, seed=42):
    rng=np.random.default_rng(seed); return sorted(rng.choice(np.asarray(indices),size=min(k,len(indices)),replace=False).tolist())
if sample_video_id:
    plot_timeline(sample_indices,uniform_selection(sample_indices,CFG.NUM_SELECTED_FRAMES),"Uniform frame sampling")
    plot_timeline(sample_indices,random_selection(sample_indices,CFG.NUM_SELECTED_FRAMES,video_seed(sample_video_id)),"Random frame sampling")
'''),
md(r'''
# 8. Motion-Based Frame Selection
'''),
code(r'''
def minmax(scores):
    scores=np.asarray(scores,dtype=float)
    return np.zeros_like(scores) if len(scores)==0 or np.ptp(scores)<1e-12 else (scores-scores.min())/np.ptp(scores)

def motion_scores(frames):
    """Downsampled Farnebäck flow gives a T4-friendly candidate-level motion proxy."""
    if len(frames)==0: return np.array([])
    gray=[cv2.resize(cv2.cvtColor(np.asarray(f),cv2.COLOR_RGB2GRAY),(160,120)) for f in frames]
    score=[0.0]
    for prev,cur in zip(gray[:-1],gray[1:]):
        flow=cv2.calcOpticalFlowFarneback(prev,cur,None,.5,3,15,3,5,1.2,0)
        score.append(float(np.linalg.norm(flow,axis=2).mean()))
    if len(score)>1: score[0]=score[1]  # frame 0 has no predecessor; reuse the first interval so it is not always the minimum
    return minmax(score)

if sample_video_id:
    sample_motion=motion_scores(sample_frames)
    pd.DataFrame({"video_id":sample_video_id,"frame_index":sample_indices,"timestamp":sample_times,"motion_score":sample_motion}).to_csv(OUT/"results/frame_scores/motion_scores.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,4)); ax.plot(sample_indices,sample_motion,marker="o"); ax.set(title="Motion score timeline",xlabel="Frame index",ylabel="Normalized motion score"); ax.grid(alpha=.25); savefig("motion_score_timeline.png")
'''),
md(r'''
# 9. Semantic Frame Selection
'''),
code(r'''
from transformers import AutoImageProcessor, AutoModel
semantic_processor=None; semantic_model=None
def load_semantic_encoder():
    global semantic_processor, semantic_model
    if semantic_model is None:
        name=CFG.SEMANTIC_MODEL_NAME  # lightweight, caption-independent frozen vision encoder
        semantic_processor=AutoImageProcessor.from_pretrained(name)
        semantic_model=AutoModel.from_pretrained(name).to(device).eval()
    return semantic_processor,semantic_model

@torch.inference_mode()
def semantic_scores(frames, batch_size=16):
    if not frames: return np.array([]), np.empty((0,0))
    processor,model=load_semantic_encoder(); vectors=[]
    for start in range(0,len(frames),batch_size):
        inputs=processor(images=frames[start:start+batch_size],return_tensors="pt").to(device)
        emb=model(**inputs).last_hidden_state[:,0].float(); vectors.append(torch.nn.functional.normalize(emb,dim=-1).cpu())
    emb=torch.cat(vectors).numpy(); novelty=np.zeros(len(emb))
    if len(emb)>1:
        novelty[1:]=1-(emb[1:]*emb[:-1]).sum(axis=1)
        novelty[0]=novelty[1]  # frame 0 has no predecessor; reuse the first interval so it is not always the minimum
    return minmax(novelty),emb

if sample_video_id:
    sample_semantic,sample_embeddings=semantic_scores(sample_frames)
    np.save(OUT/f"cache/{sample_video_id}_embeddings.npy",sample_embeddings)
    fig,ax=plt.subplots(figsize=(10,4)); ax.plot(sample_indices,sample_semantic,marker="o",color="tab:green"); ax.set(title="Semantic novelty score timeline",xlabel="Frame index",ylabel="Normalized semantic score"); ax.grid(alpha=.25); savefig("semantic_score_timeline.png")
'''),
md(r'''
# 10. Adaptive Semantic-Motion Keyframe Selection
'''),
code(r'''
def fused_scores(semantic, motion, alpha): return alpha*minmax(semantic)+(1-alpha)*minmax(motion)
if sample_video_id:
    sample_combined=fused_scores(sample_semantic,sample_motion,CFG.ALPHA)
    fig,ax=plt.subplots(figsize=(11,4)); ax.plot(sample_indices,sample_semantic,label="Semantic"); ax.plot(sample_indices,sample_motion,label="Motion"); ax.plot(sample_indices,sample_combined,label=f"Combined α={CFG.ALPHA}",linewidth=2); ax.set(title="Semantic, motion, and combined scores",xlabel="Frame index",ylabel="Normalized score"); ax.grid(alpha=.25); ax.legend(); savefig("combined_score_timeline.png")
'''),
md(r'''
# 11. Temporal Diversity Filtering
'''),
code(r'''
def select_diverse_keyframes(indices, scores, timestamps, k, min_temporal_distance):
    # Greedy temporal separation, followed by deterministic backfill for short clips.
    order=np.argsort(scores, kind="stable")[::-1]; chosen=[]
    for pos in order:
        if all(abs(timestamps[pos]-timestamps[q]) >= min_temporal_distance for q in chosen):
            chosen.append(int(pos))
        if len(chosen)==k: break
    for pos in order:
        if len(chosen)==k: break
        if int(pos) not in chosen: chosen.append(int(pos))
    chosen=sorted(chosen)
    return [indices[i] for i in chosen], [float(scores[i]) for i in chosen], [float(timestamps[i]) for i in chosen]

def select_from_candidates(frames, indices, times, method, k, alpha, seed=None):
    if not frames: raise ValueError("No decodable candidate frames")
    if k <= 0: raise ValueError("Frame budget must be positive")
    if method=="uniform": selected=uniform_selection(indices,k); score=np.zeros(len(indices))
    elif method=="random": selected=random_selection(indices,k,CFG.SEED if seed is None else seed); score=np.zeros(len(indices))
    elif method=="motion": score=motion_scores(frames)
    elif method=="semantic": score,_=semantic_scores(frames)
    elif method in {"semantic_motion","semantic_motion_diverse"}:
        motion=motion_scores(frames); semantic,_=semantic_scores(frames)
        score=fused_scores(semantic,motion,alpha)
    else: raise ValueError(f"Unknown selection method: {method}")
    if method=="semantic_motion_diverse":
        selected,_,_=select_diverse_keyframes(indices,score,times,k,CFG.MIN_TEMPORAL_DISTANCE)
    elif method not in {"uniform","random"}:
        selected=sorted(np.asarray(indices)[np.argsort(score,kind="stable")[::-1][:min(k,len(indices))]].tolist())
    take=[indices.index(i) for i in selected]
    return [frames[i] for i in take], selected, [times[i] for i in take], score

def select_frames(video_id, method, k=None, alpha=None):
    k=CFG.NUM_SELECTED_FRAMES if k is None else k
    alpha=CFG.ALPHA if alpha is None else alpha
    if k <= 0 or not 0 <= alpha <= 1: raise ValueError("k must be positive and alpha must be in [0, 1]")
    frames,indices,times=cached_candidates(video_id)
    return select_from_candidates(frames,indices,times,method,k,alpha,seed=video_seed(video_id))

if sample_video_id:
    _,chosen,_,_=select_frames(sample_video_id,"semantic_motion_diverse")
    plot_timeline(sample_indices,chosen,"Candidate versus diverse selected frames")
'''),
md(r'''
# 12. BLIP-2 Model Setup
'''),
code(r'''
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

def consensus_caption(captions):
    """Keep the per-frame caption that shares the most words with the others, so one odd frame does not set the output."""
    captions=[c.strip() for c in captions if c.strip()]
    if not captions: raise RuntimeError("BLIP-2 produced no non-empty frame caption")
    words=[set(c.lower().replace("."," ").split()) for c in captions]
    def agreement(i): return sum(len(words[i]&w)/max(1,len(words[i]|w)) for j,w in enumerate(words) if j!=i)
    return captions[max(range(len(captions)),key=agreement)]

def synchronize_blip2_processor(processor, model):
    """Keep image placeholders aligned with the loaded BLIP-2 checkpoint."""
    tokenizer=processor.tokenizer
    processor.num_query_tokens=model.config.num_query_tokens
    token=getattr(processor, "image_token", "<image>")
    token=str(token)
    image_id=tokenizer.convert_tokens_to_ids(token)
    if image_id is None or image_id == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens":[token]})
        image_id=tokenizer.convert_tokens_to_ids(token)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    configured=getattr(model.config, "image_token_index", None)
    if configured is not None and configured != image_id:
        raise RuntimeError(f"Processor/model image token mismatch: {image_id} vs {configured}")
    model.config.image_token_index=image_id
    tokenizer.padding_side="right"
    return processor

class BLIP2VideoCaptioner:
    def __init__(self, config):
        self.cfg=config; self.processor=None; self.model=None; self.loaded_source=None
    def load_model(self):
        if self.model is not None and self.loaded_source==self.cfg.MODEL_NAME: return self
        self.model=None; self.processor=None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if device=="cuda": print(f"Loading {self.cfg.MODEL_NAME} in fp16; free GPU memory: {(torch.cuda.mem_get_info()[0]/2**30):.2f} GB")
        kwargs={"torch_dtype":torch.float16 if device=="cuda" else torch.float32}
        if self.cfg.USE_4BIT:
            if device != "cuda": raise RuntimeError("4-bit loading requires a supported CUDA runtime")
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16), device_map="auto")
        self.processor=Blip2Processor.from_pretrained(self.cfg.MODEL_NAME)
        self.model=Blip2ForConditionalGeneration.from_pretrained(self.cfg.MODEL_NAME, **kwargs)
        if not self.cfg.USE_4BIT: self.model.to(device)
        synchronize_blip2_processor(self.processor,self.model)
        self.model.eval(); self.loaded_source=self.cfg.MODEL_NAME; return self
    @torch.inference_mode()
    def generate_caption(self, frames):
        """Caption every selected frame at the full BLIP-2 input size in one batch, then keep the consensus caption."""
        if not frames: raise ValueError("No decoded frames")
        images=[f.convert("RGB") for f in frames]
        inputs=self.processor(images=images,return_tensors="pt").to(device, torch.float16 if device=="cuda" else torch.float32)
        tokens=self.model.generate(**inputs,max_new_tokens=self.cfg.MAX_CAPTION_LENGTH,num_beams=self.cfg.NUM_BEAMS,
                                   eos_token_id=self.processor.tokenizer.eos_token_id,
                                   pad_token_id=self.processor.tokenizer.pad_token_id)
        return consensus_caption(self.processor.batch_decode(tokens,skip_special_tokens=True))
    def generate_batch(self, frame_batches): return [self.generate_caption(x) for x in frame_batches]
    def save_checkpoint(self, path, extra=None):
        Path(path).mkdir(parents=True,exist_ok=True); self.model.save_pretrained(path); self.processor.save_pretrained(path); json.dump(extra or {},open(Path(path)/"metadata.json","w"),indent=2)
    def load_checkpoint(self,path):
        path=Path(path)
        if not path.exists(): raise FileNotFoundError(path)
        if self.model is not None and self.loaded_source==str(path): return self
        self.model=None; self.processor=None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        is_adapter=(path/"adapter_config.json").exists()
        source=self.cfg.MODEL_NAME if is_adapter else str(path)
        self.processor=Blip2Processor.from_pretrained(path)
        kwargs={"torch_dtype":torch.float16 if device=="cuda" else torch.float32}
        if self.cfg.USE_4BIT:
            if device != "cuda": raise RuntimeError("4-bit loading requires CUDA")
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16), device_map="auto")
        base=Blip2ForConditionalGeneration.from_pretrained(source, **kwargs)
        if not self.cfg.USE_4BIT: base.to(device)
        synchronize_blip2_processor(self.processor,base)
        if is_adapter:
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(base,path).eval()
        else:
            self.model=base.eval()
        self.loaded_source=str(path)
        return self

captioner=BLIP2VideoCaptioner(CFG)
print("Configured model:",CFG.MODEL_NAME,"— download only when the next cell is run.")
'''),
md(r'''
# 13. Zero-Shot Baseline
'''),
code(r'''
def references_for(video_id): return captions.loc[captions.video_id.eq(video_id),"caption"].tolist()
def zero_shot_smoke_test(video_ids):
    captioner.load_model(); records=[]
    for vid in tqdm(list(video_ids)[:CFG.SMOKE_TEST_VIDEOS],desc="Zero-shot smoke test"):
        frames,indices,_,_=select_frames(vid,"uniform"); pred=captioner.generate_caption(frames)
        records.append({"video_id":vid,"sampling_method":"uniform","selected_frame_indices":json.dumps(indices),"caption":pred,"references":references_for(vid)})
    result=pd.DataFrame(records)
    result.to_csv(OUT/"results/generated_captions/smoke_validation.csv",index=False)
    display(result[["video_id","selected_frame_indices","caption","references"]]); return result

# Run only after paths are verified; this downloads BLIP-2 once.
LAYOUT_CAPTION=re.compile(r"\b(collages?|storyboards?|montages?|grids?|multiple images|several images|series of (?:pictures|images|photos)|set of (?:pictures|images|photos))\b",re.I)
def check_smoke_captions(predictions):
    """Fail on empty captions and on captions that describe an image layout instead of the event."""
    bad=predictions[predictions.caption.str.strip().eq("")|predictions.caption.str.contains(LAYOUT_CAPTION)]
    if len(bad): display(bad[["video_id","caption"]])
    return not predictions.empty and len(bad)==0

smoke_test_passed=False
if CFG.RUN_SMOKE_TEST:
    smoke_predictions = zero_shot_smoke_test(sorted(splits["validation"] & set(video_paths)))
    assert check_smoke_captions(smoke_predictions), "Smoke test produced empty or layout-describing captions"
    smoke_test_passed=True
else:
    print("Set CFG.RUN_SMOKE_TEST=True in Section 0 to verify captions before training.")
'''),
md(r'''
# 14. BLIP-2 Fine-Tuning

Caption targets are bounded separately from image placeholders and end in EOS. Image-placeholder and padding labels are ignored by the loss.

Training and inference use unconditional captions with no text prompt. Each arm records its training alpha, configuration, and split hash. Alpha is prespecified (`ALPHA`, default 0.5) and shared by training, validation and test for every arm; the optional validation alpha sweep is a sensitivity report only and never changes it. Use `RUN_SMOKE_TEST=True` first.
'''),
code(r'''
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

class VideoCaptionDataset(Dataset):
    def __init__(self, ids, method, training=True, alpha=None):
        self.alpha=CFG.ALPHA if alpha is None else alpha
        self.ids=[x for x in ids if x in video_paths]; self.method=method; self.training=training
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        vid=self.ids[i]; frames,_,_,_=select_frames(vid,self.method,alpha=self.alpha)
        # A random ground-truth caption per video each epoch avoids duplicate-caption row sampling.
        refs=references_for(vid); caption=random.choice(refs) if self.training else refs[0]
        # Inference captions each selected frame, so train on one selected frame: random per epoch, fixed middle frame for validation.
        frame=random.choice(frames) if self.training else frames[len(frames)//2]
        return vid, frame.convert("RGB"), caption

def caption_token_ids(tokenizer,text,max_tokens):
    """Bound caption tokens and retain EOS independently of image placeholders."""
    if max_tokens < 2: raise ValueError("MAX_TRAIN_CAPTION_TOKENS must be >= 2")
    if tokenizer.eos_token_id is None: raise RuntimeError("Caption tokenizer has no EOS token")
    ids=tokenizer.encode(str(text).strip(),add_special_tokens=False)
    while ids and ids[-1] == tokenizer.eos_token_id: ids.pop()
    return ids[:max_tokens-1]+[tokenizer.eos_token_id]

def mask_caption_labels(input_ids,attention_mask,image_token_index):
    labels=input_ids.clone()
    labels[attention_mask==0]=-100
    if image_token_index is not None: labels[input_ids==image_token_index]=-100
    return labels

def collate_caption_batch(batch):
    vids,images,texts=zip(*batch)
    tokenizer=captioner.processor.tokenizer
    image_id=captioner.model.config.image_token_index
    prefix=[image_id]*captioner.model.config.num_query_tokens
    bos=[] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    features=[{"input_ids":prefix+bos+caption_token_ids(tokenizer,text,CFG.MAX_TRAIN_CAPTION_TOKENS)} for text in texts]
    inputs=captioner.processor(images=list(images),return_tensors="pt")
    tokens=tokenizer.pad(features,padding=True,return_tensors="pt")
    inputs.update(tokens)
    inputs["labels"]=mask_caption_labels(inputs["input_ids"],inputs["attention_mask"],image_id)
    inputs["pixel_values"]=inputs["pixel_values"].to(torch.float16 if device=="cuda" else torch.float32)
    return vids,{k:v.to(device) for k,v in inputs.items()}

def configure_lora(model):
    if not CFG.USE_LORA: return model
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if CFG.USE_4BIT: model=prepare_model_for_kbit_training(model)
    for p in model.parameters(): p.requires_grad=False
    config=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=["q_proj","v_proj"],bias="none")
    model=get_peft_model(model,config); model.print_trainable_parameters(); return model

def gpu_memory_gb(): return torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0.0
def train_one_epoch(model, loader, optimizer, scheduler, scaler):
    model.train(); running=0.; optimizer.zero_grad(set_to_none=True); start=time.perf_counter()
    for step,(_,batch) in enumerate(tqdm(loader,leave=False)):
        with torch.amp.autocast("cuda", enabled=device=="cuda"):
            raw_loss=model(**batch).loss
            if not torch.isfinite(raw_loss): raise RuntimeError("Non-finite training loss; no checkpoint is valid")
            window_start=(step//CFG.GRADIENT_ACCUMULATION_STEPS)*CFG.GRADIENT_ACCUMULATION_STEPS
            window_size=min(CFG.GRADIENT_ACCUMULATION_STEPS, len(loader)-window_start)
            loss=raw_loss/window_size
        scaler.scale(loss).backward()
        if (step+1)%CFG.GRADIENT_ACCUMULATION_STEPS==0 or step+1==len(loader):
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scale_before=scaler.get_scale(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= scale_before: scheduler.step()
        running += raw_loss.item()
    return running/max(1,len(loader)),time.perf_counter()-start

@torch.no_grad()
def validate(model,loader):
    model.eval(); losses=[]
    for _,batch in loader:
        with torch.amp.autocast("cuda", enabled=device=="cuda"): losses.append(model(**batch).loss.item())
    return float(np.mean(losses)) if losses else float("nan")

def save_checkpoint(model, directory, epoch, history):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); model.save_pretrained(directory/f"epoch_{epoch}"); captioner.processor.save_pretrained(directory/f"epoch_{epoch}"); json.dump(history,open(directory/"history.json","w"),indent=2)
'''),
md(r'''
# 15. Proposed Method Fine-Tuning
'''),
code(r'''
def file_digest(path):
    import hashlib
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

FINETUNED_ARMS={"uniform_finetuned":("uniform",OUT/"checkpoints/uniform_baseline"),
                "proposed_finetuned":("semantic_motion_diverse",OUT/"checkpoints/proposed")}

def best_checkpoint_path(checkpoint_dir):
    """Best epoch directory chosen on validation loss, or None if the arm has not been fine-tuned."""
    pointer=Path(checkpoint_dir)/"best_checkpoint.json"
    if not pointer.exists(): return None
    path=Path(json.loads(pointer.read_text())["path"])
    return path if (path/"adapter_config.json").exists() or (path/"config.json").exists() else None

def checkpoint_sha256(path):
    """Hash of every file in a checkpoint directory, so a retrained epoch_N in the same place is detected."""
    import hashlib
    digest=hashlib.sha256()
    for file in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8*1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def run_finetuning(method, checkpoint_dir):
    """Same architecture/hyperparameters for uniform and adaptive control conditions."""
    if CFG.NUM_EPOCHS < 1 or CFG.BATCH_SIZE < 1 or CFG.GRADIENT_ACCUMULATION_STEPS < 1:
        raise ValueError("Epochs, batch size, and gradient accumulation must be positive.")
    if not globals().get("smoke_test_passed",False):
        raise RuntimeError("Run the validation smoke test successfully before fine-tuning.")
    if (OUT/"configs/frozen_test_config.json").exists():
        raise RuntimeError("Cannot retrain a frozen experiment. Choose a new OUTPUT_DIR.")
    seed_everything(CFG.SEED)
    training_alpha=CFG.ALPHA
    # A pointer left by an earlier run must not survive a run that saves nothing.
    (Path(checkpoint_dir)/"best_checkpoint.json").unlink(missing_ok=True)
    # Reload the same pretrained weights for each comparison arm.
    captioner.model=None; captioner.processor=None
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    captioner.load_model()
    if not CFG.USE_LORA and device=="cuda": captioner.model.float()
    model=configure_lora(captioner.model)
    captioner.model=model
    captioner.loaded_source=f"finetuning:{method}"
    train_ids=sorted(splits["train"] & set(video_paths)); val_ids=sorted(splits["validation"] & set(video_paths))
    if CFG.MAX_TRAIN_VIDEOS: train_ids=train_ids[:CFG.MAX_TRAIN_VIDEOS]; val_ids=val_ids[:max(1,CFG.MAX_TRAIN_VIDEOS//8)]
    train_loader=DataLoader(VideoCaptionDataset(train_ids,method),batch_size=CFG.BATCH_SIZE,shuffle=True,generator=torch.Generator().manual_seed(CFG.SEED),num_workers=CFG.NUM_WORKERS,collate_fn=collate_caption_batch)
    val_loader=DataLoader(VideoCaptionDataset(val_ids,method,training=False),batch_size=CFG.BATCH_SIZE,num_workers=CFG.NUM_WORKERS,collate_fn=collate_caption_batch)
    if not len(train_loader) or not len(val_loader): raise ValueError("Training and validation loaders must be nonempty")
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=CFG.LEARNING_RATE)
    steps=math.ceil(len(train_loader)/CFG.GRADIENT_ACCUMULATION_STEPS)*CFG.NUM_EPOCHS; sched=get_linear_schedule_with_warmup(opt,max(1,int(.1*steps)),max(1,steps)); scaler=torch.amp.GradScaler("cuda", enabled=device=="cuda")
    hist=[]; best=float("inf"); best_epoch=None; patience=0
    for epoch in range(1,CFG.NUM_EPOCHS+1):
        tr,seconds=train_one_epoch(model,train_loader,opt,sched,scaler); va=validate(model,val_loader); row={"epoch":epoch,"train_loss":tr,"val_loss":va,"lr":sched.get_last_lr()[0],"epoch_time_s":seconds}; hist.append(row); print(row)
        if not np.isfinite(va): raise RuntimeError(f"Validation loss is {va} for {method}; check the validation split before keeping a checkpoint.")
        # Validation loss only: the test set never influences which epoch is kept.
        if va<best: best=va; best_epoch=epoch; patience=0; save_checkpoint(model,checkpoint_dir,epoch,hist)
        else: patience+=1
        if patience>=2: print("Early stopping"); break
    # The pointer is written only after the loop finishes, so an interrupted run leaves none for a later freeze to pick up.
    if best_epoch is not None:
        (Path(checkpoint_dir)/"best_checkpoint.json").write_text(json.dumps(
            {"method":method,"training_alpha":training_alpha,"training_config":asdict(CFG),"split_sha256":file_digest(OUT/"configs/split_metadata.json"),"annotation_sha256":file_digest(CAPTION_PATH),"epoch":best_epoch,"val_loss":best,"path":str(Path(checkpoint_dir)/f"epoch_{best_epoch}")},indent=2))
    pd.DataFrame(hist).to_csv(OUT/f"results/logs/{method}_training.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4)); pd.DataFrame(hist).plot(x="epoch",y=["train_loss","val_loss"],marker="o",ax=ax); ax.set(title=f"{method}: training and validation loss",xlabel="Epoch",ylabel="Loss"); ax.grid(alpha=.25); savefig(f"{method}_training_curve.png")
    return hist

# Controlled experiment: enable CFG.RUN_FINETUNING after the zero-shot smoke test.
if CFG.RUN_FINETUNING:
    for _,(arm_method,arm_directory) in FINETUNED_ARMS.items(): run_finetuning(arm_method,arm_directory)
'''),
md(r'''
# 16. Caption Generation
'''),
code(r'''
def generate_for_ids(video_ids, method, alpha=None, checkpoint=None, run_tag="manual", budget_override=False):
    video_ids=sorted(set(video_ids))
    if not video_ids: raise ValueError("No videos requested")
    if set(video_ids) & splits["test"]:
        record=assert_test_configuration_frozen(allow_budget_override=budget_override)
        if checkpoint:
            matching=[x for x in record["finetuned_checkpoints"].values() if x and x["path"]==str(checkpoint)]
            if not matching: raise RuntimeError("Test checkpoint was not frozen")
            expected=matching[0]["training_alpha"]
            if method != matching[0]["method"]: raise RuntimeError("Test selection method differs from checkpoint training")
        else: expected=record["config"]["ALPHA"]
        actual=CFG.ALPHA if alpha is None else alpha
        if actual != expected: raise RuntimeError(f"Test alpha differs from the frozen model protocol: {actual} vs {expected}")
    if checkpoint: captioner.load_checkpoint(checkpoint)
    else: captioner.load_model()
    if method in {"semantic","semantic_motion","semantic_motion_diverse"}: load_semantic_encoder()  # model load is not selection time
    output=[]
    for vid in tqdm(list(video_ids),desc=f"Generate {method}"):
        try:
            candidates=cached_candidates(vid)  # decode or cache load stays untimed so the first method does not pay for the cache
            a=CFG.ALPHA if alpha is None else alpha
            if not 0<=a<=1: raise ValueError("alpha must be in [0, 1]")
            begin=time.perf_counter()  # inference_time_s = frame selection + captioning
            frames,indices,_,_=select_from_candidates(*candidates,method,CFG.NUM_SELECTED_FRAMES,a,seed=video_seed(vid))
            pred=captioner.generate_caption(frames)
            output.append({"video_id":vid,"sampling_method":method,"selected_frame_indices":json.dumps(indices),
                           "caption":pred,"error":None,"inference_time_s":time.perf_counter()-begin,
                           "gpu_memory_gb":gpu_memory_gb(),"frame_budget":CFG.NUM_SELECTED_FRAMES,
                           "alpha":CFG.ALPHA if alpha is None else alpha,"checkpoint":str(checkpoint or CFG.MODEL_NAME),"run_tag":run_tag})
        except Exception as exc:
            output.append({"video_id":vid,"sampling_method":method,"selected_frame_indices":None,
                           "caption":None,"error":str(exc),"inference_time_s":np.nan,
                           "gpu_memory_gb":np.nan,"frame_budget":CFG.NUM_SELECTED_FRAMES,
                           "alpha":CFG.ALPHA if alpha is None else alpha,"checkpoint":str(checkpoint or CFG.MODEL_NAME),"run_tag":run_tag})
    current=pd.DataFrame(output)
    safe_tag="".join(c if c.isalnum() or c in "-_" else "_" for c in run_tag)
    current.to_csv(OUT/f"results/generated_captions/{method}_{safe_tag}_k{CFG.NUM_SELECTED_FRAMES}_captions.csv",index=False)
    ledger_path=OUT/"results/generated_captions.csv"
    if ledger_path.exists():
        ledger=pd.read_csv(ledger_path)
        keys=["video_id","sampling_method","frame_budget","alpha","checkpoint","run_tag"]
        if all(k in ledger.columns for k in keys):
            ledger=ledger[~ledger.set_index(keys).index.isin(current.set_index(keys).index)]
            ledger=pd.concat([ledger,current],ignore_index=True)
        else:
            print("Existing caption ledger has an older schema; leaving it unchanged.")
            return current
    else: ledger=current
    ledger.to_csv(ledger_path,index=False)
    return current

# Example after freezing configuration:
# generated = generate_for_ids(sorted(splits["test"]), "semantic_motion_diverse", run_tag="test_final")
'''),
md(r'''
# 17. Automatic Evaluation

All metrics receive the same lowercase ASCII alphanumeric tokens (punctuation becomes a boundary; whitespace collapses). ROUGE uses that tokenizer without stemming; METEOR retains its standard matching. This is a documented internal protocol, **not PTB/coco-caption benchmark evaluation**. CIDEr is required, and methods are compared only on a saved common paired-ID set. Retrain and rerun after changing this protocol; old scores are not comparable.
'''),
code(r'''
import nltk
for resource in ["wordnet"]:
    try: nltk.data.find("corpora/"+resource)
    except LookupError: nltk.download(resource,quiet=True)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

import re
def normalize_metric_caption(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))

class MetricTokenizer:
    def tokenize(self, text): return normalize_metric_caption(text).split()

METRIC_NAMES=["BLEU-1","BLEU-2","BLEU-3","BLEU-4","METEOR","ROUGE-L","CIDEr"]
def evaluate_captions(predictions, references):
    # One successful prediction per video. The caller supplies a common paired ID set.
    if predictions.empty or "caption" not in predictions:
        return {k:np.nan for k in METRIC_NAMES}
    if predictions.video_id.duplicated().any(): raise ValueError("Duplicate prediction video IDs")
    ok=predictions[predictions.caption.notna() & predictions.caption.map(normalize_metric_caption).ne("")]
    pairs=[(str(r.caption),references[r.video_id]) for r in ok.itertuples()
           if r.video_id in references and references[r.video_id]]
    if not pairs: return {k:np.nan for k in METRIC_NAMES}
    pairs=[(normalize_metric_caption(p),[normalize_metric_caption(x) for x in rs]) for p,rs in pairs]
    hyp=[p.split() for p,_ in pairs]; refs=[[x.split() for x in rs] for _,rs in pairs]
    smooth=SmoothingFunction().method1
    result={f"BLEU-{n}":corpus_bleu(refs,hyp,weights=tuple([1/n]*n),smoothing_function=smooth) for n in range(1,5)}
    result["METEOR"]=float(np.mean([meteor_score([x.split() for x in rs],p.split()) for p,rs in pairs]))
    rouge=rouge_scorer.RougeScorer(["rougeL"],use_stemmer=False,tokenizer=MetricTokenizer())
    result["ROUGE-L"]=float(np.mean([max(rouge.score(x,p)["rougeL"].fmeasure for x in rs) for p,rs in pairs]))
    result["CIDEr"]=np.nan
    try:
        from pycocoevalcap.cider.cider import Cider
        gts={str(i):rs for i,(_,rs) in enumerate(pairs)}
        res={str(i):[p] for i,(p,_) in enumerate(pairs)}
        result["CIDEr"]=Cider().compute_score(gts,res)[0]
    except ImportError:
        raise RuntimeError("CIDEr requires the pinned pycocoevalcap package; fix the environment before reporting results.")
    return result

def paired_success_ids(prediction_tables, requested_ids):
    if not prediction_tables: raise ValueError("No prediction tables to pair")
    common=set(requested_ids)
    for table in prediction_tables.values():
        valid=table[table.caption.notna() & table.caption.map(normalize_metric_caption).ne("")]  # punctuation-only captions are failures, as in src
        common &= set(valid.video_id)
    if not common: raise ValueError("No common successful videos; inspect generation error tables")
    return sorted(common)

test_references={vid:references_for(vid) for vid in splits["test"]}
'''),
md(r'''
# 17b. Validation Alpha Sensitivity (Report Only) and Test Freeze
'''),
code(r'''
def plot_alpha_sensitivity(validation_ids, alphas=(0,.25,.5,.75,1.0), checkpoint=None):
    """Validation-only alpha sensitivity sweep; it never sets the test alpha. By default captions come from the zero-shot
    base model; pass a fine-tuned checkpoint (e.g. best_checkpoint_path(...)) to sweep that model instead."""
    suffix="_finetuned" if checkpoint else ""
    validation_ids=sorted(set(validation_ids) & splits["validation"])
    if not validation_ids: raise ValueError("No validation videos available")
    rows=[]; predictions={}
    refs={v:references_for(v) for v in validation_ids}
    for alpha in alphas:
        preds=generate_for_ids(validation_ids,"semantic_motion_diverse",alpha=alpha,checkpoint=checkpoint,
                               run_tag=f"validation_alpha_{alpha:g}{suffix}")
        predictions[alpha]=preds
    paired_ids=paired_success_ids(predictions,validation_ids)
    print(f"Paired validation videos across all alpha values: {len(paired_ids)}/{len(validation_ids)}")
    for alpha,preds in predictions.items():
        paired=preds[preds.video_id.isin(paired_ids)]
        rows.append({"alpha":alpha,"Requested Videos":len(validation_ids),
                     "Successful Videos":int(preds.caption.notna().sum()),"Paired Videos":len(paired_ids),
                     **evaluate_captions(paired,refs)})
    df=pd.DataFrame(rows); df.to_csv(OUT/f"results/metrics/alpha_validation{suffix}.csv",index=False)
    df.attrs["source_csv"]=f"results/metrics/alpha_validation{suffix}.csv"; df.attrs["checkpoint"]=str(checkpoint) if checkpoint else None
    for metric in ["CIDEr","BLEU-4"]:
        if df[metric].notna().any():
            fig,ax=plt.subplots(figsize=(6,4)); sns.lineplot(data=df,x="alpha",y=metric,marker="o",ax=ax)
            ax.set(title=f"Alpha sensitivity ({metric}, validation only)",xlabel="Semantic weight α",ylabel=metric)
            ax.grid(alpha=.25); savefig(f"alpha_vs_{metric.lower().replace('-','_')}.png")
    return df

def video_manifest():
    records=[]
    for video_id,path in sorted(video_paths.items()):
        stat=Path(path).stat()
        records.append({"video_id":video_id,"path":str(Path(path).resolve()),
                        "size":stat.st_size,"mtime_ns":stat.st_mtime_ns})
    return records

def video_manifest_sha256():
    import hashlib
    return hashlib.sha256(json.dumps(video_manifest(),sort_keys=True).encode()).hexdigest()

TRAINING_COMPATIBILITY_FIELDS=["MODEL_NAME","SEMANTIC_MODEL_NAME","NUM_CANDIDATE_FRAMES","NUM_SELECTED_FRAMES",
    "MIN_TEMPORAL_DISTANCE","IMAGE_SIZE","CAPTION_PROMPT","USE_LORA","USE_4BIT","SEED",
    "MAX_TRAIN_CAPTION_TOKENS","BATCH_SIZE","GRADIENT_ACCUMULATION_STEPS","LEARNING_RATE","NUM_EPOCHS","MAX_TRAIN_VIDEOS"]
FROZEN_FIELDS=["SEED","NUM_CANDIDATE_FRAMES","NUM_SELECTED_FRAMES","ALPHA","MIN_TEMPORAL_DISTANCE",
               "MODEL_NAME","SEMANTIC_MODEL_NAME","IMAGE_SIZE","MAX_CAPTION_LENGTH","NUM_BEAMS",
               "CAPTION_PROMPT","USE_LORA","USE_4BIT","FRAME_BUDGETS","DATA_ROOT","VIDEO_DIR","CAPTION_FILE"]

def freeze_test_configuration(validation_results=None):
    """Freeze the prespecified CFG.ALPHA. The validation sweep is a sensitivity report and never changes alpha,
    so every arm trains and tests at the same value."""
    if (OUT/"configs/frozen_test_config.json").exists():
        raise RuntimeError("Configuration is already frozen. Use a new OUTPUT_DIR for a new experiment.")
    chosen=CFG.ALPHA; metric="prespecified"
    # Fine-tuned checkpoints (best validation-loss epoch per arm) are frozen with the config, path and file hash;
    # None if the arm has not been trained yet.
    checkpoints={}
    for arm,(method,directory) in FINETUNED_ARMS.items():
        best=best_checkpoint_path(directory)
        metadata=json.loads((directory/"best_checkpoint.json").read_text()) if best else None
        checkpoints[arm]={"method":method,"path":str(best),"sha256":checkpoint_sha256(best),
                          "training_alpha":metadata["training_alpha"], "training_config":metadata["training_config"]} if best else None
        if best:
            mismatch=[field for field in TRAINING_COMPATIBILITY_FIELDS
                      if metadata["training_config"].get(field) != getattr(CFG,field)]
            if mismatch: raise RuntimeError(f"{arm} training configuration is incompatible: {mismatch}")
            if metadata["training_alpha"] != CFG.ALPHA: raise RuntimeError(f"{arm} was trained at alpha={metadata['training_alpha']} but ALPHA={CFG.ALPHA}; every arm must share one alpha.")
            if metadata["split_sha256"] != file_digest(OUT/"configs/split_metadata.json"):
                raise RuntimeError(f"{arm} was trained on different splits; retrain in a new OUTPUT_DIR.")
            if metadata.get("annotation_sha256") != file_digest(CAPTION_PATH):
                raise RuntimeError(f"{arm} training annotations changed; retrain in a new OUTPUT_DIR.")
    CFG.ALPHA=chosen
    (OUT/"configs/frozen_video_manifest.json").write_text(json.dumps(video_manifest(),indent=2))
    record={"config":asdict(CFG),"selection_metric":metric,
            "validation_results":validation_results.attrs.get("source_csv","results/metrics/alpha_validation.csv") if validation_results is not None else None,
            "alpha_sweep_checkpoint":validation_results.attrs.get("checkpoint") if validation_results is not None else None,
            "alpha_sweep_csv_sha256":{p.name:file_digest(p) for p in sorted((OUT/"results/metrics").glob("alpha_validation*.csv"))},  # shows whether a sweep existed before alpha was frozen
            "finetuned_checkpoints":checkpoints,"split_sha256":file_digest(OUT/"configs/split_metadata.json"),
            "annotation_sha256":file_digest(CAPTION_PATH),"video_manifest_sha256":video_manifest_sha256(),"metric_protocol":"lowercase_ascii_alphanumeric_v1",
            "run_id":RUN_ID}
    with open(OUT/"configs/frozen_test_config.json","w") as f: json.dump(record,f,indent=2)
    print(f"Frozen alpha={chosen:g} using {metric}; test evaluation may now run.")
    return record

def assert_test_configuration_frozen(allow_budget_override=False):
    path=OUT/"configs/frozen_test_config.json"
    if not path.exists(): raise RuntimeError("Freeze the prespecified configuration before test evaluation.")
    record=json.loads(path.read_text()); frozen=record["config"]
    current=json.loads(json.dumps(asdict(CFG)))
    changed=[field for field in FROZEN_FIELDS if frozen[field] != current[field]
             and not (allow_budget_override and field == "NUM_SELECTED_FRAMES")]
    if allow_budget_override and CFG.NUM_SELECTED_FRAMES not in CFG.FRAME_BUDGETS:
        raise RuntimeError("Frame budget was not prespecified")
    if changed: raise RuntimeError(f"Test configuration changed after freeze: {changed}")
    if record["split_sha256"] != file_digest(OUT/"configs/split_metadata.json"):
        raise RuntimeError("Split metadata changed after freeze")
    if record["annotation_sha256"] != file_digest(CAPTION_PATH):
        raise RuntimeError("Annotations changed after freeze")
    if record["video_manifest_sha256"] != video_manifest_sha256():
        raise RuntimeError("Video paths, sizes or modification times changed after freeze")
    for arm, checkpoint in record["finetuned_checkpoints"].items():
        if checkpoint:
            directory=FINETUNED_ARMS[arm][1]; current_path=best_checkpoint_path(directory)
            if current_path is None or str(current_path)!=checkpoint["path"] or checkpoint_sha256(current_path)!=checkpoint["sha256"]:
                raise RuntimeError(f"Frozen checkpoint changed or is missing: {arm}")
    return record

# Optional validation sweep (sensitivity report only); alpha is always the prespecified CFG.ALPHA.
test_requested=any([CFG.RUN_TEST_ABLATIONS,CFG.RUN_FRAME_BUDGET_STUDY,
                    CFG.RUN_EFFICIENCY_ANALYSIS,CFG.RUN_QUALITATIVE_ANALYSIS])
freeze_path=OUT/"configs/frozen_test_config.json"
if CFG.RUN_VALIDATION_ALPHA_SWEEP:
    if freeze_path.exists(): raise RuntimeError("Cannot run the alpha sweep on a frozen experiment; use a new OUTPUT_DIR")
    alpha_validation=plot_alpha_sensitivity(sorted(splits["validation"]))
if test_requested:
    if not globals().get("smoke_test_passed",False):
        raise RuntimeError("Run the validation smoke test before test experiments")
    if freeze_path.exists():
        assert_test_configuration_frozen()
    else:
        freeze_test_configuration(alpha_validation if CFG.RUN_VALIDATION_ALPHA_SWEEP else None)
else:
    print("Test experiments disabled; no test metrics will be produced.")
'''),
md(r'''
# 18. Ablation Study
'''),
code(r'''
METHODS=["uniform","random","motion","semantic","semantic_motion","semantic_motion_diverse"]
def frozen_finetuned_checkpoints(arms=None):
    """Checkpoints frozen with the test configuration. Raises if one is missing or its files changed after the freeze."""
    arms=arms or FINETUNED_ARMS
    frozen=json.loads((OUT/"configs/frozen_test_config.json").read_text()).get("finetuned_checkpoints") or {}
    checkpoints={}
    for arm,(method,directory) in arms.items():
        record=frozen.get(arm); current=best_checkpoint_path(directory)
        if not record:
            raise RuntimeError(f"{arm} was not frozen; fine-tune, then re-run freeze_test_configuration.")
        if current is None:
            raise RuntimeError(f"The frozen checkpoint for {arm} is missing (its best_checkpoint.json or epoch folder was removed): {record['path']}")
        if str(current)!=record["path"] or checkpoint_sha256(current)!=record["sha256"]:
            raise RuntimeError(f"Checkpoint for {arm} changed after the freeze: {current}")
        checkpoints[arm]=(method,current)
    return checkpoints

def run_ablation(ids, methods=METHODS, include_finetuned=False):
    """Zero-shot selection methods and, optionally, the frozen fine-tuned arms, all scored on one paired test-ID set."""
    assert_test_configuration_frozen()
    ids=sorted(set(ids) & splits["test"])
    # Check the fine-tuned checkpoints before spending GPU time on generation.
    finetuned=frozen_finetuned_checkpoints() if include_finetuned else {}
    predictions={}; sources={}
    for method in methods:
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        predictions[method]=generate_for_ids(ids,method,run_tag="test_ablation"); sources[method]=CFG.MODEL_NAME
    for arm,(method,checkpoint) in finetuned.items():
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        training_alpha=json.loads((OUT/"configs/frozen_test_config.json").read_text())["finetuned_checkpoints"][arm]["training_alpha"]
        predictions[arm]=generate_for_ids(ids,method,alpha=training_alpha,checkpoint=checkpoint,run_tag="test_finetuned"); sources[arm]=str(checkpoint)
    paired_ids=paired_success_ids(predictions,ids)
    pd.DataFrame({"video_id":paired_ids}).to_csv(OUT/"results/metrics/ablation_paired_ids.csv",index=False)
    print(f"Paired test videos: {len(paired_ids)}/{len(ids)}. Failures remain in saved prediction tables.")
    rows=[]
    for name,preds in predictions.items():
        paired=preds[preds.video_id.isin(paired_ids)]
        metric=evaluate_captions(paired,test_references)
        rows.append({"Method":name,"Frames":CFG.NUM_SELECTED_FRAMES,"Requested Videos":len(ids),
                     "Successful Videos":int(preds.caption.notna().sum()),"Paired Videos":len(paired_ids),
                     **metric,"Inference Time":paired.inference_time_s.mean(),
                     "GPU Memory":paired.gpu_memory_gb.max(),"Checkpoint":sources[name],"Selection Alpha":float(preds.alpha.iloc[0]),"Evaluation Set":"ablation_paired"})
    table=pd.DataFrame(rows)
    table.to_csv(OUT/"results/metrics/ablation_metrics.csv",index=False)
    table.to_csv(OUT/"results/metrics.csv",index=False)
    return table

def finetuned_arms_frozen():
    """True if the frozen test configuration names any fine-tuned checkpoint. The record, not the current pointers, decides."""
    path=OUT/"configs/frozen_test_config.json"
    if not path.exists(): return False
    return any(v for v in (json.loads(path.read_text()).get("finetuned_checkpoints") or {}).values())

if CFG.RUN_TEST_ABLATIONS:
    include_finetuned=finetuned_arms_frozen()
    if not include_finetuned: print("No fine-tuned checkpoints were frozen; the ablation covers the zero-shot methods only.")
    ablation=run_ablation(sorted(splits["test"]),include_finetuned=include_finetuned); display(ablation)
    fig,ax=plt.subplots(figsize=(10,4)); sns.barplot(data=ablation,x="Method",y="BLEU-4",ax=ax)
    ax.tick_params(axis="x",rotation=30); ax.set(title="Ablation comparison",xlabel="Method",ylabel="BLEU-4")
    ax.grid(axis="y",alpha=.25); savefig("ablation_comparison.png")
'''),
md(r'''
# 19. Frame Budget Study

This is a **frame-budget study**: every selected frame is captioned at the full 224×224 BLIP-2 input, so increasing K adds BLIP-2 passes but does not shrink any frame. Frame count is therefore not confounded with resolution. Budget results use their own common paired set and stay separate from the main ablation table.
'''),
code(r'''
def frame_budget_study(ids, budgets=None):
    assert_test_configuration_frozen()
    ids=sorted(set(ids) & splits["test"])
    budgets=CFG.FRAME_BUDGETS if budgets is None else tuple(budgets)
    if budgets != tuple(CFG.FRAME_BUDGETS): raise ValueError("Budgets must match the prespecified frozen FRAME_BUDGETS")
    if not budgets or any(k <= 0 or k > CFG.NUM_CANDIDATE_FRAMES for k in budgets): raise ValueError("Invalid frame budget")
    original_budget=CFG.NUM_SELECTED_FRAMES
    predictions={}
    try:
        for k in budgets:
            CFG.NUM_SELECTED_FRAMES=k
            for method in ["uniform","semantic_motion_diverse"]:
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                predictions[(k,method)]=generate_for_ids(ids,method,run_tag=f"test_budget_{k}",budget_override=True)
    finally: CFG.NUM_SELECTED_FRAMES=original_budget
    paired_ids=paired_success_ids(predictions,ids)
    pd.DataFrame({"video_id":paired_ids}).to_csv(OUT/"results/metrics/budget_paired_ids.csv",index=False)
    print(f"Paired test videos across all budgets: {len(paired_ids)}/{len(ids)}")
    records=[]
    for (k,method),preds in predictions.items():
        paired=preds[preds.video_id.isin(paired_ids)]
        metrics=evaluate_captions(paired,test_references)
        records.append({"Method":method,"Frames":k,"Requested Videos":len(ids),
                        "Successful Videos":int(preds.caption.notna().sum()),"Paired Videos":len(paired_ids),
                        **metrics,"Inference Time":paired.inference_time_s.mean(),
                        "GPU Memory":paired.gpu_memory_gb.max(),"Evaluation Set":"budget_paired"})
    df=pd.DataFrame(records); df.to_csv(OUT/"results/metrics/frame_budget.csv",index=False)
    return df
if CFG.RUN_FRAME_BUDGET_STUDY:
    budget_results=frame_budget_study(sorted(splits["test"]))
    for y,title in [("CIDEr","Frame count versus CIDEr"),("Inference Time","Frame count versus inference time"),("GPU Memory","Frame count versus GPU memory")]:
        if budget_results[y].notna().any():
            fig,ax=plt.subplots(figsize=(7,4)); sns.lineplot(data=budget_results,x="Frames",y=y,hue="Method",marker="o",ax=ax)
            ax.set(title=title,xlabel="Selected frames",ylabel=y); ax.grid(alpha=.25)
            savefig(title.lower().replace(" ","_")+".png")
'''),
md(r'''
# 20. Efficiency Analysis
'''),
code(r'''
def efficiency_row(video_id, method):
    if video_id in splits["test"]: assert_test_configuration_frozen()
    captioner.load_model()
    if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    # Cold decode plus method-specific scoring and one batched BLIP-2 call per video (K frames).
    start=time.perf_counter()
    frames,indices,times=extract_candidate_frames(video_paths[video_id],CFG.NUM_CANDIDATE_FRAMES)
    frames=[f.resize((CFG.IMAGE_SIZE,CFG.IMAGE_SIZE)) for f in frames]
    extraction=time.perf_counter()-start
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start=time.perf_counter()
    selected,_,_,_=select_from_candidates(frames,indices,times,method,CFG.NUM_SELECTED_FRAMES,CFG.ALPHA,seed=video_seed(video_id))
    if torch.cuda.is_available(): torch.cuda.synchronize()
    selection=time.perf_counter()-start
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start=time.perf_counter()
    _=captioner.generate_caption(selected)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    caption_time=time.perf_counter()-start
    return {"Method":method,"Candidate Frames":len(frames),"Frames":len(selected),
            "Frame Reduction":1-len(selected)/len(frames),"Frame Extraction Time":extraction,
            "Selection Time":selection,"BLIP-2 Inference Time":caption_time,
            "Total Inference Time":extraction+selection+caption_time,"GPU Memory":gpu_memory_gb()}

def plot_efficiency(efficiency):
    fig,ax=plt.subplots(figsize=(7,4))
    sns.barplot(data=efficiency,x="Method",y="Total Inference Time",ax=ax)
    ax.set(title="End-to-end runtime per video",xlabel="Method",ylabel="Seconds")
    ax.grid(axis="y",alpha=.25); savefig("end_to_end_runtime_comparison.png")
    return efficiency

if CFG.RUN_EFFICIENCY_ANALYSIS:
    captioner.load_model()
    # Warm both methods on validation, then measure test videos.
    warmup=next(iter(sorted(splits["validation"])))
    for method in ["uniform","semantic_motion_diverse"]: efficiency_row(warmup,method)
    efficiency=pd.DataFrame([efficiency_row(v,m) for v in sorted(splits["test"])[:10]
                             for m in ["uniform","semantic_motion_diverse"]])
    efficiency.to_csv(OUT/"results/metrics/efficiency.csv",index=False)
    display(plot_efficiency(efficiency))
'''),
md(r'''
### Interpreting runtime and frame reduction

All methods use the same selected-frame budget and make **one batched BLIP-2 call per video**, captioning each of the K selected frames at full resolution. Captioning cost therefore grows with K and is equal across methods at a fixed budget. Semantic scoring adds DINOv2 work and may make the proposed method slower; the research question is whether caption quality improves enough to justify that cost. Use measured end-to-end time and paired caption metrics for the tradeoff. Repeat timings after model warm-up and report hardware, video count, and variability before claiming an efficiency gain.
'''),
md(r'''
# 21. Qualitative Analysis
'''),
code(r'''
def qualitative_analysis(test_ids, n=5):
    assert_test_configuration_frozen()
    test_ids=set(test_ids) & splits["test"]
    if not test_ids: raise ValueError("No test videos requested")
    ids=np.random.default_rng(CFG.SEED).choice(sorted(test_ids),size=min(n,len(test_ids)),replace=False)
    captioner.load_model(); records=[]
    for vid in ids:
        uni,ui,_,_=select_frames(vid,"uniform"); prop,pi,_,_=select_frames(vid,"semantic_motion_diverse")
        fig,axes=plt.subplots(2,max(len(uni),len(prop)),figsize=(16,4),squeeze=False)
        for row,fs,title in zip(axes,[uni,prop],["Uniform","Adaptive semantic-motion"]):
            for ax in row: ax.axis("off")
            for ax,fr in zip(row,fs): ax.imshow(fr)
            row[0].text(-0.06,0.5,title,transform=row[0].transAxes,rotation=90,ha="right",va="center",fontsize=11)  # axis("off") hides set_ylabel
        fig.suptitle(f"{vid} | References: {' | '.join(references_for(vid)[:2])}",fontsize=10); plt.tight_layout(rect=(0.04,0,1,0.95)); plt.savefig(OUT/f"results/qualitative/{vid}.png",dpi=180); plt.show()
        records.append({"video_id":vid,"references":references_for(vid),"uniform_caption":captioner.generate_caption(uni),"proposed_caption":captioner.generate_caption(prop),"uniform_indices":ui,"proposed_indices":pi})
    return pd.DataFrame(records)
if CFG.RUN_QUALITATIVE_ANALYSIS:
    qualitative=qualitative_analysis(splits["test"] & set(video_paths))
    qualitative.to_csv(OUT/"results/qualitative/caption_comparison.csv",index=False)
'''),
md(r'''
# 22. Final Research Results
'''),
code(r'''
def final_results_table(ablation_table=None, budget_table=None):
    # Budget results have a different paired set and belong in a separate file.
    if budget_table is not None:
        budget_table.to_csv(OUT/"results/frame_budget_results.csv",index=False)
    if ablation_table is None or ablation_table.empty:
        print("No frozen test ablation has been run; no final results are written.")
        return pd.DataFrame()
    result=ablation_table.rename(columns={"Frames":"Frame Count"}).copy()
    result.to_csv(OUT/"results/final_results.csv",index=False)
    return result
workflow_status={
    "run_id":RUN_ID,
    "smoke_test_passed":bool(globals().get("smoke_test_passed",False)),
    "finetuned_arms":{arm:str(best_checkpoint_path(directory)) if best_checkpoint_path(directory) else None
                      for arm,(_,directory) in FINETUNED_ARMS.items()},
    "test_configuration_frozen":(OUT/"configs/frozen_test_config.json").exists(),
    "ablation_completed":"ablation" in globals(),
    "budget_study_completed":"budget_results" in globals(),
}
(OUT/"configs/workflow_status.json").write_text(json.dumps(workflow_status,indent=2))
print(json.dumps(workflow_status,indent=2))
final_results=final_results_table(globals().get("ablation"),globals().get("budget_results")); display(final_results)
'''),
md(r'''
# 23. Save Artifacts
'''),
code(r'''
# Config/splits/metadata, metrics, figures, generated captions, score files, qualitative figures,
# logs, checkpoints, and final_results.csv are written under OUT. Record the final runtime config.
with open(OUT/"configs/final_runtime_config.json","w") as f: json.dump(asdict(CFG),f,indent=2)
print("Artifacts saved under:",OUT)
for folder in ["results/figures","results/metrics","results/generated_captions","results/frame_scores","results/qualitative","results/logs","checkpoints/uniform_baseline","checkpoints/proposed","configs"]:
    print(" -",OUT/folder)
'''),
code(r'''
from pathlib import Path
from dataclasses import asdict, is_dataclass
import datetime as dt
import hashlib
import json
import shutil

OUT = Path(OUT)
OUT.mkdir(parents=True, exist_ok=True)

expected_directories = [
    "results/figures",
    "results/metrics",
    "results/generated_captions",
    "results/frame_scores",
    "results/qualitative",
    "results/logs",
    "checkpoints/uniform_baseline",
    "checkpoints/proposed",
    "configs",
    "exports",
]

for relative_directory in expected_directories:
    (OUT / relative_directory).mkdir(
        parents=True,
        exist_ok=True,
    )

# Save the final runtime configuration.
if is_dataclass(CFG):
    final_config = asdict(CFG)
elif hasattr(CFG, "__dict__"):
    final_config = vars(CFG)
else:
    final_config = dict(CFG)

config_path = OUT / "configs/final_runtime_config.json"
config_path.write_text(
    json.dumps(
        final_config,
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# Build a reproducibility manifest before creating the ZIP.
def keep_in_zip(path):
    """Leave out the regenerable candidate cache and every checkpoint epoch except the best one per arm."""
    parts = path.relative_to(OUT).parts
    if parts[0] in ("exports", "cache"):
        return False
    if parts[0] == "checkpoints" and len(parts) > 2 and parts[2].startswith("epoch_"):
        pointer = OUT / "checkpoints" / parts[1] / "best_checkpoint.json"
        best = json.loads(pointer.read_text())["epoch"] if pointer.exists() else None
        return parts[2] == f"epoch_{best}"
    return True


artifact_files = sorted(
    path
    for path in OUT.rglob("*")
    if path.is_file()
    and keep_in_zip(path)
    and path != OUT / "configs/artifact_manifest.json"
)

manifest = {
    "created_at_utc": dt.datetime.now(
        dt.timezone.utc
    ).isoformat(),
    "artifact_root": str(OUT),
    "run_id": globals().get("RUN_ID"),
    "run_mode": globals().get("MODE"),
    "file_count": len(artifact_files),
    "total_bytes": int(sum(
        path.stat().st_size
        for path in artifact_files
    )),
    "files": [
        {
            "path": str(path.relative_to(OUT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in artifact_files
    ],
}

manifest_path = OUT / "configs/artifact_manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

timestamp = dt.datetime.now(
    dt.timezone.utc
).strftime("%Y%m%dT%H%M%SZ")

temporary_archive_base = (
    OUT.parent
    / f"{OUT.name}_{timestamp}"
)

import zipfile
temporary_zip = temporary_archive_base.with_suffix(".zip")
with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in [*artifact_files, manifest_path]:
        archive.write(path, arcname=str(path.relative_to(OUT)))

final_zip = (
    OUT
    / "exports"
    / f"video_captioning_research_artifacts_{timestamp}.zip"
)

shutil.move(
    str(temporary_zip),
    str(final_zip),
)

print(f"Artifact root: {OUT}")
print(f"Files included: {manifest['file_count']:,}")
print(
    f"Uncompressed size: "
    f"{manifest['total_bytes'] / 2**20:.2f} MiB"
)
print(
    f"ZIP size: {final_zip.stat().st_size / 2**20:.2f} MiB"
)
print(f"Manifest: {manifest_path}")
print(f"Downloadable ZIP: {final_zip}")

# Set to True only when you are ready for the browser download.
DOWNLOAD_NOW = False

if DOWNLOAD_NOW:
    from google.colab import files
    files.download(str(final_zip))
else:
    print(
        "Set DOWNLOAD_NOW=True and rerun this cell to download the ZIP."
    )
'''),
code(r'''
from pathlib import Path
import pandas as pd
from IPython.display import display

OUT = Path(CFG.OUTPUT_DIR)

figure_directories = [
    OUT / "results/figures",
    OUT / "results/qualitative",
]

supported_extensions = {
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
}

graph_files = sorted({
    path
    for directory in figure_directories
    if directory.exists()
    for path in directory.rglob("*")
    if path.is_file()
    and path.suffix.lower() in supported_extensions
})

graph_inventory = pd.DataFrame([
    {
        "number": index,
        "filename": path.name,
        "type": path.suffix.lower(),
        "size_kb": round(path.stat().st_size / 1024, 2),
        "relative_path": str(path.relative_to(OUT)),
        "absolute_path": str(path),
    }
    for index, path in enumerate(graph_files, start=1)
])

if graph_inventory.empty:
    print(
        "No saved graphs were found. Expected them under:\n"
        f"  {OUT / 'results/figures'}\n"
        f"  {OUT / 'results/qualitative'}"
    )
else:
    print(f"Found {len(graph_files)} saved graph/figure files.")
    display(graph_inventory)
'''),
code(r'''
from IPython.display import (
    display,
    Image as IPythonImage,
    SVG,
    Markdown,
)

if "graph_files" not in globals():
    raise RuntimeError(
        "Run the graph-discovery cell first."
    )

if not graph_files:
    print("No graphs are available to display.")

for index, path in enumerate(graph_files, start=1):
    relative_path = path.relative_to(OUT)

    display(Markdown(
        f"### {index}. {path.stem.replace('_', ' ').title()}\n"
        f"`{relative_path}`"
    ))

    suffix = path.suffix.lower()

    if suffix in {".png", ".jpg", ".jpeg"}:
        display(IPythonImage(
            filename=str(path),
            width=1000,
        ))

    elif suffix == ".svg":
        display(SVG(filename=str(path)))

    elif suffix == ".pdf":
        display(Markdown(
            f"PDF figure saved at `{path}`. "
            "Open it from the Colab Files panel."
        ))

    display(Markdown("---"))
'''),
code(r'''
from PIL import Image
import matplotlib.pyplot as plt
import math

raster_graphs = [
    path
    for path in graph_files
    if path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
    }
]

if not raster_graphs:
    print("No PNG/JPG figures are available for the gallery.")

else:
    images_per_page = 6
    columns = 2

    for page_start in range(
        0,
        len(raster_graphs),
        images_per_page,
    ):
        page_files = raster_graphs[
            page_start:page_start + images_per_page
        ]

        rows = math.ceil(len(page_files) / columns)

        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(16, 6 * rows),
            squeeze=False,
            constrained_layout=True,
        )

        for axis, path in zip(
            axes.flat,
            page_files,
        ):
            with Image.open(path) as image:
                axis.imshow(image.convert("RGB"))

            axis.set_title(
                path.stem.replace("_", " ").title(),
                fontsize=11,
                fontweight="bold",
            )
            axis.axis("off")

        for axis in axes.flat[len(page_files):]:
            axis.axis("off")

        figure.suptitle(
            (
                f"Saved Research Figures "
                f"{page_start + 1}–"
                f"{page_start + len(page_files)}"
            ),
            fontsize=16,
            fontweight="bold",
        )

        plt.show()
        plt.close(figure)
'''),
md(r'''
# 24. Conclusions and Research Interpretation
'''),
code(r'''
print("""Interpret results only after a single frozen-configuration test pass.
- Compare adaptive selection with uniform at equal frame budgets; report measured efficiency rather than inferred savings.
- Use the recorded normalized metric protocol; these scores are not published MSVD benchmark scores.
- Report the prespecified alpha, seed, splits, model checkpoint, and all failed/corrupted videos from saved artifacts.
- Do not claim a method wins from qualitative examples alone; those examples are fixed-seed random test samples.""")
''')
]

for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:03d}"  # nbformat 4.5 requires stable cell ids
assert all(not c["outputs"] and c["execution_count"] is None for c in cells if c["cell_type"] == "code"), "run outputs must not enter the notebook"

notebook = {
    "cells": cells,
    "metadata": {"accelerator": "GPU",
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10"},
                 "colab": {"gpuType": "T4", "provenance": []}},
    "nbformat": 4, "nbformat_minor": 5,
}
output_path = Path(__file__).resolve().parents[1] / "notebooks" / "video_captioning_research.ipynb"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Wrote {output_path} with {len(cells)} cells")
