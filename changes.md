# Changes Required

Date: 2026-10-10
Basis: review of the full repository, plus the findings in `TODO.md`.

This document lists what should change, where, and what the change is. Nothing here has been applied yet.

Labels:
- **Verified**: confirmed by running a command (tests, grep, or notebook regeneration).
- **By reading**: found by reading the code; not yet reproduced by running it.

---

## 1. Bugs

### 1.1 Fine-tuning crashes on the second training video (By reading)

- **File:** `src/video_captioning/training/finetuning.py`, line 79
- **Current:**
  ```python
  selection, self.semantic_runtime = select_frames_from_video(
      video_paths[video_id], config, self.semantic_runtime
  )
  ```
  The third positional argument is `method` in `select_frames_from_video` (`src/video_captioning/keyframes/pipeline.py`, lines 25-29), not `semantic_runtime`.
- **Change:** pass the runtime by keyword:
  ```python
  selection, self.semantic_runtime = select_frames_from_video(
      video_paths[video_id], config, semantic_runtime=self.semantic_runtime
  )
  ```
- **Why:** after the first video, `method` becomes a tuple and `select_keyframe_positions` raises `Unknown method`.
- **Test to add:** call the dataset `__getitem__` for two videos in a row with a stubbed encoder.

### 1.2 LoRA checkpoints cannot be loaded by the inference CLI (By reading)

- **Files:**
  - `src/video_captioning/training/finetuning.py`, lines 140-141: saves the PEFT model with `model.save_pretrained(...)`, which writes adapter weights only.
  - `src/video_captioning/models/captioner.py`, lines 79-84: `load_checkpoint` loads the directory as a full `Blip2ForConditionalGeneration`.
- **Change:** choose one approach and make both sides match:
  - (a) Keep adapter-only saving. Change `load_checkpoint` to load the base model from `config.model_name`, then call `PeftModel.from_pretrained(base, path)`.
  - (b) Save a merged model after training with `merge_and_unload()`.
- **Document** the choice in `README.md` (Inference section).
- **Affects:** `scripts/inference.py --checkpoint`.

---

## 2. Notebook and documentation consistency

### 2.1 README links to a notebook filename that does not exist (Verified)

- **File:** `README.md`, lines 7, 11, 114, and 141.
- **Current:** `notebooks/video_captioning_research.ipynb`.
- **Actual file:** `notebooks/video_captioning_research-4.ipynb`.
- **Change:** rename the notebook to `video_captioning_research.ipynb` (the name `build_notebook.py` writes), or update the four links. The `-4` suffix looks like a download-duplicate name.

### 2.2 Notebook has drifted from its generator (Verified)

- **Files:**
  - `scripts/build_notebook.py`: 27 section headings, 54 cells.
  - `notebooks/video_captioning_research-4.ipynb`: 31 headings, 65 cells.
- **Differences:** the committed notebook adds sections `0b` (Architecture Diagram), `1b` (Dataset Acquisition), `2b` (Video-Caption Coverage Audit), `17b` (Validation-Only Alpha Selection and Test Freeze), and an "Interpreting runtime and frame reduction" subsection. Section numbering also differs.
- **Risk:** re-running `build_notebook.py` would silently discard the hand edits. The builder also writes to the non-suffixed filename (line 653).
- **Change:** pick one source of truth.
  - Option A (recommended for reproducibility): port the notebook edits into `build_notebook.py`, regenerate, and stop editing the `.ipynb` by hand.
  - Option B: keep the notebook as the source, and delete or archive `build_notebook.py`.

### 2.3 README documents environment variables that the code ignores (Verified)

- **File:** `README.md`, lines 183, 197, and 292-294.
- **Current:** says `DATA_ROOT` / `MSVD_DATA_ROOT`, `VIDEO_DIR` / `MSVD_VIDEO_DIR`, and `CAPTION_FILE` / `MSVD_CAPTION_FILE` configure the project.
- **Reality:** only the notebook's Section 0 code reads them (together with `OUTPUT_DIR`). That code is embedded as a string in `scripts/build_notebook.py` (lines 36-39); the builder script itself does not read them. `ExperimentConfig` (`src/video_captioning/utils/config.py`) and the CLI scripts read YAML only.
- **Change:** either add env-var overrides in `load_config`, or change the README to say these variables apply only to the notebook.

### 2.4 README roadmap says to add a license, but one exists (Verified)

- **File:** `README.md`, line 357.
- **Change:** remove the roadmap line. `LICENSE` (MIT) exists and the License section cites it.

### 2.5 README claims a notebook syntax check that does not exist (Verified)

- **File:** `README.md:328`, in the list under "The notebook additionally performs practical checks" ("syntax-safe notebook generation from `build_notebook.py`"). (Corrected: this was first described as a CI claim.)
- **Reality:** neither `scripts/build_notebook.py`, the committed notebook, nor `.github/workflows/python-package.yml` contains any `compile`/`ast.parse` syntax check. CI's blocking flake8 E9 step parses `build_notebook.py` itself, but not the cell code it generates.
- **Change:** remove the claim from the README, or add a real check. A CI step can't write the notebook to a temp path until the builder accepts an `--output` argument, because the path is hardcoded at `build_notebook.py:653`.

### 2.6 README Project Structure is out of date (Verified)

- **File:** `README.md`, Project Structure block.
- **Change:**
  - Replace `video_captioning_research.ipynb` with the actual notebook name (see 2.1).
  - Add `coursework/` and `.github/`.
  - Note that `research/READ-format.md` is a generic template (see 5.2).

### 2.7 Hardcoded Colab paths in notebook defaults (Verified)

- **File:** `scripts/build_notebook.py`, line 36 (`/content/msvd`), line 39 (`/content/video_captioning_outputs`), and line 85 (`/content/drive`, optional mount). (Corrected: line 85 was missing.) The committed notebook adds more in Section 2 (archive search under `/content`), Section 2b (fallback search paths), and Section 23 (`OUT` set directly).
- **Conflict:** the README's Contributing section says to avoid hardcoding a local path.
- **Change:** keep the Colab defaults but document them as Colab defaults, or make Section 0 read from one variable with a clear default.

---

## 3. Code quality and dependencies

### 3.1 Deprecated AMP API in fine-tuning (Verified)

- **File:** `src/video_captioning/training/finetuning.py`, lines 103, 112, and 128.
- **Current:** `torch.cuda.amp.GradScaler(...)` and `torch.cuda.amp.autocast(...)`.
- **Change:** `torch.amp.GradScaler("cuda", enabled=...)` and `torch.autocast("cuda", enabled=...)` (same class as `torch.amp.autocast`; `torch.autocast` is the name the PyTorch 2.3 docs use).

### 3.2 Duplicate `discover_caption_file` (Verified)

- **Files:** `src/video_captioning/data/dataset.py` (line 16) and `scripts/build_notebook.py` (line 119).
- **Reality:** the library version is exported from `data/__init__.py`, but no script calls it.
- **Change:** either call the library function from `scripts/train.py` and `scripts/evaluate.py`, or delete it from the library and keep the notebook copy.

### 3.3 Invalid JSON for missing CIDEr (Verified)

- **File:** `scripts/evaluate.py`, line 35: `json.dumps(metrics, indent=2, allow_nan=True)`. (Corrected: line was missing.)
- **Problem:** when CIDEr is unavailable it is `NaN`, which produces a non-standard JSON token.
- **Change:** convert NaN values to `null` before writing.

### 3.4 CIDEr is never computed by default (Verified)

- **Files:** `src/video_captioning/evaluation/metrics.py` (lines 66-74), `requirements.txt`, and `pyproject.toml`. (Corrected: originally cited as lines 293-301.)
- **Reality:** `pycocoevalcap` is not listed in `requirements.txt` or in the `analysis` extra.
- **Change:** add `pycocoevalcap` to the `analysis` extra and to `requirements.txt`, or document CIDEr as opt-in.

### 3.5 METEOR may need NLTK data that the code does not fetch (By reading)

- **File:** `src/video_captioning/evaluation/metrics.py`, line 54 (`meteor_score` call; import at line 37). (Corrected: originally cited as line 264.)
- **Change:** verify on a clean environment. If it fails, add an NLTK data check or a download step, and document it in the setup section.

### 3.6 Slow candidate frame extraction (By reading)

- **File:** `src/video_captioning/data/preprocessing.py`, line 66. (Corrected: originally cited as line 213.)
- **Problem:** each frame is fetched with `CAP_PROP_POS_FRAMES`, which seeks and decodes from the previous keyframe each time.
- **Change:** benchmark first. If it matters, decode sequentially and keep only the target indices.

### 3.7 Video extension list differs between library and notebook (Verified)

- **Files:**
  - `src/video_captioning/data/dataset.py`, lines 13 (`VIDEO_EXTENSIONS`) and 76 (regex): include `mkv`. (Corrected: originally cited as line 101.)
  - `scripts/build_notebook.py`, line 148: omits `mkv`. This line also triggers a `SyntaxWarning` for the `\.` escape when the builder runs.
- **Change:** use one shared extension list. Use a raw string for the regex in the builder.

---

## 4. Tests and CI

### 4.1 Remove the tautological assertion (Verified)

- **File:** `tests/test_metrics.py`, line 24: `assert not math.isnan(float(len(METRIC_NAMES)))`.
- **Change:** delete it. It always passes and tests nothing.

### 4.2 Add tests for untested code (Verified: no tests exist for these)

- [ ] `select_frames_from_video` in `keyframes/pipeline.py`, including the two-video case from 1.1.
- [ ] `motion_scores` and `minmax_normalize` in `keyframes/motion.py`.
- [ ] `semantic_scores` in `keyframes/semantic.py`, with a stub processor and model.
- [ ] `load_config` in `utils/config.py`: the inheritance chain (`finetuning.yaml` → `training.yaml` → `baseline.yaml`) and rejection of unknown fields.
- [ ] `read_caption_table` for CSV, TSV, and JSON input.
- [ ] `index_video_paths` in `data/dataset.py`.
- [ ] `BLIP2VideoCaptioner.load_checkpoint` with a stub (after the fix in 1.2).
- [ ] `evaluate_captions` end to end on a small fixed input.

### 4.3 CI install time (Verified)

- **File:** `.github/workflows/python-package.yml`.
- **Problem:** every matrix job (Python 3.10, 3.11, 3.12) installs the full `requirements.txt`, including torch.
- **Change:** add a pip cache, and/or split a lightweight test job that installs only what the unit tests need.

### 4.4 Lint is not enforced (Verified)

- **File:** `.github/workflows/python-package.yml`, the second flake8 command uses `--exit-zero`.
- **Change:** run flake8 locally, fix or triage the warnings, then remove `--exit-zero`.

---

## 5. Repository hygiene

### 5.1 Coursework inside the research repo (Verified)

- **Folder:** `coursework/lab8-msvd-clustering/`. Its README calls it "unrelated" to the research pipeline.
- **Change:** move it to its own repository, or list it clearly in the README Project Structure.

### 5.2 `research/READ-format.md` is a generic template (Verified)

- **Problem:** it describes Hydra configs, lockfiles, a container-first deployment, and `MODEL_REGISTRY_URI`-style env vars. None of these apply to this project.
- **Change:** rename it to make the template status obvious, or move it out of `research/`.

### 5.3 Keep documentation in sync (Verified)

- **Files:** `README.md`, `research/methodology.md`, and `research/experiments.md`.
- **Change:** after any change above, update these three so they match the code.

---

## 6. Research and experiments

### 6.1 Record real results (Verified)

- **File:** `research/experiments.md`: every EXP entry says "Not yet run."
- **File:** `README.md`, Results table: all values say "run notebook."
- **Change:** run EXP-000 (uniform sampling) and EXP-001 (`semantic_motion_diverse`). Save metrics under `results/metrics/` and record them in the experiment log. Do not fill values before they are measured.

### 6.2 Confirm alpha is tuned on validation only (By reading)

- **Files:** `research/methodology.md`, notebook Section 17b.
- **Change:** confirm the notebook selects alpha on the validation split only, then runs the test set once. Record the chosen alpha in the experiment log.

### 6.3 Add the MSVD data dictionary (Roadmap item)

- **Files:** `README.md` Roadmap, `research/methodology.md` Future work.
- **Change:** write a data dictionary covering the MSVD schema variants that `normalize_captions` supports.

### 6.4 Add a small smoke-test fixture (Roadmap item)

- **Change:** add a few short clips and a 3-row caption file. This would let CI exercise `scripts/train.py` and `scripts/inference.py` without the full dataset.

### 6.5 Qualitative error analysis (Future work)

- **File:** `research/methodology.md`, Future work.
- **Change:** run an error analysis on generated captions and record it with the results.

---

## Suggested order

1. Section 1 (bugs), with the test from 4.2.
2. Section 2.2 (notebook source-of-truth decision), since 2.1, 2.5, and 2.6 depend on it.
3. Section 2 remaining items and Section 3.
4. Section 4 remaining items.
5. Section 6 (experiments), once the code path is trustworthy.
6. Section 5 (hygiene).
