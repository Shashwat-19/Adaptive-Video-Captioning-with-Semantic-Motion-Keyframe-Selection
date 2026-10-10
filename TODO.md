# TODO: Project Change List

Review date: 2026-10-10
Scope: whole repository (`src/`, `scripts/`, `tests/`, `configs/`, `notebooks/`, `research/`, `.github/`, `README.md`).

This file is a plan only. No source, config, or notebook files were changed while writing it.

**Current state:** `pytest` passes (9/9). flake8 is not installed locally, so lint was not run. Items marked *(by reading)* were found by reading the code and have not been reproduced by running it.

**Notebook review (2026-10-11):** the cell-by-cell review of `notebooks/video_captioning_research-4.ipynb` is tracked in `notebooks/TODO.md`. This file keeps the repository review below.

---

## P0: Bugs to fix first

- [x] **Fine-tuning crashes on the second training video** *(reproduced with stubs, fixed)*
  - Where: `src/video_captioning/training/finetuning.py:79-81` (the argument is on line 80)
  - Problem: `select_frames_from_video(video_paths[video_id], config, self.semantic_runtime)` passes the semantic runtime as the third positional argument. That argument is `method` (`pipeline.py:25-29`), not `semantic_runtime`. After the first video, `method` becomes a tuple, and `select_keyframe_positions` raises `Unknown method`.
  - Fix: pass it by keyword, `semantic_runtime=self.semantic_runtime`.
  - Add a test that calls the dataset path twice.
  - Done: keyword fix in `finetuning.py:80`; `tests/test_pipeline.py` checks runtime reuse and statically checks the call site (fails on the unfixed code). A runtime test of `VideoCaptionDataset` still needs it moved out of `run_finetuning` and torch installed.

- [ ] **LoRA checkpoints cannot be loaded by the inference CLI** *(verified 2026-10-10: not a bug with the pinned versions; docstring added)*
  - Where: `finetuning.py:140-141` saves with `model.save_pretrained(...)` on a PEFT model. This writes adapter weights only. `captioner.py:79-84` (`load_checkpoint`) loads the directory as a full `Blip2ForConditionalGeneration`.
  - Suspected impact (not reproduced, see Verified): `scripts/inference.py --checkpoint <epoch_dir>` fails on a LoRA run.
  - Proposed fix (not needed): either load the base model and then `PeftModel.from_pretrained`, or save a merged model.
  - Open question (answered by the check below): the pinned `transformers==4.45.2` with `peft` installed may auto-load adapter directories.
  - Verified 2026-10-10: **not a bug** with the pinned versions. The check ran in a throwaway Python 3.11 venv with torch 2.5.1, transformers 4.45.2, peft 0.12.0 and accelerate 0.34.2, on CPU in float32 (the dtype `load_model` uses on CPU), using a tiny randomly initialised `Blip2ForConditionalGeneration` with no model download.
    - The adapter was created with the same `LoraConfig` as `finetuning.py`, and `lora_B` was randomised so the adapter changes outputs. `save_pretrained` wrote only `adapter_config.json`, `adapter_model.safetensors` and `README.md`.
    - `Blip2ForConditionalGeneration.from_pretrained(<epoch dir>)`, which is what `load_checkpoint` calls, returned a model with 8 LoRA weight tensors (A and B for each targeted module). Its logits matched the PEFT model (`allclose`, atol 1e-5) and differed from the plain base.
    - The model is loaded from the adapter's `base_model_name_or_path` (`Salesforce/blip2-opt-2.7b` in a real run), so that model must be downloadable or cached.
    - The processor loads because `run_finetuning` saves it into the same epoch directory. This was confirmed by reading `finetuning.py`, not by running it.
  - Still unverified: the CUDA float16 load, loading an adapter with `use_4bit=True` (needs CUDA and bitsandbytes), and the full `scripts/inference.py --checkpoint` run (needs real weights).
  - Done: the `load_checkpoint` docstring (`captioner.py`) documents this behaviour. No code fix is needed.
  - Review status: two haiku rounds (stopped at the CLAUDE.md cap). Round 2 left one WARN: the old "Unverified" line contradicted the result. It was reworded after round 2 and has not been re-reviewed, so the item is not checked off. After the docstring, `load_checkpoint` spans `captioner.py:79-89`. The probe script and its outputs are in the session scratchpad (`lora_probe.py`, `lora_probe/`), not in the repo.

- [x] **Library caption normalization corrupts the official MSVD corpus** *(new finding 2026-10-10, reproduced with pinned pandas 2.2.3; fixed 2026-10-10, opus review round 2: READY TO COMMIT)*
  - Where: `src/video_captioning/data/dataset.py:43-81` (`normalize_captions`, before the fix) (Corrected: was `:56-79`); used by `scripts/train.py` and `scripts/evaluate.py`.
  - Problems, each reproduced on a small CSV with the official columns:
    - No language filter: non-English captions (e.g. French) are kept as training targets and evaluation references. The notebook keeps English only.
    - A missing `Description` becomes the literal caption `"nan"` and is kept, because `astype(str)` runs before the empty-string filter.
    - A missing `Start`/`End` gives a `video_id` like `xyz_nan_9` instead of dropping the row.
    - Integer segments read as floats (any column with a missing value) give `abc_1.0_5`, which does not match the `abc_1_5` file stem. The notebook's `normalize_segment` handles this.
  - Fix: port the notebook's handling into the library (English filter when a `Language` column exists, drop missing captions/segments before `astype(str)`, normalize segments), with tests. Review on opus (normalization and splits).
  - Done:
    - `dataset.py` has a new public `normalize_segment`, also re-exported from `data/__init__.py`.
    - `normalize_captions(raw, language="english")` keeps English rows when a language column exists; `language=None` keeps every row.
    - It drops rows with a missing id, caption, or segment bound before `astype(str)`, and drops blank-string bounds.
    - It collapses whitespace in captions.
    - Six new tests in `tests/test_dataset.py`.
    - `research/data-dictionary.md` and the reviewer facts are updated.
    - Verified on pandas 3.0.6 and the pinned 2.2.3. Each removed piece of the fix fails a test.
    - The split universe is now English-only, as in the notebook. Library split membership changes from before; no committed results depend on it.
  - Open NITs (optional):
    - The data dictionary does not list one more difference: a whitespace-only bound is dropped by the library, while the notebook keeps an unmatched `abc__5` ID.
    - Called directly on NaN, `normalize_segment` returns `"nan"`; the notebook's version raises. This can't happen inside `normalize_captions`.
    - `scripts/build_notebook.py:133-151` still has the old `normalize_captions` (item 2.2).
    - `index_video_paths` does not accept `.mpg`/`.mpeg` as the notebook does.

- [ ] **`read_caption_table` does not handle a UTF-8 BOM or malformed lines** *(follow-up from the review of the item above, 2026-10-10)*
  - Status: the BOM fix is implemented, and the code was approved in both review rounds. Round 2 found one more false doc claim: that the notebook keeps short rows. That claim was corrected after round 2 and not re-reviewed (two-round cap), so the item stays open. Malformed-row handling is left open as your decision.
  - Where: `src/video_captioning/data/dataset.py:40` at HEAD (`:44` after the fix). The notebook (Section 2b) reads the official CSV with `encoding="utf-8-sig", on_bad_lines="warn"`, while the library used the default encoding. (Corrected after review: an earlier version said the library raises on a malformed line. It does not reliably; see "Malformed rows" below.)
  - Reproduced (pandas 2.2.3 and 3.0.6): a BOM-prefixed copy of the official header gives the column `\ufeffVideoID`, and `normalize_captions` raises `ValueError: Cannot map video/caption columns`. It fails loudly; nothing is corrupted silently.
  - Unverified: whether the real `MSR Video Description Corpus.csv` has a BOM or malformed lines (no local copy).
  - Fix: read with `encoding="utf-8-sig"` (harmless for files without a BOM), decide whether to skip bad lines with a warning as the notebook does, and add a BOM test.
  - Done: CSV, TSV and JSON are now decoded as `utf-8-sig`. `test_read_caption_table_strips_a_utf8_byte_order_mark` covers a BOM CSV, through `normalize_captions`, and a BOM JSON. The data dictionary is updated. Verified on pandas 3.0.6 and the pinned 2.2.3. HEAD's decoding fails the new test.
  - Malformed rows (verified 2026-10-10 on pandas 3.0.6 and 2.2.3; unchanged by this fix; left open as your decision):
    - A row with too few fields is padded with empty values, and both paths drop it silently: the library in `normalize_captions`, the notebook through its empty-`Description` filter.
    - A row with too many fields usually raises `ParserError` in the library, while the notebook skips it with a warning. A row with exactly twice the header's field count instead silently corrupts the table in both, with no warning. Examples are 4 fields in a 2-column file, or 16 in the 8-column official schema.
      - Example: `video_id,caption / clip_a,A dog runs / clip_b,too,many,fields / clip_c,A cat sits`.
      - `normalize_captions(read_caption_table(...))` returns only `[['many', 'fields']]`.
      - A notebook-style read (`on_bad_lines="warn"`) returns `[['many', 'fields'], [nan, nan]]`.
    - So neither path reliably detects malformed rows; silent mis-parsing is the real risk.
    - Possible fix: pass an explicit `sep` (`,` for `.csv`), add `on_bad_lines="error"` or a field-count check, and test the extra-field case.
  - Related, pre-existing (from the review): `sep=None` sniffing also mangles single-column and pipe-delimited files. For example, a one-column file gives columns `['v', 'deo_', 'd']`.

---

## P1: Consistency between code, docs, and notebook

- [x] **README links to a notebook that does not exist**
  - Where: `README.md:7`, `:11`, `:114`, `:141`
  - Problem: links pointed to `notebooks/video_captioning_research.ipynb`. The committed file is `notebooks/video_captioning_research-4.ipynb` (last commit `07441d6`).
  - Fix: rename the notebook to the canonical name, or update the links. The `-4` suffix looks like a download-duplicate name.
  - Done: the four links now point to `video_captioning_research-4.ipynb`. If item 2.2 renames or regenerates the notebook, update them again.

- [ ] **Notebook has drifted from its generator** *(blocked: your decision on the source of truth; several other items depend on it)*
  - Status 2026-10-11 (uncommitted, not reviewed on its own): the builder was regenerated from the committed notebook's cells, so it now has 65 cells. It still writes `video_captioning_research.ipynb`, not the `-4` name; that filename is still your decision. Leave this item open until the sync is committed separately.
  - Where: `scripts/build_notebook.py` (27 section headings, 54 cells) vs `notebooks/video_captioning_research-4.ipynb` (31 headings, 65 cells).
  - Missing from the builder: sections `0b`, `1b`, `2b`, `17b`, a "Video-Caption Coverage Audit", and an "Interpreting runtime" subsection. Section numbering also differs.
  - Problem: re-running `build_notebook.py` would silently discard the manual edits. Also, `build_notebook.py:653` writes to the non-suffixed filename.
  - Decide the source of truth: either regenerate the notebook from the builder, or port the edits back into the builder and stop editing the `.ipynb` by hand.

- [x] **README documents environment variables that the code ignores**
  - Where: `README.md:183`, `:197`, `:292-294`.
  - Done: README now says the env vars (including `OUTPUT_DIR`) apply to the notebook only, and names the YAML keys the scripts use. No code changed.
  - Problem: `MSVD_DATA_ROOT`, `MSVD_VIDEO_DIR`, `MSVD_CAPTION_FILE`, and `OUTPUT_DIR` are read only by the notebook's Section 0 code (embedded in `build_notebook.py:36-39`; the builder itself does not read them). `ExperimentConfig` (`utils/config.py`) and the scripts read YAML only.
  - Fix: either add env-var overrides to `load_config`, or change the README to say the variables apply only to the notebook.

- [x] **README roadmap says to add a license, but one exists**
  - Where: `README.md:357`. `LICENSE` (MIT) is already in the repo, and the README's License section cites it.
  - Fix: remove the roadmap item.

- [x] **README claims a notebook syntax check that does not exist**
  - Where: `README.md:328`, under "The notebook additionally performs practical checks". (Corrected: this was first described as a CI claim.)
  - Problem: no `compile`/`ast.parse` check exists in `build_notebook.py`, the committed notebook, or `.github/workflows/python-package.yml`.
  - Fix: drop the claim, or add a real check. A CI step needs an `--output` argument on the builder first (the path is hardcoded at `build_notebook.py:653`).

- [x] **Project Structure section is out of date**
  - Where: `README.md` Project Structure block.
  - Problem: it names the wrong notebook, leaves out `coursework/` and `.github/`, and does not mention `research/READ-format.md` as a template.
  - Done: added `LICENSE`, `.flake8`, `.gitignore`, `.github/workflows/`, `coursework/`; `research/` comment names the template.
  - Open (found in review round 2, not fixed): the `results/` comment says outputs are ignored, but `results/metrics.csv` and `results/final_results.csv` (named in README Results) are not ignored by `.gitignore`. Decide: reword the comment, or add those files to `.gitignore`.

- [x] **Hardcoded Colab paths in the notebook defaults** (documented in README Known Limitations; no code changed, since moving the paths needs the builder/notebook decision in item 2.2)
  - Where: `build_notebook.py:36` (`/content/msvd`), `:39` (`/content/video_captioning_outputs`), and `:85` (`/content/drive`, optional mount). (Corrected: `:85` was missing.) The committed notebook adds more in Section 2 (archive search under `/content`), Section 2b (fallback search paths), and Section 23 (`OUT` set directly).
  - Problem: `README.md` Contributing says "avoid hardcoding a single local path." These are Colab-only paths and can be overridden only through env vars.
  - Fix: keep them as defaults, but state this in the README, or make the notebook's Section 0 read from one variable.

---

## P2: Code quality and dependencies

- [x] **Deprecated AMP API in fine-tuning** (done in `finetuning.py` with `torch.amp.GradScaler("cuda", ...)` and `torch.autocast("cuda", ...)`, the documented names in PyTorch 2.3; runtime not tested because torch is not installed; `build_notebook.py:443`, `:455`, `:471` still use the old API, pending item 2.2)
  - Where: `finetuning.py:103`, `:112`, `:128`.
  - Fix: use `torch.amp.GradScaler("cuda", enabled=...)` and `torch.amp.autocast("cuda", enabled=...)`. The old calls emit FutureWarnings on newer PyTorch.

- [ ] **Duplicate `discover_caption_file`** *(blocked on the notebook decision: if the builder becomes the source of truth, the notebook can import the library copy; calling it from `scripts/` would silently pick the first caption-like file, a data-validity risk)*
  - Where: `src/video_captioning/data/dataset.py:16` and `scripts/build_notebook.py:119`.
  - Problem: the library version is exported from `data/__init__.py` but no script calls it. The notebook has its own copy.
  - Fix: either use the library function in `train.py` and `evaluate.py`, or remove it.

- [x] **Invalid JSON can be written for missing CIDEr**
  - Where: `scripts/evaluate.py:35` (`json.dumps(..., allow_nan=True)`). (Corrected: line was missing.)
  - Done: `json_ready()` maps NaN to `null`; dump uses `allow_nan=False`; test in `tests/test_evaluate_script.py`.
  - Problem: when CIDEr is unavailable it is `NaN`, which becomes an invalid JSON token.
  - Fix: write `null` for NaN values.

- [x] **CIDEr is never computed by default**
  - Where: `src/video_captioning/evaluation/metrics.py:66-74`. (Corrected: originally cited as `:293-301`.) `pycocoevalcap` is not in `requirements.txt` or the `analysis` extra in `pyproject.toml`.
  - Fix: add `pycocoevalcap` to the `analysis` extra, or document that CIDEr is opt-in.
  - Done: `pycocoevalcap==1.2` added to the `analysis` extra and `requirements.txt`; metric-comparability caveat added to README Evaluation and `research/methodology.md`.
  - Open: the notebook/Colab path still doesn't install `pycocoevalcap` (builder change, pending item 2.2); README Known Limitations "CIDEr depends on optional evaluation packages" could now say "the `analysis` extra".

- [x] **METEOR may need NLTK data that the code does not download** *(verified and fixed 2026-10-10; opus review: READY TO COMMIT)*
  - Where: `metrics.py:54` (`meteor_score` call; import at `:37`). (Corrected: originally cited as `:264`.) `meteor_score` needs the `wordnet` corpus.
  - Fix: verify on a clean environment. Add an NLTK data check or a download step to the setup docs.
  - Blocks: an end-to-end `evaluate_captions` test (including the CIDEr path from item 3.4), since METEOR runs first.
  - Verified (Python 3.11 venv with the pinned nltk 3.9.1, rouge-score 0.1.2, numpy 1.26.4; empty `HOME`, no system NLTK data): `evaluate_captions` raised `LookupError: Resource wordnet not found`. After `nltk.download("wordnet")` it scored normally, so only `wordnet` is needed, not `omw-1.4`. `nltk.download` leaves `corpora/wordnet.zip`, which `nltk.data.find("corpora/wordnet")` does not find, so the notebook's check (cell 43, `scripts/build_notebook.py:513-515`) re-downloads on every run. That is harmless but costs a network call.
  - Done: `ensure_nltk_data()` in `metrics.py`, called by `evaluate_captions`, downloads `wordnet` only when neither `corpora/wordnet` nor `corpora/wordnet.zip` is found. Tested with a fake `nltk` module. Verified in the clean venv: the first run downloads and scores, and the second run makes no download. README "Evaluation" has the offline step (`python -m nltk.downloader wordnet`).
  - Line citations in this item and in 3.4 refer to HEAD `07441d6`. After this fix, the `meteor_score` import is at `metrics.py:55`, the call at `:72`, and CIDEr at about `:84-92`.
  - Open NITs (optional):
    - The README's offline sentence asks for a download on a machine with no network. Better: "run it while online, or copy an `nltk_data` folder with `corpora/wordnet.zip` and set `NLTK_DATA`". The reviewer verified that `NLTK_DATA` works.
    - `ensure_nltk_data(resources=...)` only looks under `corpora/`, so either drop the parameter or say "corpora only".
    - The return value of `nltk.download` is ignored. Offline, nltk prints an error, and `meteor_score` then raises its usual `LookupError`.
    - A corrupt `wordnet.zip` would skip the download.

- [ ] **Slow frame extraction** *(UNVERIFIED, not implemented: needs a benchmark with OpenCV and real videos)*
  - Where: `src/video_captioning/data/preprocessing.py:66`. (Corrected: originally cited as `:213`.) Each frame is fetched with `CAP_PROP_POS_FRAMES`, which seeks and decodes from the previous keyframe every time.
  - Fix: decode sequentially and keep only the target indices, or benchmark first to confirm it matters.

- [x] **Trailing-caption regex differs between the library and the notebook**
  - Done: `build_notebook.py:148` now emits the same regex as `dataset.py:76` (adds `mkv`) and no longer triggers the escape warning; generated output differs only in that line; committed notebook untouched. A truly shared list needs item 2.2.
  - Follow-ups (pending item 2.2): the committed notebook also accepts `.mpg`/`.mpeg`, which the library and builder do not; the builder (`:139`) only detects `start`/`end` columns, while the library (`dataset.py:55-56`) also accepts `start_time`/`start_frame` and `end_time`/`end_frame`.
  - Where: `dataset.py:13` (`VIDEO_EXTENSIONS`) and `:76` (regex) include `mkv`. (Corrected: originally cited as `:101`.) `build_notebook.py:148` leaves out `mkv`. This line also triggers a `SyntaxWarning` for the `\.` escape.
  - Fix: share one extension list, and use a raw string.

---

## P3: Tests and CI

- [x] **Remove the tautological assertion**
  - Where: `tests/test_metrics.py:24`, `assert not math.isnan(float(len(METRIC_NAMES)))`. It always passes and tests nothing.

- [ ] **Add tests for untested modules** *(partial: 7 of 9 done and reviewed; `motion_scores`/`semantic_scores` tests are in `tests/test_keyframe_scores.py`; the identical-frames tolerance was loosened after a CI failure; the `semantic_scores` test skips without torch, so it has not run locally)*
  - [x] `select_frames_from_video` in `keyframes/pipeline.py`, including the two-video case from P0 (`tests/test_pipeline.py`, item 1.1).
  - [x] `minmax_normalize` (`tests/test_keyframe_selection.py`).
  - [x] `motion_scores` (`tests/test_keyframe_scores.py`; passes with the pinned `opencv-python-headless` 4.10.0.84. Identical frames normalize to about 1.6e-4, not exactly 0, so the test uses `abs=1e-3`).
  - [ ] `semantic_scores`, using a stub processor and model: needs torch, not installed locally.
  - [x] `load_config`: three-level inheritance with child overrides, repo chain smoke test, unknown-field rejection (`tests/test_config.py`).
  - [x] `read_caption_table` for CSV, TSV, and JSON (`tests/test_dataset.py`; the TSV case has a comma on every line so only the explicit `.tsv` rule parses it).
  - [x] `index_video_paths` (`tests/test_dataset.py`).
  - [x] `BLIP2VideoCaptioner.load_checkpoint` with a stub (`tests/test_model.py`; opus review: READY TO COMMIT). It stubs `load_model` and checks that the model and processor are reset and reloaded from the checkpoint path. Item 1.2 showed real LoRA loading works; that needs torch, so it is not in the unit tests.
  - [x] `evaluate_captions` end to end, with a small fixed input (`tests/test_metrics.py`; opus review: READY TO COMMIT; values recomputed on Python 3.10, 3.11 and 3.12). Expected values come from the pinned nltk 3.9.1, rouge-score 0.1.2 and pycocoevalcap 1.2, with BLEU-2 and ROUGE-L checked by hand. CIDEr is checked when `pycocoevalcap` is installed and NaN otherwise. The test skips without nltk or rouge-score; locally it skips because nltk is not installed. In CI it downloads `wordnet` (about 10 MB) on each run, since the NLTK data is not cached. There is also a no-references test (every metric NaN).
  - Open NIT: `test_repository_finetuning_config_resolves_through_its_parents` only smoke-tests the repo chain; the inheritance check is in the three-level test.

- [x] **CI speed**
  - Where: `.github/workflows/python-package.yml`. Each matrix job installs the full `requirements.txt` (including torch) on every run.
  - Fix: add a pip cache, or split a lightweight test job that installs only what the tests need.
  - Done: pip cache (`cache: pip`, keyed on `requirements.txt` and `pyproject.toml`). Speed-up unverified until CI runs. Open (your decision): splitting a lightweight test job.

- [x] **Lint is not enforced beyond syntax errors**
  - Where: the `--exit-zero` flake8 step. Run it once locally and fix or triage the warnings, then drop `--exit-zero`.
  - Done: 134 warnings triaged in `.flake8` (per-file E402 for `scripts/` and `tests/`; E122/E501 for the builder's embedded cells; global E203; C901 for `finetuning.py`, see the next item); `--exit-zero` removed; CI pins `flake8==7.4.1`. 0 warnings on Python 3.10, 3.11, 3.12 locally. Re-run both lint commands on a clean checkout before merging (untracked test files were included locally).

- [ ] **Refactor `run_finetuning` to reduce complexity** *(follow-up from the lint item)*
  - Where: `src/video_captioning/training/finetuning.py:18`; flake8 C901 complexity 15 (limit 10), currently suppressed per-file in `.flake8`.
  - Fix: move `VideoCaptionDataset`, `collate`, and the epoch loop to module level. This also makes the dataset testable (see the P0 fine-tuning item). Needs torch installed to verify, then remove the C901 ignore.

---

## P4: Research and experiments

- [ ] **Run the baseline and record the results** *(blocked: needs a GPU run, which the work rules forbid here)*
  - `research/experiments.md` has every EXP entry marked "Not yet run." The README Results table still says "run notebook."
  - Fix: run EXP-000 (uniform) and EXP-001 (`semantic_motion_diverse`), then save metrics to `results/metrics/` and record them in the log.
  - The P0 "Library caption normalization corrupts the official MSVD corpus" is fixed (uncommitted), so `scripts/` now uses English-only references. `read_caption_table` now strips a UTF-8 BOM (item "`read_caption_table` does not handle a UTF-8 BOM or malformed lines").

- [ ] **Tune alpha on validation only, then freeze the test set** *(confirmed by reading; recording the alpha is blocked on a run)*
  - Where: `research/methodology.md` and notebook Section 17b. Confirm the notebook does this and record the chosen alpha.
  - Confirmed (committed notebook): Section 17b selects alpha from `splits["validation"]` only (`idxmax` on validation metrics) and writes `configs/frozen_test_config.json`; Sections 18 and 19 call `assert_test_configuration_frozen()` before using `splits["test"]`. `scripts/build_notebook.py` has no such guard (only commented-out code), so this depends on item 2.2.

- [x] **Add the MSVD data dictionary** (roadmap item in README and `methodology.md` Future work).
  - Done: `research/data-dictionary.md`, linked from README "Dataset Layout & Configuration"; README roadmap bullet removed; `methodology.md` points to it. Every claim reproduced by the reviewer with pinned pandas 2.2.3.
  - Open NITs (optional doc additions): a JSON object without a `data` key becomes one record; the notebook's `normalize_segment` raises (stops the cell) rather than dropping rows; the notebook's own video index also accepts `.mpg`/`.mpeg` and raises on duplicate stems, while `index_video_paths` silently keeps the last duplicate; the library splits all caption IDs *before* intersecting with available videos (`finetuning.py:61-62`), while the notebook drops unmatched IDs first, so library split sizes can be smaller than the ratios.

- [ ] **Add a small smoke-test fixture** (roadmap item). A few short clips and a 3-row caption file would let CI exercise `train.py` and `inference.py` without the full dataset. *(blocked: needs your decision. Generating clips needs OpenCV (not installed here); `.gitignore` ignores `*.avi`/`*.mp4`, so committing clips needs an exception; and CI would download BLIP-2/DINOv2 weights.)*

- [ ] **Qualitative error analysis** (`methodology.md` Future work). *(blocked: needs generated captions from a run)*

---

## P5: Repository hygiene

- [x] **Move or separate the coursework**
  - Where: `coursework/lab8-msvd-clustering/`. Its README says it is "unrelated" to the research pipeline.
  - Fix: move it to its own repo, or list it clearly in the README Project Structure.
  - Done via the Project Structure item: README now lists `coursework/` as a separate course lab. Moving it to its own repo is your call.

- [ ] **Clarify `research/READ-format.md`** *(partly done; rename is your call)*
  - It is a generic README template. It mentions Hydra, lockfiles, Kubernetes-style deployment, and `MODEL_REGISTRY_URI`, none of which this project uses.
  - Fix: rename it to make it clearly a template, or move it out of `research/`.
  - The README Project Structure now labels it "a generic README template". Not renamed, because it may be a format brief you were given.

- [ ] **Keep the research/notebook docs in sync after any change above** *(ongoing)*
  - Affected files: `README.md`, `research/methodology.md`, `research/experiments.md`.
  - Applied with each item this session (README and `methodology.md` updated alongside code). `experiments.md` is unchanged because no runs were made.

---

## Suggested order

1. P0 bugs (small, testable, high impact).
2. P1 notebook decision, since the other P1 items depend on it.
3. P3 tests for the P0 fixes.
4. P2 cleanups.
5. P4 experiments, once the code path is trustworthy.
6. P5 hygiene.
