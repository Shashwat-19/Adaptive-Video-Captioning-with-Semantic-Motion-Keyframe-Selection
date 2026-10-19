# Methodology

## Problem definition and motivation

Long videos contain many near-duplicate frames. Processing every decoded frame with an image-captioning model is unnecessary for the research question in this repository: whether a small, informative set of keyframes can support video caption generation more efficiently. The implemented workflow compares uniform and random sampling with motion-only, semantic-only, fused semantic-motion, and temporally diverse fused selection.

## Input representation and candidate generation

The workflow reads MSVD-style caption tables and video files. Caption columns are normalized to `video_id` and `caption`; the notebook splits by source YouTube ID (all clips of one source video share a split) and the library splits by clip ID; both avoid caption-level leakage. OpenCV then decodes a temporally spaced set of candidate frames. The current default is 30 candidates, but it is configuration-driven.

## Motion relevance

For each pair of neighboring candidate frames, OpenCV Farnebäck optical flow is calculated on 160×120 grayscale frames. The mean flow magnitude is min-max normalized to create a lightweight motion score. The first candidate has no previous candidate, so it takes the score of the first interval instead of a hard-coded zero. It therefore ties with the second candidate; ties are broken toward the later frame by a stable sort. In the top-score methods the first two frames can both be chosen when the first interval scores high, which spends two slots on near-duplicates; the diverse method avoids this when they fall inside the minimum temporal distance.

## Semantic relevance

The current implementation loads the frozen `facebook/dinov2-small` image encoder. Each candidate frame is embedded independently, normalized, and compared with the preceding candidate through cosine similarity. The semantic score is the normalized novelty, `1 - cosine_similarity`; it is not trained on captions. As for motion, the first candidate takes the first interval's novelty.

## Adaptive selection

The fused score is:

```text
score = alpha × normalized_semantic + (1 - alpha) × normalized_motion
```

`alpha`, the candidate count, selected-frame budget, and minimum temporal separation are defined in YAML configuration. The `semantic_motion_diverse` method applies greedy temporal non-maximum suppression to high fused-score candidates, then backfills deterministically when a short clip cannot satisfy the requested separation. Selection is available independently through `scripts/select_keyframes.py`.

## Visual representation and caption generation

Each selected frame is captioned at the full 224 px input of `Salesforce/blip2-opt-2.7b` (through `Blip2Processor` and `Blip2ForConditionalGeneration`) in one batched call, and the video caption is the per-frame caption that shares the most words with the others. This is an image-model approach, not a native temporal video encoder or a separate video decoder. Fine-tuning trains on one selected frame per video (random each epoch; validation uses the middle selected frame and the first reference). The caption describes the whole video, so it may mention something the single training frame does not show.

## Training and fine-tuning

The repository includes an opt-in BLIP-2 fine-tuning path modelled on the notebook (the library masks only padding labels, while the notebook also masks image tokens and bounds captions with EOS): batch size one, gradient accumulation, automatic mixed precision when CUDA is available, validation loss tracking, early stopping after two non-improving validation epochs, and optional LoRA adapters targeting `q_proj` and `v_proj`. Training draws one random reference per video each epoch; validation uses the first reference. Full fine-tuning has not been run or benchmarked in this repository.

## Evaluation methodology

The implemented evaluator calculates BLEU-1 through BLEU-4, METEOR, and ROUGE-L using each video's available reference captions. CIDEr is calculated only when the optional `pycocoevalcap` package is installed (it is part of the `analysis` extra and `requirements.txt`); otherwise the library records it as NaN and the notebook raises an error. All metrics score captions and references after the internal protocol `lowercase_ascii_alphanumeric_v1` (lowercase, ASCII letters and digits only) and ROUGE-L is unstemmed, so scores are not directly comparable to published coco-caption or MSVD numbers. CIDEr's IDF weights come from the references of the evaluated set, so CIDEr should only be compared across methods scored on the same video IDs. Alpha is prespecified (default 0.5) and shared by training, validation and test for every arm. The validation alpha sweep is a sensitivity report only; by default it uses captions from the zero-shot base model. The fine-tuned arms are scored on the test set only through the checkpoints frozen with the configuration (path and file hash), on the same paired video IDs as the zero-shot ablation methods. No numerical results are currently committed.

## Limitations

- MSVD files and captions are not distributed in this repository.
- DINOv2 and BLIP-2 weights are downloaded at runtime.
- Per-frame captioning with a consensus caption ignores temporal ordering and may favour generic captions.
- The motion proxy operates only between sparse candidates and can be sensitive to camera movement.
- Fine-tuning, ablation, and frame-budget results have not yet been recorded.

## Future work

The MSVD schema variants are documented in `research/data-dictionary.md`. Planned work includes a representative fully logged experiment, a small smoke-test fixture, and qualitative error analysis. Native video encoders, a model registry, and a production serving layer are not implemented.
