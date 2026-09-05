# Methodology

## Problem definition and motivation

Long videos contain many near-duplicate frames. Processing every decoded frame with an image-captioning model is unnecessary for the research question in this repository: whether a small, informative set of keyframes can support video caption generation more efficiently. The implemented workflow compares uniform and random sampling with motion-only, semantic-only, fused semantic-motion, and temporally diverse fused selection.

## Input representation and candidate generation

The workflow reads MSVD-style caption tables and video files. Caption columns are normalized to `video_id` and `caption`; splits are made by unique video ID to avoid caption-level leakage. OpenCV then decodes a temporally spaced set of candidate frames. The current default is 30 candidates, but it is configuration-driven.

## Motion relevance

For each pair of neighboring candidate frames, OpenCV Farnebäck optical flow is calculated on 160×120 grayscale frames. The mean flow magnitude is min-max normalized to create a lightweight motion score. The first candidate receives a raw score of zero because it has no previous candidate.

## Semantic relevance

The current implementation loads the frozen `facebook/dinov2-small` image encoder. Each candidate frame is embedded independently, normalized, and compared with the preceding candidate through cosine similarity. The semantic score is the normalized novelty, `1 - cosine_similarity`; it is not trained on captions.

## Adaptive selection

The fused score is:

```text
score = alpha × normalized_semantic + (1 - alpha) × normalized_motion
```

`alpha`, the candidate count, selected-frame budget, and minimum temporal separation are defined in YAML configuration. The `semantic_motion_diverse` method applies greedy temporal non-maximum suppression to high fused-score candidates, then backfills deterministically when a short clip cannot satisfy the requested separation. Selection is available independently through `scripts/select_keyframes.py`.

## Visual representation and caption generation

Selected frames are arranged into a deterministic square contact sheet (storyboard). The storyboard is passed to `Salesforce/blip2-opt-2.7b` through `Blip2Processor` and `Blip2ForConditionalGeneration`. This is an image-model storyboard approach, not a native temporal video encoder or a separate video decoder.

## Training and fine-tuning

The repository includes an opt-in BLIP-2 fine-tuning path that follows the notebook: batch size one, gradient accumulation, automatic mixed precision when CUDA is available, validation loss tracking, early stopping after two non-improving validation epochs, and optional LoRA adapters targeting `q_proj` and `v_proj`. One reference caption per video is randomly selected for each sample. Full fine-tuning has not been run or benchmarked in this repository.

## Evaluation methodology

The implemented evaluator calculates BLEU-1 through BLEU-4, METEOR, and ROUGE-L using each video's available reference captions. CIDEr is calculated only when the optional `pycocoevalcap` package is installed; otherwise it is recorded as unavailable. Alpha sensitivity is intended to be chosen on validation data before a single frozen-configuration test evaluation. No numerical results are currently committed.

## Limitations

- MSVD files and captions are not distributed in this repository.
- DINOv2 and BLIP-2 weights are downloaded at runtime.
- The storyboard representation may miss temporal ordering or fine-grained action details.
- The motion proxy operates only between sparse candidates and can be sensitive to camera movement.
- Fine-tuning, ablation, and frame-budget results have not yet been recorded.

## Future work

Planned work includes a data dictionary for MSVD variants, a representative fully logged experiment, a small smoke-test fixture, and qualitative error analysis. Native video encoders, a model registry, and a production serving layer are not implemented.
