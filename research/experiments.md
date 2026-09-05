# Experiment Log

Record each completed run here or in a linked, versioned result artifact. Do not fill a value until it is measured.

## EXP-000 — Repository baseline

| Field | Value |
|---|---|
| Date | Not yet run |
| Dataset | MSVD (user-provided, not committed) |
| Dataset split | Video-level 80/10/10 split; seed 42 |
| Configuration | `configs/baseline.yaml` |
| Caption model | `Salesforce/blip2-opt-2.7b` |
| Keyframe strategy | Uniform sampling baseline |
| Candidate / selected frames | 30 / 8 |
| Learning rate / batch size / epochs | Not applicable for zero-shot inference |
| Training loss / validation loss | Not yet measured |
| BLEU / METEOR / ROUGE-L / CIDEr | Not yet measured |
| Efficiency measurements | Not yet measured |
| Observations | Awaiting a reproducible run with saved configuration and artifacts. |

## EXP-001 — Adaptive semantic-motion selection

| Field | Value |
|---|---|
| Date | Not yet run |
| Dataset | MSVD (user-provided, not committed) |
| Dataset split | Video-level split; test set held until configuration is frozen |
| Configuration | `configs/baseline.yaml` or a recorded derivative |
| Caption model | `Salesforce/blip2-opt-2.7b` |
| Keyframe strategy | `semantic_motion_diverse` |
| Candidate / selected frames | 30 / 8 by default |
| Semantic weight | `alpha: 0.5` by default; tune on validation only |
| Learning rate / batch size / epochs | Not yet measured or not applicable |
| Training loss / validation loss | Not yet measured |
| BLEU / METEOR / ROUGE-L / CIDEr | Not yet measured |
| Efficiency measurements | Not yet measured |
| Observations | Compare against uniform sampling at an equal frame budget. |

## Recording checklist

- Save the exact YAML configuration and random seed.
- Record the caption-file schema, video count, missing/corrupt-video count, and split IDs.
- Store generated captions, metric JSON/CSV, and qualitative examples outside Git unless a small, intentionally curated artifact is added.
- Report a comparison against an equal-frame baseline and include failures, not only representative examples.
