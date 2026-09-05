"""Caption-independent semantic novelty scoring with a frozen DINOv2 encoder."""

from __future__ import annotations

from typing import Any

import numpy as np

from video_captioning.keyframes.motion import minmax_normalize


def load_semantic_encoder(
    model_name: str = "facebook/dinov2-small", device: str | None = None
) -> tuple[Any, Any, str]:
    """Load the DINOv2 image processor and encoder only when semantic scoring is requested."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(resolved_device).eval()
    return processor, model, resolved_device


def semantic_scores(
    frames: list,
    processor: Any,
    model: Any,
    device: str,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Score each frame by embedding novelty relative to its previous candidate."""
    if not frames:
        return np.array([]), np.empty((0, 0))
    import torch

    vectors = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            inputs = processor(images=frames[start : start + batch_size], return_tensors="pt").to(device)
            embeddings = model(**inputs).last_hidden_state[:, 0].float()
            vectors.append(torch.nn.functional.normalize(embeddings, dim=-1).cpu())
    embedding_array = torch.cat(vectors).numpy()
    novelty = np.zeros(len(embedding_array))
    if len(embedding_array) > 1:
        novelty[1:] = 1 - (embedding_array[1:] * embedding_array[:-1]).sum(axis=1)
    return minmax_normalize(novelty), embedding_array
