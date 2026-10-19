"""BLIP-2 per-frame captioning with a consensus caption, used by the notebook and inference CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def consensus_caption(captions: list[str]) -> str:
    """Return the caption sharing the most words with the others, so one odd frame does not set the output."""
    captions = [caption.strip() for caption in captions if caption.strip()]
    if not captions:
        raise RuntimeError("BLIP-2 produced no non-empty frame caption.")
    words = [set(caption.lower().replace(".", " ").split()) for caption in captions]

    def agreement(index: int) -> float:
        return sum(
            len(words[index] & other) / max(1, len(words[index] | other))
            for position, other in enumerate(words)
            if position != index
        )

    return captions[max(range(len(captions)), key=agreement)]


class BLIP2VideoCaptioner:
    """Lazy-loading BLIP-2 wrapper that captions each selected frame and keeps the consensus caption."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        image_size: int = 224,
        max_caption_length: int = 32,
        use_4bit: bool = False,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.image_size = image_size
        self.max_caption_length = max_caption_length
        self.use_4bit = use_4bit
        self.device = device
        self.processor: Any | None = None
        self.model: Any | None = None

    def load_model(self) -> "BLIP2VideoCaptioner":
        """Download/load the configured BLIP-2 checkpoint on first use."""
        if self.model is not None:
            return self
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
        }
        if self.use_4bit:
            kwargs.update(load_in_4bit=True, device_map="auto")
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name, **kwargs)
        if not self.use_4bit:
            self.model.to(self.device)
        self.model.eval()
        return self

    def generate_caption(self, frames: list, prompt: str = "") -> str:
        """Caption every selected frame at the full BLIP-2 input size in one batch, then return the consensus caption."""
        import torch

        if not frames:
            raise ValueError("No decoded frames were provided.")
        self.load_model()
        assert self.model is not None and self.processor is not None and self.device is not None
        images = [frame.convert("RGB") for frame in frames]
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        # No prompt by default: fine-tuning trains on bare captions, so inference must not add one.
        text = {"text": [prompt] * len(images)} if prompt else {}
        inputs = self.processor(images=images, return_tensors="pt", **text).to(self.device, dtype)
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=self.max_caption_length, num_beams=3)
        return consensus_caption(self.processor.batch_decode(tokens, skip_special_tokens=True))

    def load_checkpoint(self, path: str | Path) -> "BLIP2VideoCaptioner":
        """Load a locally saved BLIP-2 checkpoint directory.

        A LoRA epoch directory from ``run_finetuning`` holds only adapter weights and the processor.
        With ``peft`` installed, ``from_pretrained`` reads its ``adapter_config.json``, loads the base
        model named there, and attaches the adapter, so the base model must also be available.
        """
        self.model_name = str(path)
        self.model = None
        self.processor = None
        return self.load_model()
