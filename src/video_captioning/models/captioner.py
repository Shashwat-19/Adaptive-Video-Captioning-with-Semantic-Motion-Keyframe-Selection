"""BLIP-2 storyboard captioning used by the notebook and inference CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def make_contact_sheet(frames: list, image_size: int = 224):
    """Compose selected frames into the one-image storyboard consumed by BLIP-2."""
    if not frames:
        raise ValueError("No decoded frames were provided.")
    import math
    from PIL import Image

    columns = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (columns * image_size, rows * image_size), (0, 0, 0))
    for index, frame in enumerate(frames):
        sheet.paste(
            frame.convert("RGB").resize((image_size, image_size)),
            ((index % columns) * image_size, (index // columns) * image_size),
        )
    return sheet


class BLIP2VideoCaptioner:
    """Lazy-loading BLIP-2 wrapper for captioning a selected-frame storyboard."""

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

    def generate_caption(self, frames: list, prompt: str = "A short video of") -> str:
        """Generate a caption for selected frames using a contact-sheet representation."""
        import torch

        self.load_model()
        assert self.model is not None and self.processor is not None and self.device is not None
        sheet = make_contact_sheet(frames, self.image_size)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        inputs = self.processor(images=sheet, text=prompt, return_tensors="pt").to(self.device, dtype)
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=self.max_caption_length, num_beams=3)
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    def load_checkpoint(self, path: str | Path) -> "BLIP2VideoCaptioner":
        """Load a locally saved BLIP-2 checkpoint directory."""
        self.model_name = str(path)
        self.model = None
        self.processor = None
        return self.load_model()
