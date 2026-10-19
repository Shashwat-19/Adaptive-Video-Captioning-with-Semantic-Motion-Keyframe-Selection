"""Caption-generation model wrappers."""

from video_captioning.models.captioner import BLIP2VideoCaptioner, consensus_caption

__all__ = ["BLIP2VideoCaptioner", "consensus_caption"]
