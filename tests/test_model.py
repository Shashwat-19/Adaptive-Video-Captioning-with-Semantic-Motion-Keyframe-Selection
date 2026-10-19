import pytest

PIL = pytest.importorskip("PIL.Image")

from video_captioning.models.captioner import BLIP2VideoCaptioner, consensus_caption


def test_load_checkpoint_reloads_model_and_processor_from_the_checkpoint_directory(monkeypatch, tmp_path):
    loaded_from = []

    def fake_load_model(self):
        assert self.model is None and self.processor is None
        loaded_from.append(self.model_name)
        self.model, self.processor = "model", "processor"
        return self

    monkeypatch.setattr(BLIP2VideoCaptioner, "load_model", fake_load_model)
    captioner = BLIP2VideoCaptioner()
    captioner.model, captioner.processor = "base model", "base processor"

    assert captioner.load_checkpoint(tmp_path / "epoch_1") is captioner
    assert loaded_from == [str(tmp_path / "epoch_1")]
    assert (captioner.model, captioner.processor) == ("model", "processor")


def test_consensus_caption_picks_the_caption_the_others_agree_with():
    assert consensus_caption(["a man cooking", "a man is cooking food", "a collage of images"]) == "a man cooking"


def test_consensus_caption_rejects_all_empty_captions():
    with pytest.raises(RuntimeError):
        consensus_caption(["", "  "])


def test_generate_caption_captions_every_frame_and_returns_the_consensus():
    pytest.importorskip("torch")
    seen = {}

    class Inputs(dict):
        def to(self, device, dtype):
            return self

    class Processor:
        def __call__(self, images, return_tensors, **kwargs):
            seen["images"], seen["kwargs"] = images, kwargs
            return Inputs(pixel_values=len(images))

        def batch_decode(self, tokens, skip_special_tokens):
            return ["a man cooking", "a man is cooking food", "a collage of images"]

    class Model:
        def generate(self, **kwargs):
            return "tokens"

    captioner = BLIP2VideoCaptioner(device="cpu")
    captioner.processor, captioner.model = Processor(), Model()

    assert captioner.generate_caption([PIL.new("RGB", (4, 4)) for _ in range(3)]) == "a man cooking"
    assert len(seen["images"]) == 3
    assert seen["kwargs"] == {}  # no prompt, matching fine-tuning
