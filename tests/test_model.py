import pytest

PIL = pytest.importorskip("PIL.Image")

from video_captioning.models.captioner import BLIP2VideoCaptioner, make_contact_sheet


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


def test_contact_sheet_preserves_the_storyboard_grid_shape():
    frames = [PIL.new("RGB", (4, 4), color="red") for _ in range(3)]

    sheet = make_contact_sheet(frames, image_size=10)

    assert sheet.mode == "RGB"
    assert sheet.size == (20, 20)
