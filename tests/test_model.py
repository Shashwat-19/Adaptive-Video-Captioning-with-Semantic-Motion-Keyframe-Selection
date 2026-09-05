import pytest

PIL = pytest.importorskip("PIL.Image")

from video_captioning.models.captioner import make_contact_sheet


def test_contact_sheet_preserves_the_storyboard_grid_shape():
    frames = [PIL.new("RGB", (4, 4), color="red") for _ in range(3)]

    sheet = make_contact_sheet(frames, image_size=10)

    assert sheet.mode == "RGB"
    assert sheet.size == (20, 20)
