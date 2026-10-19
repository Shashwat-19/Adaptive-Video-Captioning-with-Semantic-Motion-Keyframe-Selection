import types

import pytest

np = pytest.importorskip("numpy")

from video_captioning.keyframes.motion import motion_scores
from video_captioning.keyframes.semantic import semantic_scores


def test_scores_are_empty_for_no_frames():
    assert motion_scores([]).size == 0
    scores, embeddings = semantic_scores([], processor=None, model=None, device="cpu")
    assert scores.size == 0
    assert embeddings.shape == (0, 0)


def test_semantic_scores_measure_novelty_against_the_previous_frame():
    torch = pytest.importorskip("torch")
    # Each "frame" is an index into a table of CLS embeddings; the stubs stand in for DINOv2.
    table = torch.tensor([[0.0, 2.0], [1.0, 0.0], [3.0, 0.0], [1.0, 1.0]])

    class Inputs(dict):
        def to(self, device):
            return self

    def processor(images, return_tensors):
        return Inputs(pixel_values=torch.tensor(images))

    def model(pixel_values):
        return types.SimpleNamespace(last_hidden_state=table[pixel_values][:, None, :])

    # batch_size=3 splits the frames across two batches, so the 2 -> 3 comparison crosses a batch boundary.
    scores, embeddings = semantic_scores([0, 1, 2, 3], processor, model, device="cpu", batch_size=3)

    # Novelty is 1 - cosine similarity: same direction 0, orthogonal 1, 45 degrees 1 - 1/sqrt(2).
    # Frame 0 has no predecessor and takes the first interval's novelty, so it ties with frame 1.
    assert scores == pytest.approx([1.0, 1.0, 0.0, 1 - 1 / np.sqrt(2)])
    assert np.linalg.norm(embeddings, axis=1) == pytest.approx([1.0] * 4)


def test_motion_scores_rank_larger_shifts_higher():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    # Smooth texture at the scoring resolution (160x120), so Farneback flow tracks the shift.
    texture = cv2.GaussianBlur(rng.integers(0, 256, (120, 160)).astype(np.uint8), (9, 9), 3)
    frames = [np.stack([np.roll(texture, shift, axis=1)] * 3, axis=-1) for shift in (0, 0, 2, 8)]

    scores = motion_scores(frames)

    assert scores[0] == pytest.approx(scores[1])
    # Identical frames leave a small numerical residue in Farneback flow (about 1.6e-4 after normalizing), not exact zero.
    assert scores[1] == pytest.approx(0.0, abs=1e-3)
    assert 0.0 < scores[2] < scores[3] == 1.0
