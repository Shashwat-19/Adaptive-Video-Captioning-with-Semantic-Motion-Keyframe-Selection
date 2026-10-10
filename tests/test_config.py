from pathlib import Path

import pytest

pytest.importorskip("yaml")

from video_captioning.utils.config import ExperimentConfig, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_load_config_merges_a_three_level_chain_with_child_overrides(tmp_path):
    defaults = ExperimentConfig()
    (tmp_path / "grandparent.yaml").write_text("seed: 7\nalpha: 0.2\n", encoding="utf-8")
    (tmp_path / "parent.yaml").write_text("inherits: grandparent.yaml\nnum_epochs: 9\nalpha: 0.3\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("inherits: parent.yaml\nalpha: 0.9\n", encoding="utf-8")

    config = load_config(tmp_path / "child.yaml")

    assert (config.seed, config.num_epochs, config.alpha) == (7, 9, 0.9)
    assert (defaults.seed, defaults.num_epochs, defaults.alpha) != (7, 9, 0.9)


def test_repository_finetuning_config_resolves_through_its_parents():
    config = load_config(CONFIG_DIR / "finetuning.yaml")

    assert config.output_dir == "outputs/finetuning"
    assert config.keyframe_method == "semantic_motion_diverse"


def test_load_config_rejects_unknown_fields(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\nnot_a_field: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not_a_field"):
        load_config(path)
