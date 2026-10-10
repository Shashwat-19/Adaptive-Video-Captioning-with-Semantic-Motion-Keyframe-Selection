import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"


def load_evaluate_script():
    spec = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unavailable_metrics_serialize_as_valid_json_null():
    evaluate = load_evaluate_script()

    rendered = json.dumps(evaluate.json_ready({"BLEU-4": 0.25, "CIDEr": float("nan")}), allow_nan=False)

    assert json.loads(rendered) == {"BLEU-4": 0.25, "CIDEr": None}
