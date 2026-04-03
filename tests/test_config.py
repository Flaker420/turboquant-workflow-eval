from pathlib import Path

from turboquant_workflow_eval.config import load_yaml


def test_load_yaml_roundtrip(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("name: demo\nphase: 0\n", encoding="utf-8")
    data = load_yaml(path)
    assert data["name"] == "demo"
    assert data["phase"] == 0
