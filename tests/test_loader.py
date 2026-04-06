"""Tests for the .py-module config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from turboquant_workflow_eval.loader import (
    ConfigLoadError,
    load_model_module,
    load_policy_module,
    load_study_module,
)
from turboquant_workflow_eval.schema import ModelConfig, PolicyConfig, StudyConfig


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


class TestLoadModelModule:
    def test_loads_minimal(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "m.py",
            "from turboquant_workflow_eval.schema import ModelConfig\n"
            "MODEL = ModelConfig(model_name='Qwen/X', dtype='bf16')\n",
        )
        m = load_model_module(path)
        assert isinstance(m, ModelConfig)
        assert m.model_name == "Qwen/X"

    def test_missing_symbol_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "m.py", "X = 1\n")
        with pytest.raises(ConfigLoadError, match="MODEL"):
            load_model_module(path)

    def test_wrong_type_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "m.py", "MODEL = 42\n")
        with pytest.raises(ConfigLoadError, match="ModelConfig"):
            load_model_module(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigLoadError, match="not found"):
            load_model_module(tmp_path / "nope.py")

    def test_non_py_extension_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "m.yaml", "MODEL: 1\n")
        with pytest.raises(ConfigLoadError, match=r"\.py module"):
            load_model_module(path)


class TestLoadPolicyModule:
    def test_loads(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "p.py",
            "from turboquant_workflow_eval.schema import PolicyConfig, AdapterSpec\n"
            "POLICY = PolicyConfig(name='b', adapter=AdapterSpec(import_path='m:C'))\n",
        )
        p = load_policy_module(path)
        assert isinstance(p, PolicyConfig)
        assert p.name == "b"


class TestLoadStudyModule:
    def test_round_trip_via_relative_paths(self, tmp_path: Path) -> None:
        # Build a small two-file tree: a model module + a study module that
        # imports it via Path(__file__).parent.
        (tmp_path / "model.py").write_text(
            "from turboquant_workflow_eval.schema import ModelConfig\n"
            "MODEL = ModelConfig(model_name='Qwen/X', dtype='bf16')\n"
        )
        (tmp_path / "policy.py").write_text(
            "from turboquant_workflow_eval.schema import PolicyConfig, AdapterSpec\n"
            "POLICY = PolicyConfig(name='b', adapter=AdapterSpec(import_path='m:C'))\n"
        )
        (tmp_path / "prompts.yaml").write_text("prompts: []\n")  # only existence matters
        study_path = tmp_path / "study.py"
        study_path.write_text(
            "from pathlib import Path\n"
            "from turboquant_workflow_eval.loader import load_model_module, load_policy_module\n"
            "from turboquant_workflow_eval.schema import StudyConfig, RuntimeConfig\n"
            "_HERE = Path(__file__).parent\n"
            "STUDY = StudyConfig(\n"
            "    name='t',\n"
            "    model=load_model_module(_HERE / 'model.py'),\n"
            "    prompt_pack=(_HERE / 'prompts.yaml',),\n"
            "    policies=(load_policy_module(_HERE / 'policy.py'),),\n"
            "    runtime=RuntimeConfig(max_input_tokens=1024, max_new_tokens=64),\n"
            ")\n"
        )
        study = load_study_module(study_path)
        assert isinstance(study, StudyConfig)
        assert study.name == "t"
        assert study.policies[0].name == "b"
        assert study.model.model_name == "Qwen/X"

    def test_module_exec_error_surfaces(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "broken.py", "raise RuntimeError('boom')\n")
        with pytest.raises(ConfigLoadError, match="boom"):
            load_study_module(path)
