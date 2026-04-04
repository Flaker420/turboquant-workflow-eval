from __future__ import annotations

import pytest

from turboquant_workflow_eval.config import (
    ConfigValidationError,
    validate_config,
    validate_model_config,
    validate_policy_config,
    validate_study_config,
)


class TestValidateModelConfig:
    def test_valid(self, sample_model_config) -> None:
        errors = validate_model_config(sample_model_config)
        assert errors == []

    def test_missing_model_name(self) -> None:
        errors = validate_model_config({"dtype": "bf16"})
        assert any("model_name" in e for e in errors)

    def test_missing_dtype(self) -> None:
        errors = validate_model_config({"model_name": "test"})
        assert any("dtype" in e for e in errors)

    def test_bad_dtype(self) -> None:
        errors = validate_model_config({"model_name": "test", "dtype": "int4"})
        assert any("unsupported dtype" in e for e in errors)

    def test_empty(self) -> None:
        errors = validate_model_config({})
        assert len(errors) >= 2


class TestValidatePolicyConfig:
    def test_valid(self, sample_policy_config) -> None:
        errors = validate_policy_config(sample_policy_config)
        assert errors == []

    def test_missing_name(self) -> None:
        errors = validate_policy_config({"adapter": {"import_path": "x:Y"}})
        assert any("name" in e for e in errors)

    def test_missing_adapter(self) -> None:
        errors = validate_policy_config({"name": "test"})
        assert any("adapter" in e for e in errors)

    def test_adapter_missing_import_path(self) -> None:
        errors = validate_policy_config({"name": "test", "adapter": {}})
        assert any("import_path" in e for e in errors)

    def test_adapter_not_dict(self) -> None:
        errors = validate_policy_config({"name": "test", "adapter": "string"})
        assert any("import_path" in e for e in errors)


class TestValidateStudyConfig:
    def test_valid(self) -> None:
        cfg = {
            "name": "test",
            "model_config": "model.yaml",
            "prompt_pack": "prompts.yaml",
            "runtime": {"max_input_tokens": 512, "max_new_tokens": 64},
        }
        errors = validate_study_config(cfg)
        assert errors == []

    def test_missing_name(self) -> None:
        cfg = {"model_config": "m.yaml", "prompt_pack": "p.yaml", "runtime": {"max_input_tokens": 1, "max_new_tokens": 1}}
        errors = validate_study_config(cfg)
        assert any("name" in e for e in errors)

    def test_missing_runtime(self) -> None:
        cfg = {"name": "test", "model_config": "m.yaml", "prompt_pack": "p.yaml"}
        errors = validate_study_config(cfg)
        assert any("runtime" in e for e in errors)

    def test_runtime_missing_fields(self) -> None:
        cfg = {"name": "test", "model_config": "m.yaml", "prompt_pack": "p.yaml", "runtime": {}}
        errors = validate_study_config(cfg)
        assert any("max_input_tokens" in e for e in errors)
        assert any("max_new_tokens" in e for e in errors)


class TestValidateConfig:
    def test_raises_on_errors(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_config({}, "model")

    def test_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="Unknown config kind"):
            validate_config({}, "unknown")

    def test_valid_passes(self, sample_model_config) -> None:
        validate_config(sample_model_config, "model")  # Should not raise
