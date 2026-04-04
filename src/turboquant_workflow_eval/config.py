from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping at {path}, got {type(data)!r}")
    return data


def resolve_relative_path(base_file: str | Path, value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute():
        return value
    return Path(base_file).resolve().parent / value


# ---------------------------------------------------------------------------
# Config validation helpers
# ---------------------------------------------------------------------------

_REQUIRED_MODEL_FIELDS = ("model_name", "dtype")
_REQUIRED_POLICY_FIELDS = ("name", "adapter")
_REQUIRED_STUDY_FIELDS = ("name", "model_config", "prompt_pack", "runtime")
_REQUIRED_RUNTIME_FIELDS = ("max_input_tokens", "max_new_tokens")


class ConfigValidationError(ValueError):
    """Raised when a configuration file is missing required fields."""


def validate_model_config(cfg: dict[str, Any], path: str | Path | None = None) -> list[str]:
    """Validate a model config dict. Returns list of error messages (empty = valid)."""
    errors: list[str] = []
    label = f" ({path})" if path else ""
    for field in _REQUIRED_MODEL_FIELDS:
        if field not in cfg:
            errors.append(f"model config{label}: missing required field '{field}'")
    if "dtype" in cfg:
        valid_dtypes = {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}
        if str(cfg["dtype"]).lower() not in valid_dtypes:
            errors.append(f"model config{label}: unsupported dtype '{cfg['dtype']}'")
    return errors


def validate_policy_config(cfg: dict[str, Any], path: str | Path | None = None) -> list[str]:
    """Validate a policy config dict. Returns list of error messages (empty = valid)."""
    errors: list[str] = []
    label = f" ({path})" if path else ""
    for field in _REQUIRED_POLICY_FIELDS:
        if field not in cfg:
            errors.append(f"policy config{label}: missing required field '{field}'")
    if "adapter" in cfg:
        adapter = cfg["adapter"]
        if not isinstance(adapter, dict) or "import_path" not in adapter:
            errors.append(f"policy config{label}: 'adapter' must be a dict with 'import_path'")
    return errors


def validate_study_config(cfg: dict[str, Any], path: str | Path | None = None) -> list[str]:
    """Validate a study config dict. Returns list of error messages (empty = valid)."""
    errors: list[str] = []
    label = f" ({path})" if path else ""
    for field in _REQUIRED_STUDY_FIELDS:
        if field not in cfg:
            errors.append(f"study config{label}: missing required field '{field}'")
    runtime = cfg.get("runtime", {})
    if isinstance(runtime, dict):
        for field in _REQUIRED_RUNTIME_FIELDS:
            if field not in runtime:
                errors.append(f"study config{label}: runtime missing required field '{field}'")
    else:
        errors.append(f"study config{label}: 'runtime' must be a mapping")
    return errors


def validate_config(cfg: dict[str, Any], kind: str, path: str | Path | None = None) -> None:
    """Validate *cfg* as *kind* (model/policy/study) and raise on errors."""
    validators = {
        "model": validate_model_config,
        "policy": validate_policy_config,
        "study": validate_study_config,
    }
    validator = validators.get(kind)
    if validator is None:
        raise ValueError(f"Unknown config kind {kind!r}, expected one of {list(validators)}")
    errors = validator(cfg, path)
    if errors:
        raise ConfigValidationError("\n".join(errors))
