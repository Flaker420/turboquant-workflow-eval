from __future__ import annotations

import os
import re
from copy import deepcopy
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
# Config merging, env-var expansion, and CLI dot-notation overrides
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into *base*. Overlay values win on conflict.

    Returns a new dict — neither input is mutated.
    """
    merged = deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def expand_env_vars(cfg: dict) -> dict:
    """Walk all string values and replace ``${VAR}`` / ``${VAR:-default}`` with
    the corresponding environment variable.  Non-string values are left as-is.
    """

    def _expand(value: Any) -> Any:
        if isinstance(value, str):
            def _replace(m: re.Match) -> str:
                expr = m.group(1)
                if ":-" in expr:
                    var, default = expr.split(":-", 1)
                    return os.environ.get(var.strip(), default)
                return os.environ.get(expr.strip(), m.group(0))
            return _ENV_PATTERN.sub(_replace, value)
        if isinstance(value, dict):
            return {k: _expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_expand(item) for item in value]
        return value

    return _expand(cfg)


def _coerce_value(raw: str) -> Any:
    """Auto-coerce a CLI string to int, float, bool, or leave as str."""
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    if raw.lower() in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def apply_dot_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``key.subkey=value`` overrides to *cfg*.

    Dot-separated keys navigate into nested dicts.  Values are auto-coerced
    via :func:`_coerce_value`.  Returns a **new** dict.
    """
    cfg = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key_path, raw_value = item.split("=", 1)
        parts = key_path.strip().split(".")
        value = _coerce_value(raw_value.strip())

        target = cfg
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return cfg


def load_yaml_with_overrides(
    path: str | Path,
    overrides: list[str] | None = None,
    base: dict | None = None,
) -> dict[str, Any]:
    """Load a YAML file with optional base config merging, dot-notation
    overrides, and environment variable expansion.

    Order: base → file → dot-overrides → env-var expansion.
    """
    cfg = load_yaml(path)
    if base:
        cfg = deep_merge(base, cfg)
    if overrides:
        cfg = apply_dot_overrides(cfg, overrides)
    cfg = expand_env_vars(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Config validation helpers
# ---------------------------------------------------------------------------

_REQUIRED_MODEL_FIELDS = ("model_name", "dtype")
_REQUIRED_POLICY_FIELDS = ("name", "adapter")
_IMPORT_PATH_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
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
        else:
            import_path = adapter["import_path"]
            if not isinstance(import_path, str) or not _IMPORT_PATH_RE.match(import_path):
                errors.append(
                    f"policy config{label}: adapter.import_path must match "
                    f"'module.path:ClassName' (got {import_path!r})"
                )
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

    policy_configs = cfg.get("policy_configs") or []
    if isinstance(policy_configs, list) and len(policy_configs) > 1:
        if not cfg.get("baseline_policy_name"):
            errors.append(
                f"study config{label}: 'baseline_policy_name' is required when "
                f"more than one policy_config is listed (got {len(policy_configs)})."
            )
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
