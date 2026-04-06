"""Importers for the Python-module config format that replaced YAML.

Each loader resolves a path to either a ``.py`` file or a ``module:ATTR``
import-path string and returns the dataclass instance defined as the named
top-level symbol (``STUDY``, ``MODEL``, or ``POLICY``).

Module-relative paths inside config modules should be resolved using
``Path(__file__).parent`` from inside the module itself; the loader does
no path mangling.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
from typing import Any

from .schema import (
    ConfigValidationError,
    ModelConfig,
    PolicyConfig,
    StudyConfig,
)

__all__ = [
    "ConfigLoadError",
    "load_study_module",
    "load_model_module",
    "load_policy_module",
]


class ConfigLoadError(RuntimeError):
    """Raised when a config module cannot be located, imported, or read."""


def _load_symbol(path: str | Path, attr: str, expected_type: type) -> Any:
    """Locate a config module and return its top-level ``attr`` symbol."""
    spec_or_path = str(path)
    if ":" in spec_or_path and not Path(spec_or_path).exists():
        # Looks like ``package.module:ATTR``. Allow callers to override the
        # default attribute by writing ``module:OTHER`` instead of ``module``.
        module_name, _, override_attr = spec_or_path.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ConfigLoadError(
                f"Could not import config module {module_name!r}: {exc}"
            ) from exc
        symbol_name = override_attr or attr
        if not hasattr(module, symbol_name):
            raise ConfigLoadError(
                f"Module {module_name!r} has no top-level symbol {symbol_name!r}"
            )
        symbol = getattr(module, symbol_name)
    else:
        file_path = Path(spec_or_path).resolve()
        if not file_path.exists():
            raise ConfigLoadError(f"Config file not found: {file_path}")
        if file_path.suffix != ".py":
            raise ConfigLoadError(
                f"Config file must be a .py module, got {file_path.suffix!r}: {file_path}"
            )
        # Synthetic module name avoids collisions when the same name is loaded
        # from different directories (e.g. configs/policies/baseline.py used by
        # multiple studies).
        digest = hashlib.sha1(str(file_path).encode()).hexdigest()[:12]
        module_name = f"turboquant_eval_config_{file_path.stem}_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ConfigLoadError(f"Could not build import spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 — surface any module-level error
            raise ConfigLoadError(
                f"Error executing config module {file_path}: {exc}"
            ) from exc
        if not hasattr(module, attr):
            raise ConfigLoadError(
                f"Config module {file_path} must define a top-level {attr!r} symbol"
            )
        symbol = getattr(module, attr)

    if not isinstance(symbol, expected_type):
        raise ConfigLoadError(
            f"Expected {expected_type.__name__} from {spec_or_path!r}, "
            f"got {type(symbol).__name__}"
        )
    return symbol


def load_study_module(path: str | Path) -> StudyConfig:
    """Load a study config module and return its ``STUDY`` symbol."""
    return _load_symbol(path, "STUDY", StudyConfig)


def load_model_module(path: str | Path) -> ModelConfig:
    """Load a model config module and return its ``MODEL`` symbol."""
    return _load_symbol(path, "MODEL", ModelConfig)


def load_policy_module(path: str | Path) -> PolicyConfig:
    """Load a policy config module and return its ``POLICY`` symbol."""
    return _load_symbol(path, "POLICY", PolicyConfig)
