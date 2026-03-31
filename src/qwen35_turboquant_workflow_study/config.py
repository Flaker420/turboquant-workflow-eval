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
