from __future__ import annotations

import importlib
from typing import Any


def load_object(import_path: str) -> Any:
    if ":" not in import_path:
        raise ValueError(f"Expected import path in module:object format, got {import_path!r}")
    module_name, object_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)
