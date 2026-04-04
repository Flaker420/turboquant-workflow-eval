from __future__ import annotations

import importlib
from typing import Any

# Only allow adapter imports from this module prefix.
_ALLOWED_ADAPTER_PREFIX = "turboquant_workflow_eval.adapters"


def load_object(import_path: str, *, restrict_adapters: bool = True) -> Any:
    """Dynamically import *import_path* (``module:ClassName`` format).

    When *restrict_adapters* is True (the default), the module portion must
    start with an allowed prefix to prevent arbitrary code loading from
    user-controlled YAML configs.
    """
    if ":" not in import_path:
        raise ValueError(f"Expected import path in module:object format, got {import_path!r}")
    module_name, object_name = import_path.split(":", 1)

    if restrict_adapters:
        if not module_name.startswith(_ALLOWED_ADAPTER_PREFIX):
            raise ValueError(
                f"Adapter import path must start with {_ALLOWED_ADAPTER_PREFIX!r}, "
                f"got {module_name!r}. Set restrict_adapters=False to bypass."
            )

    module = importlib.import_module(module_name)
    return getattr(module, object_name)
