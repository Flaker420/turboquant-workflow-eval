from __future__ import annotations

from importlib import import_module

import torch

REQUIRED = [
    "torch",
    "transformers",
    "accelerate",
    "yaml",
]

for name in REQUIRED:
    module = import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: {version}")

print(f"cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count: {torch.cuda.device_count()}")
    print(f"cuda_device_name: {torch.cuda.get_device_name(0)}")
