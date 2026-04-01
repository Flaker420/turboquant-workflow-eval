from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

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

print(f"torch_cuda: {torch.version.cuda}")
print(f"cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count: {torch.cuda.device_count()}")
    print(f"cuda_device_name: {torch.cuda.get_device_name(0)}")

for opt in ["fla", "causal_conv1d", "triton"]:
    print(f"{opt}: {'FOUND' if find_spec(opt) else 'MISSING'}")
