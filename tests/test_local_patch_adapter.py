from __future__ import annotations

import torch
from torch import nn

from qwen35_turboquant_workflow_study.adapters.local_transformers_patch import _fake_quant_last_dim


def test_fake_quant_last_dim_preserves_shape() -> None:
    x = torch.randn(2, 3, 8)
    y = _fake_quant_last_dim(x, bits=8)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
