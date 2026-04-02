from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..model_loader import resolve_language_model_root
from ..module_discovery import discover_attention_blocks
from .base import CompressionAdapter


def _fake_quant_last_dim(x: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    if bits is None or bits >= 16:
        return x
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")
    qmax = (1 << (bits - 1)) - 1
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale


class _QuantizeOutputWrapper(nn.Module):
    def __init__(self, wrapped: nn.Module, bits: int):
        super().__init__()
        self.wrapped = wrapped
        self.bits = bits

    def forward(self, *args, **kwargs):
        y = self.wrapped(*args, **kwargs)
        if isinstance(y, torch.Tensor):
            return _fake_quant_last_dim(y, self.bits)
        return y


@dataclass
class _PatchRecord:
    parent: nn.Module
    child_name: str
    original: nn.Module
    wrapped: nn.Module
    module_path: str
    bits: int


class LocalTransformersPatchAdapter(CompressionAdapter):
    name = "local-transformers-patch"

    def __init__(self):
        self._patches: list[_PatchRecord] = []
        self._patched_layers: list[str] = []

    def _replace_submodule(self, model_root: nn.Module, module_path: str, bits: int) -> None:
        parent_path, child_name = module_path.rsplit('.', 1)
        parent = model_root.get_submodule(parent_path)
        original = getattr(parent, child_name)
        wrapped = _QuantizeOutputWrapper(original, bits)
        setattr(parent, child_name, wrapped)
        self._patches.append(_PatchRecord(parent, child_name, original, wrapped, module_path, bits))

    def prepare_model(self, model, tokenizer, model_cfg: dict, policy_cfg: dict):
        settings = policy_cfg.get("settings", {})
        k_bits = int(settings.get("k_bits", 8))
        v_bits = int(settings.get("v_bits", 8))
        expected = model_cfg.get("layout", {}).get("attention_blocks")

        model_root = resolve_language_model_root(model)
        blocks = discover_attention_blocks(model_root, expected_count=expected)

        for block in blocks:
            if block.k_proj_path and k_bits < 16:
                self._replace_submodule(model_root, block.k_proj_path, k_bits)
            if block.v_proj_path and v_bits < 16:
                self._replace_submodule(model_root, block.v_proj_path, v_bits)
            self._patched_layers.append(block.module_path)

        return model, tokenizer

    def describe(self, policy_cfg: dict) -> dict:
        settings = policy_cfg.get("settings", {})
        return {
            "adapter": self.name,
            "comparison_label": policy_cfg.get("comparison_label", policy_cfg.get("name", "safe")),
            "scope": settings.get("scope", "full_attention_only"),
            "quant_mode": settings.get("quant_mode", "symmetric_per_token_fake_quant"),
            "k_bits": settings.get("k_bits", 8),
            "v_bits": settings.get("v_bits", 8),
            "patched_layer_count": len(self._patched_layers),
            "patched_layers": self._patched_layers,
            "note": "Behavioral local patch on full-attention K/V projection outputs. This is a Transformers-side proxy, not a true compressed KV cache backend.",
        }

    def cleanup(self, model) -> None:
        for record in reversed(self._patches):
            setattr(record.parent, record.child_name, record.original)
        self._patches.clear()
        self._patched_layers.clear()
