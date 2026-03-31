from __future__ import annotations

from .base import CompressionAdapter


class NoCompressionAdapter(CompressionAdapter):
    name = "baseline-no-compression"

    def prepare_model(self, model, tokenizer, model_cfg: dict, policy_cfg: dict):
        return model, tokenizer

    def describe(self, policy_cfg: dict) -> dict:
        return {
            "adapter": self.name,
            "comparison_label": policy_cfg.get("comparison_label", "baseline"),
        }
