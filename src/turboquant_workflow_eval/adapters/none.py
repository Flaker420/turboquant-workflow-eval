from __future__ import annotations

from typing import Any

from .base import CompressionAdapter


class NoCompressionAdapter(CompressionAdapter):
    name = "baseline-no-compression"

    def prepare_model(self, model: Any, tokenizer: Any, model_cfg: dict, policy_cfg: dict) -> tuple[Any, Any]:
        return model, tokenizer

    def describe(self, policy_cfg: dict) -> dict:
        return {
            "adapter": self.name,
            "comparison_label": policy_cfg.get("comparison_label", "baseline"),
        }

    def can_revert(self) -> bool:
        return True

    def revert(self, model: Any) -> bool:
        return True

    def get_state(self) -> dict:
        return {"adapter": self.name, "compression": "none"}
