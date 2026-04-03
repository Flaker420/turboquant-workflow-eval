from __future__ import annotations

from typing import Any


class CompressionAdapter:
    name = "base"

    def prepare_model(self, model: Any, tokenizer: Any, model_cfg: dict, policy_cfg: dict) -> tuple[Any, Any]:
        return model, tokenizer

    def describe(self, policy_cfg: dict) -> dict:
        return {"adapter": self.name}

    def cleanup(self, model: Any) -> None:
        return None
