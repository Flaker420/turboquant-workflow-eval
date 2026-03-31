from __future__ import annotations


class CompressionAdapter:
    name = "base"

    def prepare_model(self, model, tokenizer, model_cfg: dict, policy_cfg: dict):
        return model, tokenizer

    def describe(self, policy_cfg: dict) -> dict:
        return {"adapter": self.name}

    def cleanup(self, model) -> None:
        return None
