from __future__ import annotations

from typing import Any

from .base import CompressionAdapter


class ExternalCompressionAdapterStub(CompressionAdapter):
    name = "external-compression-adapter-stub"

    def prepare_model(self, model: Any, tokenizer: Any, model_cfg: dict, policy_cfg: dict) -> tuple[Any, Any]:
        raise RuntimeError(
            "This policy points to the external adapter stub. "
            "Replace the adapter import path in the policy config with a real backend implementation before enabling it."
        )
