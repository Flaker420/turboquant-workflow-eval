"""Adapter wrapping turboquant-core backends for KV-cache compression."""

from __future__ import annotations

from typing import Any

from turboquant_core import TurboQuantAdapter as _CoreAdapter

from .base import CompressionAdapter


class TurboQuantAdapter(CompressionAdapter):
    """Thin wrapper that bridges turboquant-core to the workflow-eval harness.

    turboquant-core ships its own ``TurboQuantAdapter`` that duck-types this
    interface, but its variant detection reads ``model_cfg["name"]`` while
    the eval harness uses ``model_cfg["model_name"]``.  This wrapper
    normalises the config before delegating.
    """

    name = "turboquant"

    def __init__(self) -> None:
        self._core = _CoreAdapter()

    def prepare_model(
        self, model: Any, tokenizer: Any, model_cfg: dict, policy_cfg: dict
    ) -> tuple[Any, Any]:
        model_name = model_cfg.get("model_name")
        if not model_name:
            raise ValueError(
                "model_cfg['model_name'] is required by the turboquant adapter; "
                "check the model YAML."
            )
        # Bridge field name: eval harness uses "model_name", core expects "name".
        bridged_cfg = {**model_cfg, "name": model_name}
        return self._core.prepare_model(model, tokenizer, bridged_cfg, policy_cfg)

    def describe(self, policy_cfg: dict) -> dict:
        return self._core.describe(policy_cfg)

    def cleanup(self, model: Any) -> None:
        self._core.cleanup(model)

    def can_revert(self) -> bool:
        return self._core.can_revert()

    def revert(self, model: Any) -> bool:
        return self._core.revert(model)

    def get_state(self) -> dict:
        return self._core.get_state()

    def update_params(self, params: dict) -> bool:
        return self._core.update_params(params)

    def reset_generation_state(self) -> None:
        self._core.reset_generation_state()
