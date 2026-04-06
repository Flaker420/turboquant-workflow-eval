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
        self._last_policy_cfg: dict | None = None

    def prepare_model(
        self, model: Any, tokenizer: Any, model_cfg: dict, policy_cfg: dict
    ) -> tuple[Any, Any]:
        # Bridge field name: eval harness uses "model_name", core expects "name".
        bridged_cfg = {**model_cfg, "name": model_cfg.get("model_name", "")}
        self._last_policy_cfg = policy_cfg
        return self._core.prepare_model(model, tokenizer, bridged_cfg, policy_cfg)

    def describe(self, policy_cfg: dict) -> dict:
        return self._core.describe(policy_cfg)

    def cleanup(self, model: Any) -> None:
        self._core.cleanup(model)

    def can_revert(self) -> bool:
        # Delegate to core if it supports the method, otherwise assume no
        if hasattr(self._core, "can_revert"):
            return self._core.can_revert()
        return False

    def revert(self, model: Any) -> bool:
        if hasattr(self._core, "revert"):
            return self._core.revert(model)
        # Fall back to cleanup + signal reload needed
        self._core.cleanup(model)
        return False

    def get_state(self) -> dict:
        if hasattr(self._core, "get_state"):
            return self._core.get_state()
        state: dict[str, Any] = {"adapter": self.name}
        if self._last_policy_cfg:
            settings = self._last_policy_cfg.get("settings", {})
            state["bit_width"] = settings.get("bit_width", "unknown")
            state["scope"] = settings.get("scope", "unknown")
        return state

    def update_params(self, params: dict) -> bool:
        if hasattr(self._core, "update_params"):
            return self._core.update_params(params)
        return False

    def reset_generation_state(self) -> None:
        if hasattr(self._core, "reset_generation_state"):
            self._core.reset_generation_state()
