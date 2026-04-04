from __future__ import annotations

from typing import Any


class CompressionAdapter:
    """Base class for compression adapters.

    Subclasses must implement at least :meth:`prepare_model`.  The new
    ``revert`` / ``can_revert`` / ``get_state`` / ``update_params`` methods
    enable model reuse across policies, live parameter tuning, and state
    inspection.  Default implementations are safe no-ops so existing adapters
    continue to work without changes.
    """

    name = "base"

    # --- Core lifecycle (existing) ---

    def prepare_model(self, model: Any, tokenizer: Any, model_cfg: dict, policy_cfg: dict) -> tuple[Any, Any]:
        """Apply compression to *model*. Returns ``(model, tokenizer)``."""
        return model, tokenizer

    def describe(self, policy_cfg: dict) -> dict:
        """Return a metadata dict describing the current adapter configuration."""
        return {"adapter": self.name}

    def cleanup(self, model: Any) -> None:
        """Release resources held by this adapter."""

    # --- New: revert / reuse ---

    def can_revert(self) -> bool:
        """Whether :meth:`revert` can undo :meth:`prepare_model` in-place.

        If ``False``, the study runner will reload the model from scratch
        before the next policy.
        """
        return True

    def revert(self, model: Any) -> bool:
        """Undo the changes made by :meth:`prepare_model`.

        Returns ``True`` if the model is now clean and reusable for another
        adapter, ``False`` if a full model reload is needed.

        The base implementation is a no-op that always returns ``True`` (the
        base adapter never modifies the model).
        """
        return True

    # --- New: inspection ---

    def get_state(self) -> dict:
        """Return the current compression parameters for inspection / UI display."""
        return {}

    # --- New: hot-update ---

    def update_params(self, params: dict) -> bool:
        """Hot-update compression params without full revert + reapply.

        Returns ``True`` if the update was applied successfully, ``False`` if
        the adapter does not support hot-update (caller should revert and
        re-prepare instead).
        """
        return False
