from __future__ import annotations

import pytest

from turboquant_workflow_eval.import_utils import load_object


class TestLoadObject:
    def test_valid_adapter(self) -> None:
        # Loading the adapter transitively imports turboquant_core, which
        # imports torch. Skip when torch is unavailable (sandbox).
        pytest.importorskip("torch")
        cls = load_object("turboquant_workflow_eval.adapters.none:NoCompressionAdapter")
        assert cls.__name__ == "NoCompressionAdapter"

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="module:object format"):
            load_object("no_colon_here")

    def test_blocked_module(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            load_object("os:system")

    def test_blocked_arbitrary_module(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            load_object("subprocess:run")

    def test_bypass_restriction(self) -> None:
        # When explicitly bypassed, arbitrary imports work
        obj = load_object("os.path:join", restrict_adapters=False)
        assert callable(obj)
