from __future__ import annotations

from turboquant_workflow_eval.adapters.turboquant import TurboQuantAdapter
from turboquant_workflow_eval.import_utils import load_object


class TestTurboQuantAdapter:
    def test_import_via_load_object(self) -> None:
        """Adapter is importable and passes the security prefix check."""
        cls = load_object("turboquant_workflow_eval.adapters.turboquant:TurboQuantAdapter")
        assert cls is TurboQuantAdapter

    def test_describe_returns_expected_keys(self) -> None:
        adapter = TurboQuantAdapter()
        policy_cfg = {
            "name": "turboquant_safe",
            "settings": {"bit_width": 4, "scope": "full_attention_only"},
        }
        desc = adapter.describe(policy_cfg)
        assert desc["adapter"] == "turboquant"
        assert desc["bit_width"] == 4

    def test_describe_defaults(self) -> None:
        adapter = TurboQuantAdapter()
        desc = adapter.describe({"name": "test", "settings": {}})
        assert desc["bit_width"] == 4
        assert desc["seed"] == 42

    def test_cleanup_clears_cache(self) -> None:
        adapter = TurboQuantAdapter()
        # Simulate a cache with a .clear() method
        adapter._core._cache = {"dummy": "data"}
        adapter.cleanup(None)
        assert adapter._core._cache is None

    def test_name_attribute(self) -> None:
        adapter = TurboQuantAdapter()
        assert adapter.name == "turboquant"
