"""Tests for the frozen dataclass schema and replace_path helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from turboquant_workflow_eval.schema import (
    AdapterSpec,
    ConfigValidationError,
    EarlyStopConfig,
    LayoutConfig,
    ModelConfig,
    OutputsConfig,
    PolicyConfig,
    PolicySettings,
    RuntimeConfig,
    StudyConfig,
    model_to_legacy_dict,
    policy_to_legacy_dict,
    replace_path,
)


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_minimal_valid(self) -> None:
        m = ModelConfig(model_name="Qwen/X", dtype="bf16")
        assert m.dtype == "bf16"
        assert m.notes == ()

    def test_dtype_aliases_accepted(self) -> None:
        for dtype in ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32"):
            ModelConfig(model_name="X", dtype=dtype)

    def test_invalid_dtype(self) -> None:
        with pytest.raises(ConfigValidationError, match="dtype"):
            ModelConfig(model_name="X", dtype="int8")

    def test_empty_model_name(self) -> None:
        with pytest.raises(ConfigValidationError, match="model_name"):
            ModelConfig(model_name="", dtype="bf16")

    def test_notes_coerced_to_tuple(self) -> None:
        m = ModelConfig(model_name="X", dtype="bf16", notes=["a", "b"])  # list input
        assert m.notes == ("a", "b")

    def test_layout_validation(self) -> None:
        with pytest.raises(ConfigValidationError, match="total_lm_layers"):
            LayoutConfig(total_lm_layers=0, attention_blocks=8)


# ---------------------------------------------------------------------------
# AdapterSpec / PolicyConfig
# ---------------------------------------------------------------------------


class TestAdapterSpec:
    def test_valid(self) -> None:
        AdapterSpec(import_path="my_pkg.module:Class")

    def test_dotted_module(self) -> None:
        AdapterSpec(import_path="a.b.c:Cls")

    def test_invalid_format(self) -> None:
        for bad in ("nocolon", ":no_module", "module:", "module:Class.bad", "1bad:X"):
            with pytest.raises(ConfigValidationError, match="import_path"):
                AdapterSpec(import_path=bad)


class TestPolicyConfig:
    def test_minimal(self) -> None:
        p = PolicyConfig(name="baseline", adapter=AdapterSpec(import_path="m:C"))
        assert p.comparison_label == "baseline"  # auto-defaulted
        assert p.enabled is True

    def test_explicit_label(self) -> None:
        p = PolicyConfig(
            name="x", adapter=AdapterSpec(import_path="m:C"), comparison_label="X-label"
        )
        assert p.comparison_label == "X-label"

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="name"):
            PolicyConfig(name="", adapter=AdapterSpec(import_path="m:C"))


# ---------------------------------------------------------------------------
# PolicySettings — including compressible_layers validation
# ---------------------------------------------------------------------------


class TestPolicySettings:
    def test_defaults(self) -> None:
        s = PolicySettings()
        assert s.bit_width == 4
        assert s.compressible_layers is None

    def test_invalid_bit_width(self) -> None:
        with pytest.raises(ConfigValidationError, match="bit_width"):
            PolicySettings(bit_width=0)

    def test_invalid_key_strategy(self) -> None:
        with pytest.raises(ConfigValidationError, match="key_strategy"):
            PolicySettings(key_strategy="bogus")

    def test_compressible_layers_coerced_to_tuple(self) -> None:
        s = PolicySettings(compressible_layers=[3, 7, 11])  # list input
        assert s.compressible_layers == (3, 7, 11)

    def test_compressible_layers_negative_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match=">= 0"):
            PolicySettings(compressible_layers=(-1, 3))

    def test_compressible_layers_duplicate_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="duplicate"):
            PolicySettings(compressible_layers=(3, 7, 3))

    def test_compressible_layers_empty_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="non-empty"):
            PolicySettings(compressible_layers=())

    def test_compressible_layers_non_int_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="ints"):
            PolicySettings(compressible_layers=("3", 7))  # type: ignore[arg-type]

    def test_compressible_heads_default_none(self) -> None:
        assert PolicySettings().compressible_heads is None

    def test_compressible_heads_coerced_to_tuple(self) -> None:
        s = PolicySettings(compressible_heads=[0, 2])
        assert s.compressible_heads == (0, 2)

    def test_compressible_heads_negative_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match=">= 0"):
            PolicySettings(compressible_heads=(-1, 0))

    def test_compressible_heads_duplicate_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="duplicate"):
            PolicySettings(compressible_heads=(0, 2, 0))

    def test_compressible_heads_empty_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="non-empty"):
            PolicySettings(compressible_heads=())


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------


class TestRuntimeConfig:
    def test_minimal(self) -> None:
        r = RuntimeConfig(max_input_tokens=1024, max_new_tokens=64)
        assert r.repetitions == 1
        assert r.use_cache is True

    def test_max_input_tokens_zero_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="max_input_tokens"):
            RuntimeConfig(max_input_tokens=0, max_new_tokens=64)

    def test_top_p_out_of_range(self) -> None:
        with pytest.raises(ConfigValidationError, match="top_p"):
            RuntimeConfig(max_input_tokens=64, max_new_tokens=64, top_p=1.5)

    def test_temperature_negative(self) -> None:
        with pytest.raises(ConfigValidationError, match="temperature"):
            RuntimeConfig(max_input_tokens=64, max_new_tokens=64, temperature=-0.1)


# ---------------------------------------------------------------------------
# StudyConfig — composition + cross-field validation
# ---------------------------------------------------------------------------


def _make_policy(name: str = "baseline") -> PolicyConfig:
    return PolicyConfig(name=name, adapter=AdapterSpec(import_path="m:C"))


def _make_study(**overrides) -> StudyConfig:
    base = dict(
        name="t",
        model=ModelConfig(model_name="Qwen/X", dtype="bf16"),
        prompt_pack=("p.yaml",),
        policies=(_make_policy(),),
        runtime=RuntimeConfig(max_input_tokens=1024, max_new_tokens=64),
    )
    base.update(overrides)
    return StudyConfig(**base)


class TestStudyConfig:
    def test_minimal(self) -> None:
        s = _make_study()
        assert s.baseline_policy_name == "baseline"  # auto-set for single policy
        assert s.prompt_pack == (Path("p.yaml"),)

    def test_prompt_pack_string_coerced(self) -> None:
        s = _make_study(prompt_pack="p.yaml")
        assert s.prompt_pack == (Path("p.yaml"),)

    def test_prompt_pack_empty_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="prompt_pack"):
            _make_study(prompt_pack=())

    def test_multi_policy_requires_baseline(self) -> None:
        with pytest.raises(ConfigValidationError, match="baseline_policy_name"):
            _make_study(policies=(_make_policy("a"), _make_policy("b")))

    def test_multi_policy_baseline_must_exist(self) -> None:
        with pytest.raises(ConfigValidationError, match="baseline_policy_name"):
            _make_study(
                policies=(_make_policy("a"), _make_policy("b")),
                baseline_policy_name="ghost",
            )

    def test_duplicate_policy_names_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="duplicate"):
            _make_study(
                policies=(_make_policy("a"), _make_policy("a")),
                baseline_policy_name="a",
            )

    def test_policies_must_be_PolicyConfig(self) -> None:
        with pytest.raises(ConfigValidationError, match="PolicyConfig"):
            _make_study(policies=({"name": "fake"},))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# replace_path helper
# ---------------------------------------------------------------------------


class TestReplacePath:
    def test_top_level(self) -> None:
        s = _make_study()
        s2 = replace_path(s, "name", "renamed")
        assert s2.name == "renamed"
        assert s.name == "t"  # original unchanged

    def test_nested(self) -> None:
        s = _make_study()
        s2 = replace_path(s, "runtime.max_new_tokens", 256)
        assert s2.runtime.max_new_tokens == 256
        assert s.runtime.max_new_tokens == 64

    def test_deeply_nested(self) -> None:
        s = _make_study()
        s2 = replace_path(s, "policies.0.settings.bit_width", 8)
        assert s2.policies[0].settings.bit_width == 8
        assert s.policies[0].settings.bit_width == 4

    def test_unknown_field(self) -> None:
        s = _make_study()
        with pytest.raises(ConfigValidationError, match="ghost"):
            replace_path(s, "ghost", 1)

    def test_unknown_nested_field(self) -> None:
        s = _make_study()
        with pytest.raises(ConfigValidationError, match="ghost"):
            replace_path(s, "runtime.ghost", 1)

    def test_tuple_index_out_of_range(self) -> None:
        s = _make_study()
        with pytest.raises(ConfigValidationError, match="out of range"):
            replace_path(s, "policies.5.name", "x")

    def test_tuple_index_non_int(self) -> None:
        s = _make_study()
        with pytest.raises(ConfigValidationError, match="int"):
            replace_path(s, "policies.bogus.name", "x")

    def test_post_init_runs_after_replace(self) -> None:
        """A replace that produces an invalid config should still re-validate."""
        s = _make_study()
        with pytest.raises(ConfigValidationError):
            replace_path(s, "runtime.max_new_tokens", 0)


# ---------------------------------------------------------------------------
# Legacy dict bridges (used at the core adapter seam)
# ---------------------------------------------------------------------------


class TestLegacyDictBridges:
    def test_model_to_legacy(self) -> None:
        m = ModelConfig(
            model_name="Qwen/X",
            dtype="bf16",
            device_map="auto",
            attn_implementation="sdpa",
            layout=LayoutConfig(total_lm_layers=32, attention_blocks=8),
        )
        d = model_to_legacy_dict(m)
        assert d["model_name"] == "Qwen/X"
        assert d["dtype"] == "bf16"
        assert d["device_map"] == "auto"
        assert d["attn_implementation"] == "sdpa"
        assert d["layout"]["total_lm_layers"] == 32

    def test_policy_to_legacy_drops_none_settings(self) -> None:
        p = _make_policy()
        d = policy_to_legacy_dict(p)
        assert d["adapter"]["import_path"] == "m:C"
        # `compressible_layers` defaults to None and should be omitted from
        # the legacy dict so core treats it as the default.
        assert "compressible_layers" not in d["settings"]
        assert "profile" not in d["settings"]

    def test_policy_to_legacy_includes_compressible_layers(self) -> None:
        p = PolicyConfig(
            name="x",
            adapter=AdapterSpec(import_path="m:C"),
            settings=PolicySettings(compressible_layers=(3, 7, 11)),
        )
        d = policy_to_legacy_dict(p)
        assert d["settings"]["compressible_layers"] == [3, 7, 11]
