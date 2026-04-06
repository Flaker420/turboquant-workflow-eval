from __future__ import annotations

import pytest

from turboquant_workflow_eval.config import apply_policy_overrides as _apply_policy_overrides


def test_named_match_dot_path():
    cfg = {"name": "turboquant_safe", "settings": {"bit_width": 4}}
    out = _apply_policy_overrides(
        cfg, ["turboquant_safe.settings.key_strategy=mse"]
    )
    assert out["settings"]["key_strategy"] == "mse"
    assert out["settings"]["bit_width"] == 4


def test_wildcard_match():
    cfg = {"name": "turboquant_aggressive", "settings": {"bit_width": 2}}
    out = _apply_policy_overrides(cfg, ["*.settings.bit_width=8"])
    assert out["settings"]["bit_width"] == 8


def test_non_matching_name_is_noop():
    cfg = {"name": "baseline", "settings": {}}
    out = _apply_policy_overrides(
        cfg, ["turboquant_safe.settings.key_strategy=mse"]
    )
    assert out == cfg


def test_top_level_key():
    cfg = {"name": "baseline", "enabled": True, "settings": {}}
    out = _apply_policy_overrides(cfg, ["baseline.enabled=false"])
    assert out["enabled"] is False


def test_value_coercion():
    cfg = {"name": "p", "settings": {}}
    out = _apply_policy_overrides(
        cfg, ["p.settings.bit_width=4", "p.settings.flag=true"]
    )
    assert out["settings"]["bit_width"] == 4
    assert out["settings"]["flag"] is True


def test_empty_overrides_passthrough():
    cfg = {"name": "p", "settings": {"k": "v"}}
    assert _apply_policy_overrides(cfg, None) is cfg
    assert _apply_policy_overrides(cfg, []) is cfg


def test_malformed_override_raises():
    cfg = {"name": "p"}
    with pytest.raises(ValueError, match="--set-policy"):
        _apply_policy_overrides(cfg, ["no_dot_or_equals"])
    with pytest.raises(ValueError, match="--set-policy"):
        _apply_policy_overrides(cfg, ["no_dot=value"])
