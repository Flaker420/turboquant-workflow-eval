"""Tests for scripts/generate_cli_docs.py and the argparse surface."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

from turboquant_workflow_eval.__main__ import _build_parser  # noqa: E402
import generate_cli_docs  # noqa: E402


def test_every_flag_has_help_text() -> None:
    """Every non-help argparse flag must set `help=` so docgen tables are useful."""
    parser = _build_parser()
    missing: list[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if not action.option_strings:
            continue
        if not action.help:
            missing.append(action.option_strings[0])
    assert not missing, f"flags missing help text: {missing}"


def test_readme_has_sentinels() -> None:
    text = (_ROOT / "README.md").read_text()
    assert generate_cli_docs.START in text
    assert generate_cli_docs.END in text


def test_readme_cli_docs_in_sync() -> None:
    """Running `generate_cli_docs.py --check` must exit zero against a committed README."""
    result = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "generate_cli_docs.py"), "--check"],
        env={"PYTHONPATH": f"{_ROOT / 'src'}:{_ROOT / 'vendor' / 'turboquant-core' / 'src'}",
             "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"README CLI docs are stale. stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_compressible_heads_flag_present() -> None:
    """Regression: both --compressible-heads and --compressible-heads-for must exist."""
    parser = _build_parser()
    flags = {
        opt
        for action in parser._actions
        for opt in action.option_strings
    }
    assert "--compressible-heads" in flags
    assert "--compressible-heads-for" in flags


def test_compressible_heads_for_per_policy_override() -> None:
    """End-to-end: --compressible-heads-for NAME=0,2 parses via argparse
    and `_apply_per_policy_overrides` lands the value on the matching
    policy's PolicySettings (others untouched).
    """
    from turboquant_workflow_eval.__main__ import (
        _build_parser, _apply_per_policy_overrides, parse_int_list,
    )
    from turboquant_workflow_eval.schema import (
        AdapterSpec, PolicyConfig, PolicySettings, StudyConfig,
        RuntimeConfig, ModelConfig,
    )

    parser = _build_parser()
    args = parser.parse_args([
        "--study", "ignored.py",
        "--compressible-heads-for", "turboquant_safe=0,2",
    ])
    assert args.compressible_heads_for == [("turboquant_safe", "0,2")]

    adapter = AdapterSpec(
        import_path="turboquant_workflow_eval.adapters.none:NoCompressionAdapter",
    )
    safe = PolicyConfig(name="turboquant_safe", adapter=adapter,
                        settings=PolicySettings(bit_width=4))
    other = PolicyConfig(name="turboquant_other", adapter=adapter,
                         settings=PolicySettings(bit_width=4))
    study = StudyConfig(
        name="fixture",
        model=ModelConfig(model_name="fake/model", dtype="bf16"),
        prompt_pack=(Path("fixture.yaml"),),
        policies=(safe, other),
        runtime=RuntimeConfig(max_input_tokens=64, max_new_tokens=16),
        baseline_policy_name="turboquant_safe",
    )
    pairs = [(name, parse_int_list(v)) for name, v in args.compressible_heads_for]
    study2 = _apply_per_policy_overrides(study, "compressible_heads", pairs)
    assert study2.policies[0].settings.compressible_heads == (0, 2)
    assert study2.policies[1].settings.compressible_heads is None
