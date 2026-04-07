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
