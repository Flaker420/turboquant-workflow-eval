"""Preflight stats experiment config (Phase-0 discovery pass).

Exports a plain dict consumed by ``scripts/run_preflight_stats.py``. Not a
schema-validated dataclass because the preflight tooling has its own narrow
config shape that does not overlap with study/model/policy configs.
"""

from pathlib import Path

_HERE = Path(__file__).resolve().parent

EXPERIMENT: dict = {
    "name": "preflight_stats",
    "phase": 0,
    "description": "Discovery and tensor-statistics pass before any compression work.",
    "model_config_path": _HERE.parent / "model" / "qwen35_9b_text_only.py",
    "expected_attention_blocks": 8,
    "prompts": {
        "source": "builtin",
        "max_prompts": 4,
    },
    "runtime": {
        "max_length": 1024,
        "use_cache": False,
    },
    "output": {
        "filename": "preflight_report.json",
    },
}
