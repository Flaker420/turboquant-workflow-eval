from __future__ import annotations

from pathlib import Path

from turboquant_workflow_eval.prompts import load_prompt_pack


def test_load_prompt_pack(tmp_path: Path) -> None:
    path = tmp_path / "prompts.yaml"
    path.write_text(
        "prompts:\n"
        "  - id: p1\n"
        "    category: reasoning\n"
        "    title: demo\n"
        "    watch_for: signal\n"
        "    prompt: hello\n",
        encoding="utf-8",
    )
    prompts = load_prompt_pack(path)
    assert len(prompts) == 1
    assert prompts[0].id == "p1"
    assert prompts[0].title == "demo"
