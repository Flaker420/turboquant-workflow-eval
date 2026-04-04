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
    assert prompts[0].reference_answer is None
    assert prompts[0].test_cases is None
    assert prompts[0].turns is None


def test_load_prompt_with_reference_answer(tmp_path: Path) -> None:
    path = tmp_path / "prompts.yaml"
    path.write_text(
        "prompts:\n"
        "  - id: m1\n"
        "    category: math\n"
        "    title: cagr\n"
        "    watch_for: correct\n"
        '    reference_answer: "20.51"\n'
        "    prompt: compute cagr\n",
        encoding="utf-8",
    )
    prompts = load_prompt_pack(path)
    assert prompts[0].reference_answer == "20.51"


def test_load_prompt_with_test_cases(tmp_path: Path) -> None:
    path = tmp_path / "prompts.yaml"
    path.write_text(
        "prompts:\n"
        "  - id: c1\n"
        "    category: coding\n"
        "    title: helper\n"
        "    watch_for: works\n"
        "    test_cases:\n"
        '      - input: "[[1,2,3]]"\n'
        '        expected: "6"\n'
        "    prompt: write a sum function\n",
        encoding="utf-8",
    )
    prompts = load_prompt_pack(path)
    assert prompts[0].test_cases is not None
    assert len(prompts[0].test_cases) == 1
    assert prompts[0].test_cases[0]["input"] == "[[1,2,3]]"


def test_load_prompt_with_turns(tmp_path: Path) -> None:
    path = tmp_path / "prompts.yaml"
    path.write_text(
        "prompts:\n"
        "  - id: mt1\n"
        "    category: reasoning\n"
        "    title: multi-turn\n"
        "    watch_for: context retention\n"
        "    prompt: ''\n"
        "    turns:\n"
        "      - role: user\n"
        "        content: hello\n"
        "      - role: assistant\n"
        "        content: hi\n"
        "      - role: user\n"
        "        content: what did I say first?\n",
        encoding="utf-8",
    )
    prompts = load_prompt_pack(path)
    assert prompts[0].turns is not None
    assert len(prompts[0].turns) == 3
    assert prompts[0].turns[0]["role"] == "user"
