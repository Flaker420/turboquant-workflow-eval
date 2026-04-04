from __future__ import annotations

from pathlib import Path

from .config import load_yaml
from .types import PromptSpec


BUILTIN_PROMPTS = [
    "Explain a difference between two quantization methods.",
    "Compute a simple growth rate and explain the steps.",
    "Write a short Python helper and describe its complexity.",
    "List likely sections in a technical report.",
]


def load_prompt_source(source: str, prompts_file: str | None = None, max_prompts: int | None = None) -> list[str]:
    """Return plain-text prompts for lightweight operations like preflight warmup.

    For full evaluation studies, use :func:`load_prompt_pack` instead, which
    returns structured :class:`PromptSpec` objects with scoring metadata.
    """
    if source == "builtin":
        prompts = list(BUILTIN_PROMPTS)
    elif source == "file":
        if not prompts_file:
            raise ValueError("prompts_file is required when source=file")
        with Path(prompts_file).open("r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported prompt source: {source}")
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    return prompts


def load_prompt_pack(path: str | Path) -> list[PromptSpec]:
    path = Path(path)
    if not path.exists():
        hint = ""
        if "generated" in path.name:
            hint = " Run 'make generate-prompts' to create it first."
        raise FileNotFoundError(f"Prompt pack not found: {path}.{hint}")
    data = load_yaml(path)
    raw_prompts = data.get("prompts", [])
    prompts: list[PromptSpec] = []
    for item in raw_prompts:
        test_cases = item.get("test_cases")
        if test_cases is not None:
            test_cases = tuple(test_cases)
        turns = item.get("turns")
        if turns is not None:
            turns = tuple(turns)
        prompts.append(
            PromptSpec(
                id=item["id"],
                category=item["category"],
                title=item["title"],
                prompt=item.get("prompt", "").rstrip(),
                watch_for=item.get("watch_for", ""),
                reference_answer=item.get("reference_answer"),
                test_cases=test_cases,
                turns=turns,
            )
        )
    return prompts
