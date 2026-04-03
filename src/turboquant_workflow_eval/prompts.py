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
    data = load_yaml(path)
    raw_prompts = data.get("prompts", [])
    prompts: list[PromptSpec] = []
    for item in raw_prompts:
        prompts.append(
            PromptSpec(
                id=item["id"],
                category=item["category"],
                title=item["title"],
                prompt=item["prompt"].rstrip(),
                watch_for=item.get("watch_for", ""),
            )
        )
    return prompts
