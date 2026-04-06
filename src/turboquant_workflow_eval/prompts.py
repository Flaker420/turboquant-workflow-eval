from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .types import PromptSpec


def _load_yaml(path: str | Path) -> Any:
    """Inline YAML loader.

    Prompt-pack files (``prompts/*.yaml``) are still YAML by design — they
    are content, not configuration, and live in a separate location from
    the dataclass-based study/model/policy configs.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def filter_prompts(
    prompts: list[PromptSpec],
    prompt_ids: list[str] | None = None,
    categories: list[str] | None = None,
    pattern: str | None = None,
) -> list[PromptSpec]:
    """Filter a prompt list by IDs, categories, and/or regex pattern.

    All filters are combined with AND logic (a prompt must match all specified
    filters).  Passing ``None`` for a filter disables it.
    """
    result = list(prompts)
    if prompt_ids is not None:
        id_set = set(prompt_ids)
        result = [p for p in result if p.id in id_set]
    if categories is not None:
        cat_set = {c.lower() for c in categories}
        result = [p for p in result if p.category.lower() in cat_set]
    if pattern is not None:
        regex = re.compile(pattern, re.IGNORECASE)
        result = [p for p in result if regex.search(p.id) or regex.search(p.title)]
    return result


def load_prompt_pack(path: str | Path) -> list[PromptSpec]:
    path = Path(path)
    if not path.exists():
        hint = ""
        if "generated" in path.name:
            hint = " Run 'make generate-prompts' to create it first."
        raise FileNotFoundError(f"Prompt pack not found: {path}.{hint}")
    data = _load_yaml(path)
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
