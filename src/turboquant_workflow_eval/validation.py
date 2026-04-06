"""Dry-run validation: verify all configs, paths, and adapter imports without GPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .download import check_cache_status
from .import_utils import load_object
from .loader import ConfigLoadError, load_study_module
from .prompts import filter_prompts, load_prompt_pack
from .schema import ConfigValidationError, StudyConfig


def dry_run(
    study: StudyConfig | str | Path,
    *,
    prompt_ids: list[str] | None = None,
    prompt_categories: list[str] | None = None,
    prompt_pattern: str | None = None,
) -> int:
    """Validate a study configuration without touching the GPU.

    Loads prompt packs, resolves adapter classes, and checks model cache
    status. Prints a formatted execution plan and returns 0 on success,
    1 on any validation error.

    Accepts either a fully-resolved :class:`StudyConfig` (the modern path,
    used after the CLI has applied ``--set`` / ``--KNOB`` overrides) or a
    path to a Python config module for one-shot loading.
    """
    errors: list[str] = []

    # --- Load study config ---
    if isinstance(study, (str, Path)):
        print(f"Study config: {study}")
        try:
            study = load_study_module(study)
        except (ConfigLoadError, ConfigValidationError, FileNotFoundError) as exc:
            errors.append(f"Study config: {exc}")
            _print_errors(errors)
            return 1
    else:
        print(f"Study config: <inline {study.name!r}>")

    print(f"Model: {study.model.model_name} ({study.model.dtype})")

    # --- Load and filter prompts ---
    prompts: list = []
    for pp_path in study.prompt_pack:
        if not Path(pp_path).exists():
            errors.append(f"Prompt pack not found: {pp_path}")
        else:
            try:
                prompts.extend(load_prompt_pack(pp_path))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Prompt pack {pp_path}: {exc}")

    prompts = filter_prompts(
        prompts,
        prompt_ids=prompt_ids,
        categories=prompt_categories,
        pattern=prompt_pattern,
    )
    print(f"Prompts: {len(prompts)} selected")
    if not prompts:
        errors.append("No prompts matched the specified filters")

    # --- Resolve policy adapters ---
    enabled_policies = []
    for policy in study.policies:
        if not policy.enabled:
            print(f"  Policy '{policy.name}' — disabled, skipping")
            continue
        try:
            load_object(policy.adapter.import_path)
            print(f"  Policy '{policy.name}' — adapter OK ({policy.adapter.import_path})")
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"Policy '{policy.name}' adapter import failed: {exc}"
            )
            continue
        enabled_policies.append(policy)

    if not enabled_policies:
        errors.append("No enabled policies found")

    # --- Model cache check ---
    if not errors:
        cache = check_cache_status(study.model.model_name)
        cache_status = "cached" if cache["model_cached"] else "NOT cached (will download)"
        print(f"Model: {study.model.model_name} — {cache_status}")

    # --- Summary ---
    if errors:
        _print_errors(errors)
        return 1

    total_gens = len(prompts) * len(enabled_policies) * study.runtime.repetitions
    print()
    print("=" * 60)
    print("  DRY RUN — Execution Plan")
    print("=" * 60)
    print(f"  Study:       {study.name}")
    print(f"  Model:       {study.model.model_name}")
    print(f"  Policies:    {len(enabled_policies)}")
    for p in enabled_policies:
        print(f"               - {p.name}")
    print(f"  Prompts:     {len(prompts)}")
    categories: dict[str, int] = {}
    for p in prompts:
        categories[p.category] = categories.get(p.category, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"               - {cat}: {count}")
    print(f"  Repetitions: {study.runtime.repetitions}")
    print(f"  Total gens:  {total_gens}")
    print(f"  Max tokens:  {study.runtime.max_new_tokens}")
    print("=" * 60)
    print("  All configs valid. Ready to run.")
    print("=" * 60)
    return 0


def _print_errors(errors: list[str]) -> None:
    print()
    print("VALIDATION ERRORS:")
    for err in errors:
        print(f"  - {err}")
