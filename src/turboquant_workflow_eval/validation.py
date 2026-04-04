"""Dry-run validation: verify all configs, paths, and adapter imports without GPU."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .config import (
    ConfigValidationError,
    load_yaml,
    load_yaml_with_overrides,
    resolve_relative_path,
    validate_config,
)
from .download import check_cache_status
from .import_utils import load_object
from .prompts import filter_prompts, load_prompt_pack


def dry_run(
    study_config_path: str | Path,
    overrides: list[str] | None = None,
    policy_configs_arg: str | None = None,
    model_config_override: str | Path | None = None,
    prompt_ids: list[str] | None = None,
    prompt_categories: list[str] | None = None,
    prompt_pattern: str | None = None,
) -> int:
    """Validate a study configuration without touching the GPU.

    Loads all configs, resolves paths, filters prompts, resolves adapter
    classes, and checks model cache status.  Prints a formatted execution
    plan and returns 0 on success, 1 on any validation error.
    """
    errors: list[str] = []
    study_config_path = Path(study_config_path)

    # --- Load and validate study config ---
    print(f"Study config: {study_config_path}")
    try:
        study_cfg = load_yaml_with_overrides(study_config_path, overrides=overrides)
        validate_config(study_cfg, "study", study_config_path)
    except (ConfigValidationError, TypeError, FileNotFoundError, ValueError) as exc:
        errors.append(f"Study config: {exc}")
        _print_errors(errors)
        return 1

    # --- Resolve and validate model config ---
    if model_config_override:
        model_cfg_path = Path(model_config_override)
    else:
        model_cfg_path = resolve_relative_path(study_config_path, study_cfg["model_config"])

    print(f"Model config: {model_cfg_path}")
    if not model_cfg_path.exists():
        errors.append(f"Model config not found: {model_cfg_path}")
    else:
        try:
            model_cfg = load_yaml(model_cfg_path)
            validate_config(model_cfg, "model", model_cfg_path)
        except (ConfigValidationError, TypeError, ValueError) as exc:
            errors.append(f"Model config: {exc}")

    # --- Load and filter prompts ---
    prompt_pack_cfg = study_cfg["prompt_pack"]
    prompt_paths: list[Path] = []
    if isinstance(prompt_pack_cfg, list):
        for pp in prompt_pack_cfg:
            prompt_paths.append(resolve_relative_path(study_config_path, pp))
    else:
        prompt_paths.append(resolve_relative_path(study_config_path, prompt_pack_cfg))

    prompts = []
    for pp_path in prompt_paths:
        if not pp_path.exists():
            errors.append(f"Prompt pack not found: {pp_path}")
        else:
            try:
                prompts.extend(load_prompt_pack(pp_path))
            except Exception as exc:
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

    # --- Resolve policy configs ---
    if policy_configs_arg:
        policy_paths = [Path(p.strip()) for p in policy_configs_arg.split(",") if p.strip()]
    else:
        policy_paths = [
            resolve_relative_path(study_config_path, item)
            for item in study_cfg.get("policy_configs", [])
        ]

    enabled_policies: list[dict[str, Any]] = []
    for pp in policy_paths:
        if not pp.exists():
            errors.append(f"Policy config not found: {pp}")
            continue
        try:
            pcfg = load_yaml(pp)
            validate_config(pcfg, "policy", pp)
        except (ConfigValidationError, TypeError, ValueError) as exc:
            errors.append(f"Policy config {pp}: {exc}")
            continue

        if not bool(pcfg.get("enabled", False)):
            print(f"  Policy '{pcfg.get('name', '?')}' — disabled, skipping")
            continue

        # Verify adapter import resolves
        adapter_import = pcfg.get("adapter", {}).get("import_path", "")
        try:
            load_object(adapter_import)
            print(f"  Policy '{pcfg['name']}' — adapter OK ({adapter_import})")
        except Exception as exc:
            errors.append(f"Policy '{pcfg.get('name', '?')}' adapter import failed: {exc}")
            continue

        enabled_policies.append(pcfg)

    if not enabled_policies:
        errors.append("No enabled policies found")

    # --- Runtime config ---
    runtime_cfg = study_cfg.get("runtime", {})
    repetitions = int(runtime_cfg.get("repetitions", 1))

    # --- Model cache check ---
    if not errors and model_cfg_path.exists():
        model_name = model_cfg.get("model_name", "unknown")
        cache = check_cache_status(model_name)
        cache_status = "cached" if cache["model_cached"] else "NOT cached (will download)"
        print(f"Model: {model_name} — {cache_status}")

    # --- Summary ---
    if errors:
        _print_errors(errors)
        return 1

    total_gens = len(prompts) * len(enabled_policies) * repetitions
    print()
    print("=" * 60)
    print("  DRY RUN — Execution Plan")
    print("=" * 60)
    print(f"  Study:       {study_cfg.get('name', '?')}")
    print(f"  Model:       {model_cfg.get('model_name', '?')}")
    print(f"  Policies:    {len(enabled_policies)}")
    for p in enabled_policies:
        print(f"               - {p['name']}")
    print(f"  Prompts:     {len(prompts)}")
    categories = {}
    for p in prompts:
        categories[p.category] = categories.get(p.category, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"               - {cat}: {count}")
    print(f"  Repetitions: {repetitions}")
    print(f"  Total gens:  {total_gens}")
    print(f"  Max tokens:  {runtime_cfg.get('max_new_tokens', '?')}")
    print("=" * 60)
    print("  All configs valid. Ready to run.")
    print("=" * 60)
    return 0


def _print_errors(errors: list[str]) -> None:
    print()
    print("VALIDATION ERRORS:")
    for err in errors:
        print(f"  - {err}")
