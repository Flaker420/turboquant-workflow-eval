"""CLI entry point: ``python -m turboquant_workflow_eval``."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="turboquant_workflow_eval",
        description="Run a workflow evaluation study comparing compression policies.",
    )

    # --- Core arguments (backward-compatible) ---
    parser.add_argument(
        "--study-config",
        default=None,
        help=(
            "Path to the study configuration YAML "
            "(e.g. configs/studies/default.yaml). Required for normal/dry-run "
            "modes; optional for --rescore (used as a thresholds source)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for result artefacts (default: outputs/)",
    )
    parser.add_argument(
        "--policy-configs",
        default=None,
        help="Comma-separated policy config paths (overrides study config)",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Override the model config from the study YAML",
    )

    # --- New: config overrides ---
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Override any config value using dot-notation "
            "(e.g. --set runtime.max_new_tokens=128 --set thresholds.latency_red_pct=50). "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Override repetition count (shorthand for --set runtime.repetitions=N)",
    )
    parser.add_argument(
        "--set-policy",
        action="append",
        dest="policy_overrides",
        metavar="NAME.KEY=VALUE",
        default=[],
        help=(
            "Override a key inside a policy YAML at load time. Format: "
            "'<policy_name|*>.<dot.key>=<value>'. The first segment matches "
            "policy_cfg['name']; '*' matches every policy. Examples: "
            "--set-policy turboquant_safe.settings.key_strategy=mse "
            "--set-policy '*.settings.bit_width=8' "
            "--set-policy baseline.enabled=false. Can be repeated."
        ),
    )

    # --- New: prompt filtering ---
    parser.add_argument(
        "--prompt-id",
        action="append",
        dest="prompt_ids",
        default=None,
        help="Run only the specified prompt ID (can be repeated for multiple IDs)",
    )
    parser.add_argument(
        "--prompt-category",
        action="append",
        dest="prompt_categories",
        default=None,
        help="Run only prompts in the specified category (can be repeated)",
    )
    parser.add_argument(
        "--prompt-filter",
        default=None,
        metavar="REGEX",
        help="Filter prompts by regex on id/title",
    )

    # --- New: execution modes ---
    parser.add_argument(
        "--single",
        action="store_true",
        help="Quick smoke test: first matching prompt, first policy, 1 repetition",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all configs and print execution plan without touching the GPU",
    )
    parser.add_argument(
        "--rescore",
        default=None,
        metavar="ROWS_JSONL",
        help="Re-score existing results with new thresholds (no GPU). Use with --set for threshold overrides.",
    )

    args = parser.parse_args(argv)

    # Fold --repetitions into --set overrides
    if args.repetitions is not None:
        args.overrides.append(f"runtime.repetitions={args.repetitions}")

    # --single implies 1 repetition
    if args.single:
        args.overrides.append("runtime.repetitions=1")

    # --- Dry-run mode ---
    if args.dry_run:
        if not args.study_config:
            parser.error("--study-config is required for --dry-run")
        from .validation import dry_run

        exit_code = dry_run(
            study_config_path=args.study_config,
            overrides=args.overrides or None,
            policy_configs_arg=args.policy_configs,
            model_config_override=args.model_config,
            policy_overrides=args.policy_overrides or None,
            prompt_ids=args.prompt_ids,
            prompt_categories=args.prompt_categories,
            prompt_pattern=args.prompt_filter,
        )
        sys.exit(exit_code)

    # --- Re-score mode ---
    if args.rescore:
        from .rescoring import rescore

        rescore(
            rows_jsonl_path=args.rescore,
            study_config=args.study_config,
            overrides=args.overrides or None,
            output_dir=args.output_dir if args.output_dir != "outputs" else None,
        )
        return

    if not args.study_config:
        parser.error("--study-config is required unless --rescore is given")

    # --- Normal run ---
    from .study import run_workflow_study

    summary = run_workflow_study(
        study_config_path=args.study_config,
        output_dir=args.output_dir,
        policy_configs_arg=args.policy_configs,
        runtime_overrides=None,
        model_config_override=args.model_config,
        config_overrides=args.overrides or None,
        policy_overrides=args.policy_overrides or None,
        prompt_ids=args.prompt_ids,
        prompt_categories=args.prompt_categories,
        prompt_pattern=args.prompt_filter,
        single_mode=args.single,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
