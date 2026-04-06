from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant_workflow_eval.study import run_workflow_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the workflow comparison study.")
    parser.add_argument("--study-config", default="configs/studies/default.yaml")
    parser.add_argument("--policy-configs", default=None, help="Comma-separated policy config paths.")
    parser.add_argument("--model-config", default=None, help="Override the model config from the study YAML.")
    parser.add_argument("--output-dir", default="outputs/study_run")

    # Config overrides
    parser.add_argument(
        "--set", action="append", dest="overrides", metavar="KEY=VALUE", default=[],
        help="Override any config value (e.g. --set runtime.max_new_tokens=128)",
    )
    parser.add_argument("--repetitions", type=int, default=None, help="Override repetition count")
    parser.add_argument(
        "--set-policy", action="append", dest="policy_overrides",
        metavar="NAME.KEY=VALUE", default=[],
        help=(
            "Override a key inside a policy YAML at load time. "
            "Format: '<policy_name|*>.<dot.key>=<value>'. Repeatable."
        ),
    )

    # Prompt filtering
    parser.add_argument("--prompt-id", action="append", dest="prompt_ids", default=None)
    parser.add_argument("--prompt-category", action="append", dest="prompt_categories", default=None)
    parser.add_argument("--prompt-filter", default=None, metavar="REGEX")

    # Execution modes
    parser.add_argument("--single", action="store_true", help="Quick smoke test: 1 prompt, 1 policy, 1 rep")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs without GPU")

    args = parser.parse_args()

    if args.repetitions is not None:
        args.overrides.append(f"runtime.repetitions={args.repetitions}")
    if args.single:
        args.overrides.append("runtime.repetitions=1")

    if args.dry_run:
        from turboquant_workflow_eval.validation import dry_run

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

    summary = run_workflow_study(
        study_config_path=args.study_config,
        output_dir=args.output_dir,
        policy_configs_arg=args.policy_configs,
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
