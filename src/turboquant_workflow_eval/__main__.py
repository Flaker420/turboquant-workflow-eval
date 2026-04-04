"""CLI entry point: ``python -m turboquant_workflow_eval``."""

from __future__ import annotations

import argparse
import json

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="turboquant_workflow_eval",
        description="Run a workflow evaluation study comparing compression policies.",
    )
    parser.add_argument(
        "--study-config",
        required=True,
        help="Path to the study configuration YAML (e.g. configs/studies/default.yaml)",
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
    args = parser.parse_args(argv)

    from .study import run_workflow_study

    summary = run_workflow_study(
        study_config_path=args.study_config,
        output_dir=args.output_dir,
        policy_configs_arg=args.policy_configs,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
