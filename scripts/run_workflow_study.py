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
    args = parser.parse_args()

    summary = run_workflow_study(
        study_config_path=args.study_config,
        output_dir=args.output_dir,
        policy_configs_arg=args.policy_configs,
        model_config_override=args.model_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
