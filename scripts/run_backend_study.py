from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen35_turboquant_workflow_study.backend_study import run_backend_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the backend-based workflow comparison study.")
    parser.add_argument("--study-config", default="configs/backend_studies/default.yaml")
    parser.add_argument("--backend-configs", default=None, help="Comma-separated backend config paths.")
    parser.add_argument("--output-dir", default="outputs/backend_study_run")
    args = parser.parse_args()

    summary = run_backend_study(
        study_config_path=args.study_config,
        output_dir=args.output_dir,
        backend_configs_arg=args.backend_configs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
