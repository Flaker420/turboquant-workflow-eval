from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant_workflow_eval.config import load_yaml
from turboquant_workflow_eval.model_loader import load_model_and_tokenizer, resolve_language_model_root
from turboquant_workflow_eval.module_discovery import discover_attention_blocks
from turboquant_workflow_eval.preflight import run_preflight
from turboquant_workflow_eval.prompts import load_prompt_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a preflight Q/K/V stats pass.")
    parser.add_argument("--experiment-config", default="configs/experiments/preflight_stats.yaml")
    parser.add_argument("--output-dir", default="outputs/preflight")
    parser.add_argument("--prompts-file", default=None)
    args = parser.parse_args()

    exp_cfg = load_yaml(args.experiment_config)
    model_cfg = load_yaml(exp_cfg["model_config"])

    prompts = load_prompt_source(
        source=exp_cfg["prompts"]["source"],
        prompts_file=args.prompts_file,
        max_prompts=exp_cfg["prompts"].get("max_prompts"),
    )

    model, tokenizer, loader_name = load_model_and_tokenizer(model_cfg)
    lm_root = resolve_language_model_root(model)
    blocks = discover_attention_blocks(
        lm_root,
        expected_count=exp_cfg["expected_attention_blocks"],
    )

    report = run_preflight(
        model=model,
        tokenizer=tokenizer,
        language_model_root=lm_root,
        attention_blocks=blocks,
        prompts=prompts,
        max_length=exp_cfg["runtime"]["max_length"],
        use_cache=bool(exp_cfg["runtime"].get("use_cache", False)),
        loader_name=loader_name,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / exp_cfg["output"]["filename"]
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
