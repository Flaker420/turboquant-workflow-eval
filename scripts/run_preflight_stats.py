from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import importlib.util
import hashlib

from turboquant_workflow_eval.loader import load_model_module
from turboquant_workflow_eval.model_loader import load_model_and_tokenizer, resolve_language_model_root
from turboquant_workflow_eval.module_discovery import discover_attention_blocks
from turboquant_workflow_eval.preflight import run_preflight
from turboquant_workflow_eval.prompts import load_prompt_source
from turboquant_workflow_eval.schema import model_to_legacy_dict


def _load_experiment_module(path: str | Path) -> dict:
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {file_path}")
    digest = hashlib.sha1(str(file_path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(
        f"turboquant_eval_experiment_{file_path.stem}_{digest}", file_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "EXPERIMENT"):
        raise RuntimeError(
            f"Experiment config {file_path} must define a top-level EXPERIMENT dict"
        )
    return module.EXPERIMENT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a preflight Q/K/V stats pass.")
    parser.add_argument("--experiment-config", default="configs/experiments/preflight_stats.py")
    parser.add_argument("--output-dir", default="outputs/preflight")
    parser.add_argument("--prompts-file", default=None)
    args = parser.parse_args()

    exp_cfg = _load_experiment_module(args.experiment_config)
    model_cfg = model_to_legacy_dict(load_model_module(exp_cfg["model_config_path"]))

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
