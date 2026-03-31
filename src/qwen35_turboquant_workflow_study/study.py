from __future__ import annotations

import gc
from pathlib import Path

import torch

from .config import load_yaml, resolve_relative_path
from .generation import generate_one
from .import_utils import load_object
from .model_loader import load_model_and_tokenizer
from .prompts import load_prompt_pack
from .reporting import write_csv, write_examples_markdown, write_jsonl, write_run_summary, write_text_outputs


def _load_policy_configs(study_config_path: Path, study_cfg: dict, policy_configs_arg: str | None) -> list[Path]:
    if policy_configs_arg:
        return [Path(item.strip()) for item in policy_configs_arg.split(",") if item.strip()]
    return [resolve_relative_path(study_config_path, item) for item in study_cfg.get("policy_configs", [])]


def run_workflow_study(study_config_path: str | Path, output_dir: str | Path, policy_configs_arg: str | None = None) -> dict:
    study_config_path = Path(study_config_path)
    study_cfg = load_yaml(study_config_path)
    model_cfg = load_yaml(resolve_relative_path(study_config_path, study_cfg["model_config"]))
    prompt_pack = load_prompt_pack(resolve_relative_path(study_config_path, study_cfg["prompt_pack"]))
    policy_paths = _load_policy_configs(study_config_path, study_cfg, policy_configs_arg)
    runtime_cfg = study_cfg["runtime"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    policies_used: list[dict] = []

    for policy_path in policy_paths:
        policy_cfg = load_yaml(policy_path)
        if not bool(policy_cfg.get("enabled", False)):
            continue

        adapter_cls = load_object(policy_cfg["adapter"]["import_path"])
        adapter = adapter_cls()

        model, tokenizer, loader_name = load_model_and_tokenizer(model_cfg)
        model, tokenizer = adapter.prepare_model(model, tokenizer, model_cfg, policy_cfg)

        policy_metadata = {
            "name": policy_cfg["name"],
            "comparison_label": policy_cfg.get("comparison_label", policy_cfg["name"]),
            "adapter_name": getattr(adapter, "name", adapter.__class__.__name__),
            "loader_name": loader_name,
            "adapter_description": adapter.describe(policy_cfg),
            "policy_path": str(policy_path),
        }
        policies_used.append(policy_metadata)

        for prompt in prompt_pack:
            result = generate_one(model, tokenizer, prompt.prompt, runtime_cfg)
            rows.append(
                {
                    "policy_name": policy_cfg["name"],
                    "comparison_label": policy_cfg.get("comparison_label", policy_cfg["name"]),
                    "adapter_name": getattr(adapter, "name", adapter.__class__.__name__),
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "title": prompt.title,
                    "watch_for": prompt.watch_for,
                    "prompt_text": prompt.prompt,
                    **result,
                }
            )

        adapter.cleanup(model)
        if bool(runtime_cfg.get("cleanup_between_policies", True)):
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No enabled policies were run. Enable at least one policy config.")

    outputs_cfg = study_cfg.get("outputs", {})
    write_jsonl(output_dir / "rows.jsonl", rows)
    write_csv(
        output_dir / "workflow_compare.csv",
        rows,
        truncate_output_to_chars=int(outputs_cfg.get("truncate_csv_output_to_chars", 180)),
    )
    if bool(outputs_cfg.get("write_individual_text_files", True)):
        write_text_outputs(output_dir, rows)
    write_examples_markdown(output_dir / "examples.md", rows)

    summary = {
        "study_name": study_cfg["name"],
        "study_config": str(study_config_path),
        "model_name": model_cfg["model_name"],
        "policy_count": len(policies_used),
        "prompt_count": len(prompt_pack),
        "row_count": len(rows),
        "policies_used": policies_used,
        "output_dir": str(output_dir),
    }
    write_run_summary(output_dir / "run_summary.json", summary)
    return summary
