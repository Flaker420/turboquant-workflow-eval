from __future__ import annotations

from pathlib import Path

from .config import load_yaml, resolve_relative_path
from .import_utils import load_object
from .prompts import load_prompt_pack
from .reporting import write_csv, write_examples_markdown, write_jsonl, write_run_summary, write_text_outputs


def _load_backend_configs(study_config_path: Path, study_cfg: dict, backend_configs_arg: str | None) -> list[Path]:
    if backend_configs_arg:
        return [Path(item.strip()) for item in backend_configs_arg.split(",") if item.strip()]
    return [resolve_relative_path(study_config_path, item) for item in study_cfg.get("backend_configs", [])]


def run_backend_study(study_config_path: str | Path, output_dir: str | Path, backend_configs_arg: str | None = None) -> dict:
    study_config_path = Path(study_config_path)
    study_cfg = load_yaml(study_config_path)
    prompt_pack = load_prompt_pack(resolve_relative_path(study_config_path, study_cfg["prompt_pack"]))
    backend_paths = _load_backend_configs(study_config_path, study_cfg, backend_configs_arg)
    runtime_cfg = study_cfg["runtime"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    backends_used: list[dict] = []

    for backend_path in backend_paths:
        backend_cfg = load_yaml(backend_path)
        if not bool(backend_cfg.get("enabled", True)):
            continue

        backend_cls = load_object(backend_cfg["backend"]["import_path"])
        backend = backend_cls(backend_cfg)

        metadata = {
            "name": backend_cfg["name"],
            "comparison_label": backend_cfg.get("comparison_label", backend_cfg["name"]),
            "backend_name": getattr(backend, "name", backend.__class__.__name__),
            "backend_description": backend.describe(),
            "backend_path": str(backend_path),
        }
        backends_used.append(metadata)

        for prompt in prompt_pack:
            result = backend.generate_one(prompt.prompt, runtime_cfg)
            rows.append(
                {
                    "policy_name": backend_cfg["name"],
                    "comparison_label": backend_cfg.get("comparison_label", backend_cfg["name"]),
                    "adapter_name": getattr(backend, "name", backend.__class__.__name__),
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "title": prompt.title,
                    "watch_for": prompt.watch_for,
                    "prompt_text": prompt.prompt,
                    **result,
                }
            )

    if not rows:
        raise RuntimeError("No enabled backends were run. Enable at least one backend config.")

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
        "policy_count": len(backends_used),
        "prompt_count": len(prompt_pack),
        "row_count": len(rows),
        "policies_used": backends_used,
        "output_dir": str(output_dir),
    }
    write_run_summary(output_dir / "run_summary.json", summary)
    return summary
