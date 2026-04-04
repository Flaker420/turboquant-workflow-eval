from __future__ import annotations

import gc
import math
import random
from pathlib import Path
from typing import Any

import torch

from .code_runner import extract_python_code, run_code_with_tests
from .config import load_yaml, resolve_relative_path
from .generation import generate_one
from .import_utils import load_object
from .model_loader import load_model_and_tokenizer
from .prompts import load_prompt_pack
from .reporting import write_csv, write_examples_markdown, write_jsonl, write_run_summary, write_text_outputs
from .scoring import check_reference_answer, compute_semantic_similarity, compute_verdict


def _load_policy_configs(study_config_path: Path, study_cfg: dict, policy_configs_arg: str | None) -> list[Path]:
    if policy_configs_arg:
        return [Path(item.strip()) for item in policy_configs_arg.split(",") if item.strip()]
    return [resolve_relative_path(study_config_path, item) for item in study_cfg.get("policy_configs", [])]


def _aggregate_stats(values: list[float]) -> dict[str, float]:
    """Compute mean and std for a list of values."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "std": 0.0}
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return {"mean": mean, "std": math.sqrt(variance)}


def run_workflow_study(study_config_path: str | Path, output_dir: str | Path, policy_configs_arg: str | None = None) -> dict:
    study_config_path = Path(study_config_path)
    study_cfg = load_yaml(study_config_path)
    model_cfg = load_yaml(resolve_relative_path(study_config_path, study_cfg["model_config"]))

    # Support single path or list of paths for prompt_pack
    prompt_pack_cfg = study_cfg["prompt_pack"]
    if isinstance(prompt_pack_cfg, list):
        prompt_pack = []
        for pp in prompt_pack_cfg:
            prompt_pack.extend(load_prompt_pack(resolve_relative_path(study_config_path, pp)))
    else:
        prompt_pack = load_prompt_pack(resolve_relative_path(study_config_path, prompt_pack_cfg))

    policy_paths = _load_policy_configs(study_config_path, study_cfg, policy_configs_arg)
    runtime_cfg = study_cfg["runtime"]
    thresholds_cfg = study_cfg.get("thresholds", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repetitions = int(runtime_cfg.get("repetitions", 1))

    # Rec 9: optionally shuffle policy order
    if bool(runtime_cfg.get("shuffle_policies", False)):
        seed = int(runtime_cfg.get("shuffle_seed", 42))
        rng = random.Random(seed)
        policy_paths = list(policy_paths)
        rng.shuffle(policy_paths)

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

        # Rec 1: warmup — prime JIT/CUDA kernels with an untimed generation
        generate_one(model, tokenizer, "Say hello.", runtime_cfg)

        for prompt in prompt_pack:
            # First run: capture full result including text
            first_result = generate_one(model, tokenizer, prompt.prompt, runtime_cfg, turns=prompt.turns)

            latencies = [first_result["latency_s"]]
            tps_values = [first_result["tokens_per_second"]] if first_result["tokens_per_second"] is not None else []
            vram_values = [first_result["peak_vram_gb"]] if first_result["peak_vram_gb"] is not None else []

            # Rec 6: additional repetitions (text is deterministic, only collect metrics)
            for _ in range(repetitions - 1):
                rep_result = generate_one(model, tokenizer, prompt.prompt, runtime_cfg, turns=prompt.turns)
                latencies.append(rep_result["latency_s"])
                if rep_result["tokens_per_second"] is not None:
                    tps_values.append(rep_result["tokens_per_second"])
                if rep_result["peak_vram_gb"] is not None:
                    vram_values.append(rep_result["peak_vram_gb"])

            lat_stats = _aggregate_stats(latencies)
            tps_stats = _aggregate_stats(tps_values)
            vram_stats = _aggregate_stats(vram_values)

            row: dict[str, Any] = {
                "policy_name": policy_cfg["name"],
                "comparison_label": policy_cfg.get("comparison_label", policy_cfg["name"]),
                "adapter_name": getattr(adapter, "name", adapter.__class__.__name__),
                "prompt_id": prompt.id,
                "category": prompt.category,
                "title": prompt.title,
                "watch_for": prompt.watch_for,
                "prompt_text": prompt.prompt,
                **first_result,
            }

            # Add repetition stats
            if repetitions > 1:
                row["latency_mean"] = lat_stats["mean"]
                row["latency_std"] = lat_stats["std"]
                row["tps_mean"] = tps_stats["mean"]
                row["tps_std"] = tps_stats["std"]
                row["vram_mean"] = vram_stats["mean"]
                row["vram_std"] = vram_stats["std"]
                row["repetitions"] = repetitions

            # Rec 2: math reference-answer checking
            row["math_correct"] = check_reference_answer(
                first_result["output_text"],
                prompt.reference_answer,
            )

            # Rec 5: code execution for coding prompts
            if prompt.test_cases:
                code = extract_python_code(first_result["output_text"])
                if code:
                    code_result = run_code_with_tests(code, list(prompt.test_cases))
                    row["code_passed"] = code_result["passed"]
                    row["code_failed"] = code_result["failed"]
                    row["code_errors"] = code_result["errors"]
                    row["code_verdict"] = code_result["verdict"]
                else:
                    row["code_verdict"] = "error"
                    row["code_passed"] = 0
                    row["code_failed"] = 0
                    row["code_errors"] = 1

            rows.append(row)

        adapter.cleanup(model)
        if bool(runtime_cfg.get("cleanup_between_policies", True)):
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No enabled policies were run. Enable at least one policy config.")

    # --- Post-processing: baseline deltas, similarity, verdicts ---

    # Build baseline lookup (first policy encountered per prompt_id)
    baseline_by_prompt: dict[str, dict] = {}
    for row in rows:
        pid = row["prompt_id"]
        if pid not in baseline_by_prompt:
            baseline_by_prompt[pid] = row

    for row in rows:
        pid = row["prompt_id"]
        bl = baseline_by_prompt.get(pid)

        # Rec 3: output-length delta vs baseline
        bl_tokens = bl["output_tokens"] if bl else 0
        if bl_tokens and bl_tokens > 0:
            row["output_length_delta_pct"] = ((row["output_tokens"] - bl_tokens) / bl_tokens) * 100
        else:
            row["output_length_delta_pct"] = 0.0

        # Rec 7: semantic similarity vs baseline
        if bl and bl is not row:
            row["semantic_similarity"] = compute_semantic_similarity(
                bl["output_text"], row["output_text"]
            )
        else:
            row["semantic_similarity"] = None

        # Rec 8: verdict
        row["verdict"] = compute_verdict(
            row,
            bl if bl is not row else None,
            thresholds_cfg,
        )

    # --- Write outputs ---
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

    # Verdict summary
    verdict_counts = {"green": 0, "yellow": 0, "red": 0}
    for row in rows:
        v = row.get("verdict", "green")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    summary = {
        "study_name": study_cfg["name"],
        "study_config": str(study_config_path),
        "model_name": model_cfg["model_name"],
        "policy_count": len(policies_used),
        "prompt_count": len(prompt_pack),
        "row_count": len(rows),
        "repetitions": repetitions,
        "verdict_summary": verdict_counts,
        "policies_used": policies_used,
        "output_dir": str(output_dir),
    }
    write_run_summary(output_dir / "run_summary.json", summary)
    return summary
