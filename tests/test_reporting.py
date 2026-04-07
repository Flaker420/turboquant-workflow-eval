from __future__ import annotations

from pathlib import Path

import json

from turboquant_workflow_eval.reporting import (
    IncrementalWriter,
    slugify,
    summarize_divergence,
    write_csv,
    write_examples_markdown,
    write_jsonl,
)


def test_slugify() -> None:
    assert slugify("Hello, World!") == "hello-world"


def _baseline_row() -> dict:
    return {
        "policy_name": "baseline",
        "comparison_label": "baseline",
        "adapter_name": "none",
        "prompt_id": "p1",
        "category": "reasoning",
        "title": "demo",
        "watch_for": "signal",
        "prompt_text": "prompt",
        "prompt_tokens": 5,
        "output_tokens": 3,
        "latency_s": 0.25,
        "tokens_per_second": 12.0,
        "peak_vram_gb": None,
        "output_text": "answer",
        "output_token_ids": [1, 2, 3],
        # Divergence sentinels (filled by score_results in real runs).
        "exact_match": True,
        "first_divergence_token": -1,
        "common_prefix_tokens": 3,
        "common_prefix_frac": 1.0,
        "token_edit_distance": 0,
        "output_length_delta_tokens": 0,
        # KV cache sentinels.
        "kv_cache_bytes_baseline": 1000,
        "kv_cache_bytes_policy": 1000,
        "kv_cache_compression_ratio": 1.0,
        "kv_cache_bytes_saved": 0,
    }


def _policy_row() -> dict:
    row = _baseline_row()
    row.update({
        "policy_name": "turbo_safe",
        "comparison_label": "safe",
        "adapter_name": "turboquant",
        "output_token_ids": [1, 2, 9],
        "exact_match": False,
        "first_divergence_token": 2,
        "common_prefix_tokens": 2,
        "common_prefix_frac": 2 / 3,
        "token_edit_distance": 1,
        "output_length_delta_tokens": 0,
        "kv_cache_bytes_policy": 250,
        "kv_cache_compression_ratio": 4.0,
        "kv_cache_bytes_saved": 750,
    })
    return row


def test_reporting_outputs(tmp_path: Path) -> None:
    rows = [_baseline_row(), _policy_row()]
    write_jsonl(tmp_path / "rows.jsonl", rows)
    write_csv(tmp_path / "workflow_compare.csv", rows, truncate_output_to_chars=40)
    write_examples_markdown(tmp_path / "examples.md", rows)
    assert (tmp_path / "rows.jsonl").exists()
    assert (tmp_path / "workflow_compare.csv").exists()
    assert (tmp_path / "examples.md").exists()


def test_csv_has_new_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "out.csv"
    write_csv(csv_path, [_baseline_row(), _policy_row()])
    header = csv_path.read_text().splitlines()[0]
    assert "exact_match" in header
    assert "first_divergence_token" in header
    assert "common_prefix_frac" in header
    assert "token_edit_distance" in header
    assert "kv_cache_compression_ratio" in header
    assert "kv_cache_bytes_saved" in header
    # And the legacy verdict-system columns are gone.
    assert "verdict" not in header
    assert "math_correct" not in header
    assert "semantic_similarity" not in header
    assert "code_verdict" not in header
    assert "output_length_delta_pct" not in header


def test_csv_with_repetition_stats(tmp_path: Path) -> None:
    rows = [_baseline_row()]
    rows[0].update({
        "latency_mean": 0.25,
        "latency_std": 0.01,
        "tps_mean": 12.0,
        "tps_std": 0.5,
        "vram_mean": 1.5,
        "vram_std": 0.1,
        "repetitions": 3,
    })
    csv_path = tmp_path / "out.csv"
    write_csv(csv_path, rows)
    header = csv_path.read_text().splitlines()[0]
    assert "latency_mean" in header
    assert "latency_std" in header
    assert "repetitions" in header


def _minimal_finalize_args() -> dict:
    return {
        "study_cfg": {"name": "s"},
        "model_cfg": {"model_name": "m"},
        "policies_used": [{"name": "baseline"}],
        "prompt_count": 1,
        "repetitions": 1,
    }


def test_incremental_finalize_with_baseline_policy_name(tmp_path: Path) -> None:
    writer = IncrementalWriter(tmp_path)
    writer.write_row(_baseline_row())
    summary = writer.finalize(**_minimal_finalize_args(), baseline_policy_name="baseline")
    assert summary["baseline_policy_name"] == "baseline"
    on_disk = json.loads((tmp_path / "run_summary.json").read_text())
    assert on_disk["baseline_policy_name"] == "baseline"
    assert "divergence_summary" in on_disk
    assert "verdict_summary" not in on_disk


def test_incremental_finalize_without_baseline_policy_name(tmp_path: Path) -> None:
    writer = IncrementalWriter(tmp_path)
    writer.write_row(_baseline_row())
    summary = writer.finalize(**_minimal_finalize_args())
    assert summary["baseline_policy_name"] is None
    assert summary["study_name"] == "s"
    assert summary["row_count"] == 1
    assert "divergence_summary" in summary
    assert "verdict_summary" not in summary
    on_disk = json.loads((tmp_path / "run_summary.json").read_text())
    assert on_disk["baseline_policy_name"] is None


# ---------------------------------------------------------------------------
# summarize_divergence
# ---------------------------------------------------------------------------


def test_summarize_divergence_excludes_baseline_and_aggregates() -> None:
    rows = [
        _baseline_row(),
        _policy_row(),
        {**_policy_row(), "exact_match": True, "first_divergence_token": 3,
         "token_edit_distance": 0, "common_prefix_frac": 1.0,
         "kv_cache_compression_ratio": 4.0, "kv_cache_bytes_saved": 750,
         "peak_vram_gb": 6.2},
    ]
    summary = summarize_divergence(rows, baseline_policy_name="baseline")
    assert "baseline" not in summary
    assert "turbo_safe" in summary
    s = summary["turbo_safe"]
    assert s["prompts"] == 2
    assert s["exact_match_count"] == 1
    assert s["exact_match_rate"] == 0.5
    assert s["max_token_edit_distance"] == 1
    assert s["mean_kv_cache_compression_ratio"] == 4.0


def test_summarize_divergence_handles_only_errors() -> None:
    err_row = _policy_row()
    err_row["error"] = "boom"
    summary = summarize_divergence([_baseline_row(), err_row], baseline_policy_name="baseline")
    assert summary["turbo_safe"]["prompts"] == 0
    assert summary["turbo_safe"]["errors"] == 1
    assert summary["turbo_safe"]["mean_kv_cache_compression_ratio"] is None
