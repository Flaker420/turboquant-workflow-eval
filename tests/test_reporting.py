from __future__ import annotations

from pathlib import Path

from turboquant_workflow_eval.reporting import slugify, write_csv, write_examples_markdown, write_jsonl


def test_slugify() -> None:
    assert slugify("Hello, World!") == "hello-world"


def _make_rows() -> list[dict]:
    return [
        {
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
            "output_length_delta_pct": 0.0,
            "latency_s": 0.25,
            "tokens_per_second": 12.0,
            "peak_vram_gb": None,
            "output_text": "answer",
            "math_correct": None,
            "code_verdict": "",
            "semantic_similarity": None,
            "verdict": "green",
        }
    ]


def test_reporting_outputs(tmp_path: Path) -> None:
    rows = _make_rows()
    write_jsonl(tmp_path / "rows.jsonl", rows)
    write_csv(tmp_path / "workflow_compare.csv", rows, truncate_output_to_chars=40)
    write_examples_markdown(tmp_path / "examples.md", rows)
    assert (tmp_path / "rows.jsonl").exists()
    assert (tmp_path / "workflow_compare.csv").exists()
    assert (tmp_path / "examples.md").exists()


def test_csv_has_new_columns(tmp_path: Path) -> None:
    rows = _make_rows()
    csv_path = tmp_path / "out.csv"
    write_csv(csv_path, rows)
    header = csv_path.read_text().splitlines()[0]
    assert "output_length_delta_pct" in header
    assert "verdict" in header
    assert "math_correct" in header
    assert "semantic_similarity" in header
    assert "code_verdict" in header


def test_csv_with_repetition_stats(tmp_path: Path) -> None:
    rows = _make_rows()
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
