from pathlib import Path

from qwen35_turboquant_workflow_study.reporting import slugify, write_csv, write_examples_markdown, write_jsonl


def test_slugify() -> None:
    assert slugify("Hello, World!") == "hello-world"


def test_reporting_outputs(tmp_path: Path) -> None:
    rows = [
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
            "latency_s": 0.25,
            "tokens_per_second": 12.0,
            "peak_vram_gb": None,
            "output_text": "answer",
        }
    ]
    write_jsonl(tmp_path / "rows.jsonl", rows)
    write_csv(tmp_path / "workflow_compare.csv", rows, truncate_output_to_chars=40)
    write_examples_markdown(tmp_path / "examples.md", rows)
    assert (tmp_path / "rows.jsonl").exists()
    assert (tmp_path / "workflow_compare.csv").exists()
    assert (tmp_path / "examples.md").exists()
