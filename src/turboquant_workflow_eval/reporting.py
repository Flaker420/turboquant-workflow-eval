from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], truncate_output_to_chars: int = 180) -> None:
    fieldnames = [
        "policy_name",
        "comparison_label",
        "adapter_name",
        "prompt_id",
        "category",
        "title",
        "prompt_tokens",
        "output_tokens",
        "latency_s",
        "tokens_per_second",
        "peak_vram_gb",
        "watch_for",
        "output_excerpt",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_excerpt = row["output_text"][:truncate_output_to_chars].replace("\n", " ")
            writer.writerow(
                {
                    "policy_name": row["policy_name"],
                    "comparison_label": row["comparison_label"],
                    "adapter_name": row["adapter_name"],
                    "prompt_id": row["prompt_id"],
                    "category": row["category"],
                    "title": row["title"],
                    "prompt_tokens": row["prompt_tokens"],
                    "output_tokens": row["output_tokens"],
                    "latency_s": f"{row['latency_s']:.4f}",
                    "tokens_per_second": "" if row["tokens_per_second"] is None else f"{row['tokens_per_second']:.4f}",
                    "peak_vram_gb": "" if row["peak_vram_gb"] is None else f"{row['peak_vram_gb']:.4f}",
                    "watch_for": row["watch_for"],
                    "output_excerpt": output_excerpt,
                }
            )


def write_text_outputs(root: Path, rows: list[dict]) -> None:
    text_root = root / "text_outputs"
    text_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        filename = f"{slugify(row['policy_name'])}__{slugify(row['prompt_id'])}.md"
        path = text_root / filename
        path.write_text(
            "\n".join(
                [
                    f"# {row['policy_name']} — {row['prompt_id']}",
                    "",
                    f"**Category:** {row['category']}",
                    f"**Title:** {row['title']}",
                    f"**Watch for:** {row['watch_for']}",
                    "",
                    "## Prompt",
                    "",
                    row["prompt_text"],
                    "",
                    "## Output",
                    "",
                    row["output_text"] or "_<empty output>_",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def write_examples_markdown(path: Path, rows: list[dict]) -> None:
    by_prompt: dict[str, list[dict]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], []).append(row)

    lines = ["# Workflow comparison examples", ""]
    for prompt_id in sorted(by_prompt):
        prompt_rows = by_prompt[prompt_id]
        reference = prompt_rows[0]
        lines.extend(
            [
                f"## {prompt_id} — {reference['title']}",
                "",
                f"**Category:** {reference['category']}",
                f"**Watch for:** {reference['watch_for']}",
                "",
                "### Prompt",
                "",
                reference["prompt_text"],
                "",
            ]
        )
        for row in prompt_rows:
            tps = row["tokens_per_second"]
            vram = row["peak_vram_gb"]
            lines.extend(
                [
                    f"### Policy: {row['policy_name']}",
                    "",
                    f"- Adapter: `{row['adapter_name']}`",
                    f"- Prompt tokens: {row['prompt_tokens']}",
                    f"- Output tokens: {row['output_tokens']}",
                    f"- Latency (s): {row['latency_s']:.4f}",
                    f"- Tokens/sec: {'n/a' if tps is None else f'{tps:.4f}'}",
                    f"- Peak VRAM (GB): {'n/a' if vram is None else f'{vram:.4f}'}",
                    "",
                    "```text",
                    row["output_text"] or "",
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_run_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
