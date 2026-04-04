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
        "output_length_delta_pct",
        "latency_s",
        "tokens_per_second",
        "peak_vram_gb",
    ]

    # Add repetition stat columns if present
    has_reps = any("latency_mean" in r for r in rows)
    if has_reps:
        fieldnames.extend(["latency_mean", "latency_std", "tps_mean", "tps_std", "vram_mean", "vram_std", "repetitions"])

    fieldnames.extend([
        "math_correct",
        "code_verdict",
        "semantic_similarity",
        "verdict",
        "watch_for",
        "output_excerpt",
    ])

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_excerpt = row["output_text"][:truncate_output_to_chars].replace("\n", " ")
            csv_row = {
                "policy_name": row["policy_name"],
                "comparison_label": row["comparison_label"],
                "adapter_name": row["adapter_name"],
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "title": row["title"],
                "prompt_tokens": row["prompt_tokens"],
                "output_tokens": row["output_tokens"],
                "output_length_delta_pct": f"{row.get('output_length_delta_pct', 0.0):.1f}",
                "latency_s": f"{row['latency_s']:.4f}",
                "tokens_per_second": "" if row["tokens_per_second"] is None else f"{row['tokens_per_second']:.4f}",
                "peak_vram_gb": "" if row["peak_vram_gb"] is None else f"{row['peak_vram_gb']:.4f}",
                "math_correct": _fmt_optional(row.get("math_correct")),
                "code_verdict": row.get("code_verdict", ""),
                "semantic_similarity": "" if row.get("semantic_similarity") is None else f"{row['semantic_similarity']:.4f}",
                "verdict": row.get("verdict", ""),
                "watch_for": row["watch_for"],
                "output_excerpt": output_excerpt,
            }
            if has_reps:
                csv_row.update({
                    "latency_mean": f"{row.get('latency_mean', 0.0):.4f}" if "latency_mean" in row else "",
                    "latency_std": f"{row.get('latency_std', 0.0):.4f}" if "latency_std" in row else "",
                    "tps_mean": f"{row.get('tps_mean', 0.0):.4f}" if "tps_mean" in row else "",
                    "tps_std": f"{row.get('tps_std', 0.0):.4f}" if "tps_std" in row else "",
                    "vram_mean": f"{row.get('vram_mean', 0.0):.4f}" if "vram_mean" in row else "",
                    "vram_std": f"{row.get('vram_std', 0.0):.4f}" if "vram_std" in row else "",
                    "repetitions": row.get("repetitions", ""),
                })
            writer.writerow(csv_row)


def _fmt_optional(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "pass" if value else "fail"
    return str(value)


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
                    f"**Verdict:** {row.get('verdict', 'n/a')}",
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
            verdict = row.get("verdict", "")
            verdict_badge = f" [{verdict.upper()}]" if verdict else ""
            lines.extend(
                [
                    f"### Policy: {row['policy_name']}{verdict_badge}",
                    "",
                    f"- Adapter: `{row['adapter_name']}`",
                    f"- Prompt tokens: {row['prompt_tokens']}",
                    f"- Output tokens: {row['output_tokens']}",
                    f"- Output length delta: {row.get('output_length_delta_pct', 0.0):.1f}%",
                    f"- Latency (s): {row['latency_s']:.4f}",
                    f"- Tokens/sec: {'n/a' if tps is None else f'{tps:.4f}'}",
                    f"- Peak VRAM (GB): {'n/a' if vram is None else f'{vram:.4f}'}",
                ]
            )
            # Repetition stats
            if "latency_mean" in row:
                lines.append(f"- Latency mean +/- std: {row['latency_mean']:.4f} +/- {row['latency_std']:.4f}")
            # Scoring
            mc = row.get("math_correct")
            if mc is not None:
                lines.append(f"- Math correct: {'PASS' if mc else 'FAIL'}")
            cv = row.get("code_verdict")
            if cv:
                code_total = row.get('code_passed', 0) + row.get('code_failed', 0) + row.get('code_errors', 0)
                lines.append(f"- Code verdict: {cv.upper()} ({row.get('code_passed', 0)}/{code_total})")
            sim = row.get("semantic_similarity")
            if sim is not None:
                lines.append(f"- Semantic similarity: {sim:.4f}")

            lines.extend(
                [
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


class IncrementalWriter:
    """Writes result rows to JSONL incrementally (one per prompt completion).

    Partial results survive crashes — rows are flushed to disk after each
    write.  Call :meth:`finalize` at the end to generate CSV, markdown, and
    summary outputs from the accumulated rows.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = output_dir / "rows.jsonl"
        self._file = self._jsonl_path.open("w", encoding="utf-8")
        self._rows: list[dict] = []

    def write_row(self, row: dict) -> None:
        """Append one result row immediately and flush to disk."""
        self._rows.append(row)
        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()

    def finalize(
        self,
        study_cfg: dict,
        model_cfg: dict,
        policies_used: list[dict],
        prompt_count: int,
        repetitions: int,
        truncate_csv_output_to_chars: int = 180,
        write_individual_text_files: bool = True,
    ) -> dict:
        """Write CSV, markdown, and summary.  Returns the summary dict."""
        self._file.close()

        rows = self._rows
        write_csv(
            self.output_dir / "workflow_compare.csv",
            rows,
            truncate_output_to_chars=truncate_csv_output_to_chars,
        )
        if write_individual_text_files:
            write_text_outputs(self.output_dir, rows)
        write_examples_markdown(self.output_dir / "examples.md", rows)

        verdict_counts = {"green": 0, "yellow": 0, "red": 0}
        for row in rows:
            v = row.get("verdict", "green")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

        summary = {
            "study_name": study_cfg.get("name", ""),
            "model_name": model_cfg.get("model_name", ""),
            "policy_count": len(policies_used),
            "prompt_count": prompt_count,
            "row_count": len(rows),
            "repetitions": repetitions,
            "verdict_summary": verdict_counts,
            "policies_used": policies_used,
            "output_dir": str(self.output_dir),
        }
        write_run_summary(self.output_dir / "run_summary.json", summary)
        return summary

    @property
    def rows(self) -> list[dict]:
        return self._rows

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
