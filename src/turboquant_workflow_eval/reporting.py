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


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_BASE_FIELDNAMES = [
    "policy_name",
    "comparison_label",
    "adapter_name",
    "prompt_id",
    "category",
    "title",
    "prompt_tokens",
    "output_tokens",
    "output_length_delta_tokens",
    "exact_match",
    "first_divergence_token",
    "common_prefix_frac",
    "token_edit_distance",
    "kv_cache_bytes_baseline",
    "kv_cache_bytes_policy",
    "kv_cache_compression_ratio",
    "kv_cache_bytes_saved",
    "latency_s",
    "tokens_per_second",
    "peak_vram_gb",
]

_REP_FIELDNAMES = [
    "latency_mean", "latency_std",
    "tps_mean", "tps_std",
    "vram_mean", "vram_std",
    "repetitions",
]

_TAIL_FIELDNAMES = ["watch_for", "output_excerpt"]


def _fmt_float(value, fmt: str = ".4f") -> str:
    if value is None:
        return ""
    return format(value, fmt)


def _fmt_int(value) -> str:
    if value is None:
        return ""
    return str(int(value))


def _fmt_bool(value) -> str:
    if value is None:
        return ""
    return "yes" if bool(value) else "no"


def write_csv(path: Path, rows: list[dict], truncate_output_to_chars: int = 180) -> None:
    has_reps = any("latency_mean" in r for r in rows)
    fieldnames = list(_BASE_FIELDNAMES)
    if has_reps:
        fieldnames.extend(_REP_FIELDNAMES)
    fieldnames.extend(_TAIL_FIELDNAMES)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_text = row.get("output_text") or ""
            output_excerpt = output_text[:truncate_output_to_chars].replace("\n", " ")
            csv_row = {
                "policy_name": row.get("policy_name", ""),
                "comparison_label": row.get("comparison_label", ""),
                "adapter_name": row.get("adapter_name", ""),
                "prompt_id": row.get("prompt_id", ""),
                "category": row.get("category", ""),
                "title": row.get("title", ""),
                "prompt_tokens": _fmt_int(row.get("prompt_tokens")),
                "output_tokens": _fmt_int(row.get("output_tokens")),
                "output_length_delta_tokens": _fmt_int(row.get("output_length_delta_tokens")),
                "exact_match": _fmt_bool(row.get("exact_match")),
                "first_divergence_token": _fmt_int(row.get("first_divergence_token")),
                "common_prefix_frac": _fmt_float(row.get("common_prefix_frac")),
                "token_edit_distance": _fmt_int(row.get("token_edit_distance")),
                "kv_cache_bytes_baseline": _fmt_int(row.get("kv_cache_bytes_baseline")),
                "kv_cache_bytes_policy": _fmt_int(row.get("kv_cache_bytes_policy")),
                "kv_cache_compression_ratio": _fmt_float(row.get("kv_cache_compression_ratio")),
                "kv_cache_bytes_saved": _fmt_int(row.get("kv_cache_bytes_saved")),
                "latency_s": _fmt_float(row.get("latency_s")),
                "tokens_per_second": _fmt_float(row.get("tokens_per_second")),
                "peak_vram_gb": _fmt_float(row.get("peak_vram_gb")),
                "watch_for": row.get("watch_for", ""),
                "output_excerpt": output_excerpt,
            }
            if has_reps:
                csv_row.update({
                    "latency_mean": _fmt_float(row.get("latency_mean")),
                    "latency_std": _fmt_float(row.get("latency_std")),
                    "tps_mean": _fmt_float(row.get("tps_mean")),
                    "tps_std": _fmt_float(row.get("tps_std")),
                    "vram_mean": _fmt_float(row.get("vram_mean")),
                    "vram_std": _fmt_float(row.get("vram_std")),
                    "repetitions": _fmt_int(row.get("repetitions")),
                })
            writer.writerow(csv_row)


# ---------------------------------------------------------------------------
# Per-prompt text dumps
# ---------------------------------------------------------------------------

def _format_bytes_mb(n_bytes) -> str:
    if n_bytes is None:
        return "n/a"
    return f"{n_bytes / (1024 * 1024):.2f} MB"


def _divergence_oneline(row: dict) -> str:
    if row.get("error"):
        return f"Error: {row['error']}"
    if row.get("exact_match") is True and row.get("first_divergence_token", -1) == -1:
        # Baseline row.
        return "Baseline reference (no comparison)."
    if row.get("exact_match") is True:
        return "Exact match with baseline."
    fdt = row.get("first_divergence_token")
    out_tok = row.get("output_tokens")
    edit = row.get("token_edit_distance")
    if fdt is None or out_tok is None:
        return "Divergence: unknown."
    return (
        f"Diverges at token {fdt}/{out_tok}, "
        f"edit distance {edit}, "
        f"common prefix {row.get('common_prefix_frac', 0.0):.2f}."
    )


def _kv_oneline(row: dict) -> str:
    base = row.get("kv_cache_bytes_baseline")
    policy = row.get("kv_cache_bytes_policy")
    ratio = row.get("kv_cache_compression_ratio")
    saved = row.get("kv_cache_bytes_saved")
    if base is None or policy is None or ratio is None:
        return "KV cache: n/a"
    return (
        f"KV cache: {ratio:.2f}x smaller "
        f"({_format_bytes_mb(base)} -> {_format_bytes_mb(policy)}, "
        f"saved {_format_bytes_mb(saved)})"
    )


def write_text_outputs(root: Path, rows: list[dict]) -> None:
    text_root = root / "text_outputs"
    text_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        filename = f"{slugify(row['policy_name'])}__{slugify(row['prompt_id'])}.md"
        path = text_root / filename
        front = [
            f"# {row['policy_name']} — {row['prompt_id']}",
            "",
            f"**Category:** {row['category']}",
            f"**Title:** {row['title']}",
            f"**Watch for:** {row['watch_for']}",
            f"**Divergence:** {_divergence_oneline(row)}",
            f"**{_kv_oneline(row)}**",
            "",
            "## Prompt",
            "",
            row["prompt_text"],
            "",
            "## Output",
            "",
        ]
        if row.get("exact_match") is True and row.get("first_divergence_token", 0) != -1:
            front.append("Exact match with baseline.")
            front.append("")
        front.append(row.get("output_text") or "_<empty output>_")
        front.append("")
        path.write_text("\n".join(front), encoding="utf-8")


# ---------------------------------------------------------------------------
# Combined examples.md
# ---------------------------------------------------------------------------

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
            tps = row.get("tokens_per_second")
            vram = row.get("peak_vram_gb")
            lines.extend(
                [
                    f"### Policy: {row['policy_name']}",
                    "",
                    f"- Adapter: `{row['adapter_name']}`",
                    f"- Prompt tokens: {row.get('prompt_tokens', 0)}",
                    f"- Output tokens: {row.get('output_tokens', 0)}",
                    f"- Output length delta (tokens): {row.get('output_length_delta_tokens', 0)}",
                    f"- {_divergence_oneline(row)}",
                    f"- {_kv_oneline(row)}",
                    f"- Latency (s): {_fmt_float(row.get('latency_s'))}",
                    f"- Tokens/sec: {'n/a' if tps is None else f'{tps:.4f}'}",
                    f"- Peak VRAM (GB): {'n/a' if vram is None else f'{vram:.4f}'}",
                ]
            )
            if "latency_mean" in row:
                lines.append(
                    f"- Latency mean +/- std: {row['latency_mean']:.4f} +/- {row.get('latency_std', 0.0):.4f}"
                )
            lines.extend(
                [
                    "",
                    "```text",
                    row.get("output_text") or "",
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Pareto-style summary
# ---------------------------------------------------------------------------

def summarize_divergence(
    rows: list[dict],
    baseline_policy_name: str | None,
) -> dict[str, dict]:
    """Aggregate per-policy divergence + KV-cache stats from a row list.

    Baseline policy is excluded from the output (it is the reference).
    Error rows are excluded from the per-policy means but are surfaced via
    a top-level ``errors`` count on the policy entry.

    The intent of this block is to make Pareto-frontier plotting trivial:
    every entry exposes ``exact_match_rate`` and
    ``mean_kv_cache_compression_ratio`` as the two axes the next PR will
    plot. The other fields are diagnostic.
    """
    by_policy: dict[str, list[dict]] = {}
    for row in rows:
        name = row.get("policy_name")
        if name is None:
            continue
        if name == baseline_policy_name:
            continue
        by_policy.setdefault(name, []).append(row)

    summary: dict[str, dict] = {}
    for name, policy_rows in by_policy.items():
        scored = [r for r in policy_rows if not r.get("error")]
        errors = len(policy_rows) - len(scored)
        n = len(scored)
        if n == 0:
            summary[name] = {
                "prompts": 0,
                "errors": errors,
                "exact_match_count": 0,
                "exact_match_rate": 0.0,
                "mean_common_prefix_frac": 0.0,
                "mean_token_edit_distance": 0.0,
                "max_token_edit_distance": 0,
                "mean_kv_cache_compression_ratio": None,
                "mean_kv_cache_bytes_saved": None,
                "mean_peak_vram_gb": None,
            }
            continue

        exact_count = sum(1 for r in scored if r.get("exact_match"))
        prefix_fracs = [float(r.get("common_prefix_frac") or 0.0) for r in scored]
        edits = [int(r.get("token_edit_distance") or 0) for r in scored]
        ratios = [r.get("kv_cache_compression_ratio") for r in scored if r.get("kv_cache_compression_ratio") is not None]
        saved = [r.get("kv_cache_bytes_saved") for r in scored if r.get("kv_cache_bytes_saved") is not None]
        vrams = [r.get("peak_vram_gb") for r in scored if r.get("peak_vram_gb") is not None]

        summary[name] = {
            "prompts": n,
            "errors": errors,
            "exact_match_count": exact_count,
            "exact_match_rate": exact_count / n,
            "mean_common_prefix_frac": sum(prefix_fracs) / n,
            "mean_token_edit_distance": sum(edits) / n,
            "max_token_edit_distance": max(edits),
            "mean_kv_cache_compression_ratio": (sum(ratios) / len(ratios)) if ratios else None,
            "mean_kv_cache_bytes_saved": (sum(saved) / len(saved)) if saved else None,
            "mean_peak_vram_gb": (sum(vrams) / len(vrams)) if vrams else None,
        }
    return summary


def write_run_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Incremental writer
# ---------------------------------------------------------------------------

class IncrementalWriter:
    """Writes result rows to JSONL incrementally (one per prompt completion).

    Partial results survive crashes — rows are flushed to disk after each
    write.  Call :meth:`finalize` at the end to generate CSV, markdown, and
    summary outputs from the accumulated rows.

    The legacy ``finalize`` accumulated a green/yellow/red verdict count;
    that bookkeeping is gone, replaced by a post-hoc divergence summary
    built from the full row list (which is the only way to get a correct
    answer once per-row scoring depends on a join against the baseline).
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
        baseline_policy_name: str | None = None,
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

        divergence_summary = summarize_divergence(rows, baseline_policy_name)
        summary = {
            "study_name": study_cfg.get("name", ""),
            "model_name": model_cfg.get("model_name", ""),
            "policy_count": len(policies_used),
            "prompt_count": prompt_count,
            "row_count": len(rows),
            "repetitions": repetitions,
            "baseline_policy_name": baseline_policy_name,
            "divergence_summary": divergence_summary,
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
