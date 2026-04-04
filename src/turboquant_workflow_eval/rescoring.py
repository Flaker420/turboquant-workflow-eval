"""Re-score existing study results with new thresholds — no GPU needed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .reporting import write_csv, write_examples_markdown, write_run_summary
from .scoring import compute_verdict


def rescore(
    rows_jsonl_path: str | Path,
    thresholds: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Load rows from JSONL, recompute verdicts with *thresholds*, and
    optionally rewrite output artefacts.

    Returns the re-scored rows.  No model loading, no GPU — pure computation.
    """
    rows_jsonl_path = Path(rows_jsonl_path)
    rows: list[dict] = []
    with rows_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        raise RuntimeError(f"No rows found in {rows_jsonl_path}")

    # Rebuild baseline lookup
    baseline_by_prompt: dict[str, dict] = {}
    for row in rows:
        pid = row["prompt_id"]
        if pid not in baseline_by_prompt:
            baseline_by_prompt[pid] = row

    # Recompute verdicts
    old_verdicts = {(r["policy_name"], r["prompt_id"]): r.get("verdict") for r in rows}
    for row in rows:
        pid = row["prompt_id"]
        bl = baseline_by_prompt.get(pid)
        row["verdict"] = compute_verdict(row, bl if bl is not row else None, thresholds)

    # Report changes
    changed = 0
    for row in rows:
        key = (row["policy_name"], row["prompt_id"])
        if old_verdicts.get(key) != row.get("verdict"):
            changed += 1
            print(f"  {key[0]}:{key[1]}: {old_verdicts.get(key)} -> {row['verdict']}")

    print(f"\nRe-scored {len(rows)} rows, {changed} verdict(s) changed.")

    # Write outputs if requested
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Write rescored JSONL
        with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_csv(output_dir / "workflow_compare.csv", rows)
        write_examples_markdown(output_dir / "examples.md", rows)

        verdict_counts = {"green": 0, "yellow": 0, "red": 0}
        for row in rows:
            v = row.get("verdict", "green")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        print(f"Verdicts: {verdict_counts}")
        print(f"Rescored outputs written to: {output_dir}")
    else:
        # Overwrite original JSONL in-place
        with rows_jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return rows
