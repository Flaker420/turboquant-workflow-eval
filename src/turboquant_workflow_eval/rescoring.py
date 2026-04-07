"""Re-score existing study results — no GPU needed.

The legacy version of this module pushed configurable verdict thresholds
through ``score_results``. Verdicts are gone, and so is the threshold
plumbing. The remaining job of ``rescore`` is mechanical: load a
``rows.jsonl`` produced by a prior run, recompute the new divergence /
KV-cache-bytes annotations, and write a refreshed
``run_summary.json`` / ``workflow_compare.csv`` / ``examples.md`` next to
the rows.

KV-cache-bytes need model topology (``num_hidden_layers``,
``num_key_value_heads``, ``head_dim``); when the rescore is invoked without
a live model — which is the whole point of the cold-rescore mode — we
attempt to read those values from the prior ``run_summary.json``'s
``policies_used`` block (where they were stashed during the original run),
and if they're missing we fall through with ``model_info=None``. In that
fallback the divergence metrics are still populated; only the KV-cache
columns end up empty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .reporting import write_csv, write_examples_markdown, write_run_summary, summarize_divergence
from .study import score_results


def _load_run_summary(rows_jsonl_path: Path) -> dict | None:
    summary_path = rows_jsonl_path.parent / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _model_info_from_summary(prior_summary: dict | None) -> dict[str, int] | None:
    if not prior_summary:
        return None
    info = prior_summary.get("model_info")
    if not isinstance(info, dict):
        return None
    required = {"num_hidden_layers", "num_key_value_heads", "head_dim"}
    if not required.issubset(info):
        return None
    return {k: int(info[k]) for k in required}


def rescore(
    rows_jsonl_path: str | Path,
    output_dir: str | Path | None = None,
    baseline_policy_name: str | None = None,
    **_legacy_kwargs: Any,
) -> list[dict]:
    """Reload a ``rows.jsonl`` and recompute divergence / KV-cache annotations.

    Any legacy keyword arguments (``thresholds``, ``study_config``,
    ``overrides``) are accepted and silently dropped — they were the
    verdict-system knobs and have no analogue in the divergence-metrics
    world. Callers should remove them.
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

    # Hard requirement of the divergence pipeline: every non-error row must
    # carry the raw output token IDs. Old runs that predate the schema bump
    # cannot be rescored — fail with one actionable message.
    missing_ids = [
        row.get("prompt_id", "?")
        for row in rows
        if not row.get("error") and "output_token_ids" not in row
    ]
    if missing_ids:
        raise RuntimeError(
            "rescore: rows.jsonl is missing the 'output_token_ids' field on "
            f"{len(missing_ids)} row(s) (e.g. {missing_ids[:3]!r}). This file "
            "was produced before the divergence-metrics schema bump and "
            "cannot be rescored cold; rerun the study to regenerate it."
        )

    prior_summary = _load_run_summary(rows_jsonl_path)
    if baseline_policy_name is None and prior_summary:
        baseline_policy_name = prior_summary.get("baseline_policy_name")

    model_info = _model_info_from_summary(prior_summary)

    score_results(
        rows,
        baseline_policy_name=baseline_policy_name,
        model_info=model_info,
    )

    target_dir = Path(output_dir) if output_dir else rows_jsonl_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    if output_dir:
        with (target_dir / "rows.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with rows_jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(target_dir / "workflow_compare.csv", rows)
    write_examples_markdown(target_dir / "examples.md", rows)

    divergence_summary = summarize_divergence(rows, baseline_policy_name)

    summary: dict[str, Any] = dict(prior_summary or {})
    summary.update(
        {
            "row_count": len(rows),
            "divergence_summary": divergence_summary,
            "baseline_policy_name": baseline_policy_name
            or summary.get("baseline_policy_name"),
            "rescored": True,
            "output_dir": str(target_dir),
        }
    )
    # Drop the stale verdict_summary field if the prior summary still
    # carried one — it would otherwise survive the dict.update() above.
    summary.pop("verdict_summary", None)
    write_run_summary(target_dir / "run_summary.json", summary)

    print(f"Divergence summary: {divergence_summary}")
    print(f"Rescored outputs written to: {target_dir}")

    return rows
