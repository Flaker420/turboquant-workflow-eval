"""Re-score existing study results with new thresholds — no GPU needed."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from .loader import load_study_module
from .reporting import write_csv, write_examples_markdown, write_run_summary
from .schema import ThresholdsConfig
from .study import score_results


def _load_run_summary(rows_jsonl_path: Path) -> dict | None:
    summary_path = rows_jsonl_path.parent / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _coerce_threshold_value(raw: str) -> Any:
    """Match the legacy ``apply_dot_overrides`` coercion semantics."""
    s = raw.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _resolve_thresholds(
    study_config: str | Path | None,
    overrides: list[str] | None,
) -> ThresholdsConfig:
    """Build a :class:`ThresholdsConfig` from an optional study module plus
    dot-notation CLI overrides.

    Override forms accepted (the prefix is optional)::

        thresholds.latency_red_pct=50
        latency_red_pct=50
    """
    if study_config:
        study = load_study_module(study_config)
        base = study.thresholds
    else:
        base = ThresholdsConfig()

    if not overrides:
        return base

    valid_fields = {f.name for f in dataclasses.fields(ThresholdsConfig)}
    valid_fields.discard("per_category")  # not exposed via flat overrides

    updates: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key.startswith("thresholds."):
            key = key[len("thresholds.") :]
        if key not in valid_fields:
            raise ValueError(
                f"Unknown threshold field {key!r}. Valid fields: {sorted(valid_fields)}"
            )
        updates[key] = _coerce_threshold_value(value)

    return dataclasses.replace(base, **updates)


def rescore(
    rows_jsonl_path: str | Path,
    thresholds: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    study_config: str | Path | None = None,
    overrides: list[str] | None = None,
    baseline_policy_name: str | None = None,
) -> list[dict]:
    """Load rows from JSONL, recompute verdicts with refreshed thresholds.

    Resolution order for thresholds (later wins):
        1. ``study_config`` YAML's ``thresholds`` block (if provided)
        2. explicit ``thresholds`` argument
        3. ``overrides`` (dot-notation, e.g. ``thresholds.latency_red_pct=50``
           or the bare ``latency_red_pct=50``)

    Baseline policy resolution:
        1. ``baseline_policy_name`` argument
        2. ``baseline_policy_name`` from a sibling ``run_summary.json``
        3. single-policy auto-detect (handled by ``score_results``)

    Always emits a refreshed ``run_summary.json`` next to the rescored rows.
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

    resolved_thresholds = _resolve_thresholds(study_config, overrides)
    if thresholds:
        # Merge any inline overrides on top of the resolved dataclass.
        valid_fields = {f.name for f in dataclasses.fields(ThresholdsConfig)} - {"per_category"}
        unknown = set(thresholds) - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown threshold field(s): {sorted(unknown)}. "
                f"Valid: {sorted(valid_fields)}"
            )
        resolved_thresholds = dataclasses.replace(resolved_thresholds, **thresholds)

    prior_summary = _load_run_summary(rows_jsonl_path)
    if baseline_policy_name is None and prior_summary:
        baseline_policy_name = prior_summary.get("baseline_policy_name")

    old_verdicts = {(r["policy_name"], r["prompt_id"]): r.get("verdict") for r in rows}

    score_results(
        rows,
        thresholds=resolved_thresholds,
        baseline_policy_name=baseline_policy_name,
    )

    changed = 0
    for row in rows:
        key = (row["policy_name"], row["prompt_id"])
        if old_verdicts.get(key) != row.get("verdict"):
            changed += 1
            print(f"  {key[0]}:{key[1]}: {old_verdicts.get(key)} -> {row['verdict']}")
    print(f"\nRe-scored {len(rows)} rows, {changed} verdict(s) changed.")

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

    verdict_counts = {"green": 0, "yellow": 0, "red": 0}
    for row in rows:
        v = row.get("verdict", "green")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    summary: dict[str, Any] = dict(prior_summary or {})
    summary.update(
        {
            "row_count": len(rows),
            "verdict_summary": verdict_counts,
            "baseline_policy_name": baseline_policy_name
            or summary.get("baseline_policy_name"),
            "rescored": True,
            "rescore_thresholds": dataclasses.asdict(resolved_thresholds),
            "rescore_verdicts_changed": changed,
            "output_dir": str(target_dir),
        }
    )
    write_run_summary(target_dir / "run_summary.json", summary)

    print(f"Verdicts: {verdict_counts}")
    print(f"Rescored outputs written to: {target_dir}")

    return rows
