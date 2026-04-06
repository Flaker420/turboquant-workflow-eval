"""Automated quality scoring for workflow study outputs."""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Math reference-answer checking
# ---------------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(
    r"""
    (?<!\w)           # not preceded by a word char
    -?                # optional negative sign
    \d[\d,]*          # digits (with optional thousands commas)
    (?:\.\d+)?        # optional decimal part
    (?:%)?            # optional trailing percent sign
    (?!\w)            # not followed by a word char
    """,
    re.VERBOSE,
)


def extract_numbers(text: str) -> list[float]:
    """Extract all standalone numbers from *text*, in order of appearance."""
    results: list[float] = []
    for match in _NUMBER_PATTERN.finditer(text):
        raw = match.group().rstrip("%").replace(",", "")
        try:
            results.append(float(raw))
        except ValueError:
            continue
    return results


def extract_final_number(text: str) -> float | None:
    """Return the last standalone number found in *text*, or ``None``."""
    nums = extract_numbers(text)
    return nums[-1] if nums else None


def check_reference_answer(
    output_text: str,
    reference: str | None,
    tolerance: float = 0.05,
) -> bool | None:
    """Check whether *output_text* contains the expected numeric answer.

    Returns ``True`` if a number within *tolerance* (relative) of *reference*
    appears anywhere in the output, ``False`` if not, or ``None`` when no
    reference answer is defined.
    """
    if reference is None:
        return None
    try:
        ref_value = float(reference.rstrip("%").replace(",", ""))
    except (ValueError, AttributeError):
        return None

    # Strategy: accept if *any* number in the output matches the reference.
    # This tolerates intermediate-step numbers (e.g. "12, 21, 3 → 20.51")
    # at the cost of possible false positives on number-heavy outputs.
    for num in extract_numbers(output_text):
        if ref_value == 0:
            if num == 0:
                return True
        elif abs(num - ref_value) / abs(ref_value) <= tolerance:
            return True
    return False


# ---------------------------------------------------------------------------
# Semantic similarity (optional dependency)
# ---------------------------------------------------------------------------

_SIMILARITY_MODEL = None
_SIMILARITY_AVAILABLE: bool | None = None


def _load_similarity_model() -> Any | None:
    global _SIMILARITY_MODEL, _SIMILARITY_AVAILABLE
    if _SIMILARITY_AVAILABLE is False:
        return None
    if _SIMILARITY_MODEL is not None:
        return _SIMILARITY_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        _SIMILARITY_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _SIMILARITY_AVAILABLE = True
        return _SIMILARITY_MODEL
    except ImportError:
        _SIMILARITY_AVAILABLE = False
        return None


def compute_semantic_similarity(text_a: str, text_b: str) -> float | None:
    """Cosine similarity between two texts using sentence-transformers.

    Returns ``None`` if the library is not installed.
    """
    model = _load_similarity_model()
    if model is None:
        return None
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    from sentence_transformers.util import cos_sim  # type: ignore[import-untyped]

    return float(cos_sim(embeddings[0], embeddings[1]).item())


# ---------------------------------------------------------------------------
# Green / Yellow / Red verdict system
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "latency_yellow_pct": 10.0,
    "latency_red_pct": 25.0,
    "similarity_yellow": 0.92,
    "similarity_red": 0.80,
    "output_length_yellow_pct": 15.0,
    "output_length_red_pct": 30.0,
}


def _thresholds_to_dict(thresholds: Any) -> dict[str, Any]:
    """Coerce a ``ThresholdsConfig`` (or already-flat dict) to a flat dict.

    None-valued fields are dropped so they fall back to ``_DEFAULT_THRESHOLDS``.
    The ``per_category`` mapping is unfolded into top-level keys keyed by
    category name (matching the legacy nested format that ``_resolve_thresholds``
    already understands).
    """
    if thresholds is None:
        return {}
    if isinstance(thresholds, dict):
        return thresholds
    # Treat as ThresholdsConfig dataclass.
    out: dict[str, Any] = {}
    for f in ("latency_yellow_pct", "latency_red_pct", "similarity_yellow",
              "similarity_red", "output_length_yellow_pct", "output_length_red_pct"):
        v = getattr(thresholds, f, None)
        if v is not None:
            out[f] = v
    per_category = getattr(thresholds, "per_category", None) or {}
    if per_category:
        out["default"] = dict(out)  # snapshot of the flat fields
        for cat, sub in per_category.items():
            out[cat] = _thresholds_to_dict(sub)
    return out


def _resolve_thresholds(thresholds: Any, category: str | None = None) -> dict[str, float]:
    """Resolve thresholds with per-category override support.

    Accepts a ``ThresholdsConfig``, a flat dict, or a nested dict with
    ``"default"`` and category-specific keys::

        thresholds:
          default:
            latency_red_pct: 25.0
          math:
            latency_red_pct: 50.0

    Falls back to ``_DEFAULT_THRESHOLDS`` for any missing key.
    """
    thresholds = _thresholds_to_dict(thresholds)
    if not thresholds:
        return dict(_DEFAULT_THRESHOLDS)

    # Detect nested vs flat format
    has_nested = "default" in thresholds and isinstance(thresholds["default"], dict)
    if has_nested:
        base = {**_DEFAULT_THRESHOLDS, **thresholds.get("default", {})}
        if category and category in thresholds and isinstance(thresholds[category], dict):
            base = {**base, **thresholds[category]}
        return base

    # Flat format (backward compatible)
    return {**_DEFAULT_THRESHOLDS, **thresholds}


def compute_verdict(
    row: dict[str, Any],
    baseline_row: dict[str, Any] | None,
    thresholds: Any = None,
) -> str:
    """Return ``'green'``, ``'yellow'``, or ``'red'`` for a single result row.

    *baseline_row* is the corresponding row from the baseline policy for the
    same prompt. If ``None`` (i.e. this row **is** the baseline), returns
    ``'green'``.

    *thresholds* can be a flat dict of threshold values, or a nested dict
    with ``"default"`` and per-category overrides (e.g. ``"math"``,
    ``"coding"``).
    """
    if baseline_row is None:
        return "green"

    category = row.get("category")
    t = _resolve_thresholds(thresholds, category)
    worst = "green"

    def _escalate(level: str) -> str:
        order = {"green": 0, "yellow": 1, "red": 2}
        return level if order.get(level, 0) > order.get(worst, 0) else worst

    # --- Latency regression ---
    bl_lat = baseline_row.get("latency_s") or baseline_row.get("latency_mean")
    cur_lat = row.get("latency_s") or row.get("latency_mean")
    if bl_lat and cur_lat and bl_lat > 0:
        pct = ((cur_lat - bl_lat) / bl_lat) * 100
        if pct > t["latency_red_pct"]:
            worst = _escalate("red")
        elif pct > t["latency_yellow_pct"]:
            worst = _escalate("yellow")

    # --- Output-length delta ---
    bl_tokens = baseline_row.get("output_tokens", 0)
    cur_tokens = row.get("output_tokens", 0)
    if bl_tokens and bl_tokens > 0:
        length_pct = abs((cur_tokens - bl_tokens) / bl_tokens) * 100
        if length_pct > t["output_length_red_pct"]:
            worst = _escalate("red")
        elif length_pct > t["output_length_yellow_pct"]:
            worst = _escalate("yellow")

    # --- Semantic similarity ---
    sim = row.get("semantic_similarity")
    if sim is not None:
        if sim < t["similarity_red"]:
            worst = _escalate("red")
        elif sim < t["similarity_yellow"]:
            worst = _escalate("yellow")

    # --- Math correctness ---
    math_correct = row.get("math_correct")
    if math_correct is False:
        worst = _escalate("red")

    # --- Code execution ---
    code_verdict = row.get("code_verdict")
    if code_verdict == "fail":
        worst = _escalate("red")
    elif code_verdict == "error":
        worst = _escalate("yellow")

    return worst
