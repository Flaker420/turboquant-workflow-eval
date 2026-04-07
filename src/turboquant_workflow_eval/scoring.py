"""Automated quality scoring for workflow study outputs.

This module used to expose a green/yellow/red ``compute_verdict`` system that
mixed latency-regression, fuzzy semantic similarity, math-rubric and code-
execution checks into one categorical label per row. That system measured
the wrong thing for this project: a turboquant policy whose generated tokens
are byte-identical to baseline could still be flagged "yellow" on latency
jitter, and the verdict gave no insight into either the actual divergence
from baseline or the actual KV-cache compression achieved.

The new world: every non-baseline row carries direct token-level divergence
metrics against its matching baseline row, plus a theoretical KV-cache-bytes
calculation derived from the policy settings. Reports become a Pareto-style
``(compression_ratio, exact_match_rate)`` view instead of verdict counts.

The legacy ``extract_numbers`` / ``check_reference_answer`` /
``compute_semantic_similarity`` helpers stay in the module — they have other
callers in the test suite and may yet be useful for diagnostic side-channels
— but the verdict pipeline (``compute_verdict``, ``_resolve_thresholds``,
``_DEFAULT_THRESHOLDS``) is gone.
"""

from __future__ import annotations

import re
from typing import Any, Sequence


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
# Token-level divergence vs baseline
# ---------------------------------------------------------------------------


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    """Plain O(n*m) Levenshtein distance over two integer sequences.

    Used here only over short token-id lists (n, m <= max_new_tokens, typically
    <= a few hundred), so the quadratic memory cost is irrelevant. Kept
    dependency-free so it stays inside the torch-free sandbox tests.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    # Two-row DP. previous[j] = distance(a[:i-1], b[:j]).
    previous = list(range(m + 1))
    current = [0] * (m + 1)
    for i in range(1, n + 1):
        current[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            current[j] = min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + cost,  # substitution
            )
        previous, current = current, previous
    return previous[m]


def _common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def compute_divergence(
    policy_ids: Sequence[int],
    baseline_ids: Sequence[int],
) -> dict[str, Any]:
    """Token-level comparison of one policy output against the baseline.

    Returns a dict with the metrics that the new reporting layer consumes.
    All values are JSON-serializable plain Python types.

    ``first_divergence_token`` semantics:
      * If the two sequences are identical, equals ``len(baseline_ids)`` and
        ``exact_match`` is ``True``.
      * If one is a strict prefix of the other (and the other has extra
        tokens), equals ``min(len(policy), len(baseline))`` — the index where
        the shorter sequence "ran out".
      * Otherwise equals the index of the first differing position.
    The baseline row itself never reaches this function; the study driver
    fills sentinel values for it directly.
    """
    p = list(policy_ids)
    b = list(baseline_ids)
    prefix = _common_prefix_length(p, b)
    exact = (len(p) == len(b)) and prefix == len(b)
    base_len = max(len(b), 1)
    return {
        "exact_match": exact,
        "first_divergence_token": prefix,
        "common_prefix_tokens": prefix,
        "common_prefix_frac": prefix / base_len,
        "token_edit_distance": levenshtein(p, b),
        "output_length_delta_tokens": len(p) - len(b),
    }


# ---------------------------------------------------------------------------
# KV-cache byte accounting
# ---------------------------------------------------------------------------

# Bytes per element in the various dtypes the cache can hold. We treat the
# baseline KV cache as bf16 (2 bytes/elem); turboquant compresses entries
# down to ``bit_width / 8`` bytes per element via the MSE codebook. These
# constants are intentionally simple — the goal is "directionally correct
# memory accounting that tracks the policy knobs", not a byte-exact
# reproduction of every allocator quirk in the runtime cache.
_BASELINE_BYTES_PER_ELEM = 2.0  # bf16


def _resolve_indices(
    indices: Sequence[int] | None,
    total: int,
) -> tuple[int, ...]:
    """Resolve a ``compressible_layers`` / ``compressible_heads`` selector.

    ``None`` (or an empty tuple coming back from JSON deserialization that
    originally meant "all") expands to ``range(total)`` so the bookkeeping
    matches the runtime cache, which treats ``None`` as "compress everything".
    """
    if indices is None:
        return tuple(range(total))
    return tuple(int(i) for i in indices)


def compute_kv_cache_bytes(
    *,
    prompt_tokens: int,
    output_tokens: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bit_width: int,
    residual_window: int,
    compressible_layers: Sequence[int] | None,
    compressible_heads: Sequence[int] | None,
    key_strategy: str,
    value_strategy: str,
) -> dict[str, Any]:
    """Theoretical KV-cache bytes for one (policy, sequence) pair.

    The baseline number is the bf16 cache size for the full sequence at the
    given layer/head/head_dim shape. The policy number applies the per-policy
    knobs:

      * Only layers in ``compressible_layers`` are eligible for compression
        (None ⇒ all layers).
      * Within an eligible layer, only KV heads in ``compressible_heads`` are
        eligible (None ⇒ all heads).
      * Within an eligible head, the most recent ``residual_window`` tokens
        stay at bf16 (the runtime "uncompressed window") and the remainder
        is stored at ``bit_width / 8`` bytes per element for both K and V.
      * ``key_strategy == "mse+qjl"`` adds a small QJL side-channel (one
        byte per compressed K token per head, sign bits packed to int8).
        This is a crude but directionally-correct overhead — refine when
        the QJL bookkeeping in core.py grows additional metadata.

    Returns kv_cache_bytes_baseline / _policy / compression_ratio /
    bytes_saved.

    The function is pure: no torch, no model, no I/O. It is the single source
    of truth for the "compression" axis in the new Pareto report.
    """
    seq_len = max(int(prompt_tokens) + int(output_tokens), 0)
    layers = _resolve_indices(compressible_layers, num_layers)
    heads = _resolve_indices(compressible_heads, num_kv_heads)
    layer_set = frozenset(layers)
    head_set = frozenset(heads)

    # Baseline: every layer * every kv head * seq_len * head_dim, K + V, bf16.
    elems_per_token = num_kv_heads * head_dim * 2  # K + V
    baseline = float(num_layers * seq_len * elems_per_token * _BASELINE_BYTES_PER_ELEM)

    compressed_bytes_per_elem = bit_width / 8.0
    qjl_overhead_bytes_per_token = 1.0 if key_strategy == "mse+qjl" else 0.0

    window = max(int(residual_window), 0)
    uncompressed_tokens = min(window, seq_len)
    compressed_tokens = max(seq_len - window, 0)

    # Per-head, per-layer cost when the head IS compressed.
    compressed_head_kv_bytes = (
        uncompressed_tokens * head_dim * 2 * _BASELINE_BYTES_PER_ELEM  # window K+V
        + compressed_tokens * head_dim * 2 * compressed_bytes_per_elem  # quantized K+V
        + compressed_tokens * qjl_overhead_bytes_per_token  # QJL side channel on K
    )
    # Per-head, per-layer cost when the head is NOT compressed (full bf16).
    full_head_kv_bytes = seq_len * head_dim * 2 * _BASELINE_BYTES_PER_ELEM

    policy = 0.0
    for layer_idx in range(num_layers):
        if layer_idx not in layer_set:
            policy += num_kv_heads * full_head_kv_bytes
            continue
        for head_idx in range(num_kv_heads):
            if head_idx in head_set:
                policy += compressed_head_kv_bytes
            else:
                policy += full_head_kv_bytes

    baseline_int = int(round(baseline))
    policy_int = int(round(policy))
    ratio = (baseline / policy) if policy > 0 else 1.0
    return {
        "kv_cache_bytes_baseline": baseline_int,
        "kv_cache_bytes_policy": policy_int,
        "kv_cache_compression_ratio": float(ratio),
        "kv_cache_bytes_saved": baseline_int - policy_int,
    }
