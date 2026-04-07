from __future__ import annotations

from turboquant_workflow_eval.scoring import (
    check_reference_answer,
    compute_divergence,
    compute_kv_cache_bytes,
    extract_final_number,
    extract_numbers,
    levenshtein,
)


# ---------------------------------------------------------------------------
# Legacy regex helpers (still exported for callers outside the verdict path)
# ---------------------------------------------------------------------------


class TestExtractNumbers:
    def test_basic(self) -> None:
        assert extract_numbers("The answer is 42.") == [42.0]

    def test_multiple(self) -> None:
        assert extract_numbers("From 12 to 21 over 3 years.") == [12.0, 21.0, 3.0]

    def test_decimal(self) -> None:
        assert extract_numbers("CAGR is 20.51%.") == [20.51]

    def test_negative(self) -> None:
        assert extract_numbers("Change: -5.2%") == [-5.2]

    def test_comma_thousands(self) -> None:
        assert extract_numbers("Total: 1,234 items") == [1234.0]

    def test_no_numbers(self) -> None:
        assert extract_numbers("no numbers here") == []


class TestExtractFinalNumber:
    def test_basic(self) -> None:
        assert extract_final_number("Step 1: 12. Step 2: 21. Result: 20.51") == 20.51

    def test_none(self) -> None:
        assert extract_final_number("no numbers") is None


class TestCheckReferenceAnswer:
    def test_match(self) -> None:
        assert check_reference_answer("The CAGR is approximately 20.51%.", "20.51") is True

    def test_no_match(self) -> None:
        assert check_reference_answer("The answer is 15.", "20.51") is False

    def test_no_reference(self) -> None:
        assert check_reference_answer("anything", None) is None

    def test_tolerance(self) -> None:
        assert check_reference_answer("Result: 20.5", "20.51") is True

    def test_exact_integer(self) -> None:
        assert check_reference_answer("12 * 170 = 2040 tokens total.", "2040") is True

    def test_weighted_average(self) -> None:
        assert check_reference_answer("The weighted average is 86.2.", "86.2") is True


# ---------------------------------------------------------------------------
# Levenshtein
# ---------------------------------------------------------------------------


class TestLevenshtein:
    def test_identical(self) -> None:
        assert levenshtein([1, 2, 3], [1, 2, 3]) == 0

    def test_empty_left(self) -> None:
        assert levenshtein([], [1, 2, 3]) == 3

    def test_empty_right(self) -> None:
        assert levenshtein([1, 2, 3], []) == 3

    def test_substitution(self) -> None:
        assert levenshtein([1, 2, 3], [1, 9, 3]) == 1

    def test_insertion(self) -> None:
        assert levenshtein([1, 2, 3], [1, 2, 3, 4]) == 1

    def test_deletion(self) -> None:
        assert levenshtein([1, 2, 3, 4], [1, 2, 3]) == 1

    def test_disjoint(self) -> None:
        # Different elements, same length: every position substitutes.
        assert levenshtein([1, 2, 3], [4, 5, 6]) == 3

    def test_known_value(self) -> None:
        # "kitten" -> "sitting" classic = 3.
        a = list(map(ord, "kitten"))
        b = list(map(ord, "sitting"))
        assert levenshtein(a, b) == 3


# ---------------------------------------------------------------------------
# compute_divergence
# ---------------------------------------------------------------------------


class TestComputeDivergence:
    def test_identical(self) -> None:
        out = compute_divergence([10, 20, 30], [10, 20, 30])
        assert out["exact_match"] is True
        assert out["first_divergence_token"] == 3
        assert out["common_prefix_tokens"] == 3
        assert out["common_prefix_frac"] == 1.0
        assert out["token_edit_distance"] == 0
        assert out["output_length_delta_tokens"] == 0

    def test_disjoint(self) -> None:
        out = compute_divergence([10, 20, 30], [40, 50, 60])
        assert out["exact_match"] is False
        assert out["first_divergence_token"] == 0
        assert out["common_prefix_tokens"] == 0
        assert out["common_prefix_frac"] == 0.0
        assert out["token_edit_distance"] == 3
        assert out["output_length_delta_tokens"] == 0

    def test_policy_strict_prefix_of_baseline(self) -> None:
        out = compute_divergence([1, 2, 3], [1, 2, 3, 4, 5])
        assert out["exact_match"] is False
        assert out["first_divergence_token"] == 3
        assert out["common_prefix_tokens"] == 3
        assert out["common_prefix_frac"] == 3 / 5
        assert out["token_edit_distance"] == 2
        assert out["output_length_delta_tokens"] == -2

    def test_baseline_strict_prefix_of_policy(self) -> None:
        out = compute_divergence([1, 2, 3, 9], [1, 2, 3])
        assert out["exact_match"] is False
        assert out["first_divergence_token"] == 3
        assert out["common_prefix_tokens"] == 3
        assert out["common_prefix_frac"] == 1.0
        assert out["token_edit_distance"] == 1
        assert out["output_length_delta_tokens"] == 1

    def test_middle_divergence(self) -> None:
        out = compute_divergence([1, 2, 7, 4, 5], [1, 2, 3, 4, 5])
        assert out["exact_match"] is False
        assert out["first_divergence_token"] == 2
        assert out["common_prefix_tokens"] == 2
        assert out["common_prefix_frac"] == 2 / 5
        assert out["token_edit_distance"] == 1
        assert out["output_length_delta_tokens"] == 0

    def test_empty_baseline_does_not_div_by_zero(self) -> None:
        out = compute_divergence([1, 2, 3], [])
        assert out["exact_match"] is False
        assert out["common_prefix_tokens"] == 0
        # max(len(baseline), 1) = 1 in the denominator.
        assert out["common_prefix_frac"] == 0.0
        assert out["token_edit_distance"] == 3
        assert out["output_length_delta_tokens"] == 3


# ---------------------------------------------------------------------------
# compute_kv_cache_bytes
# ---------------------------------------------------------------------------

# A small "fake model" topology used by every test below: 4 layers, 2 KV
# heads, head_dim 8. Sequence length 10 (prompt 6 + output 4). Pure-bf16
# baseline = 4 layers * 10 tokens * (2 kv_heads * 8 head_dim * 2 [K+V])
# elements * 2 bytes/elem = 2560 bytes.

_BASE_KW = dict(
    prompt_tokens=6,
    output_tokens=4,
    num_layers=4,
    num_kv_heads=2,
    head_dim=8,
    value_strategy="mse",
)


class TestComputeKvCacheBytes:
    def test_full_compression_4bit(self) -> None:
        out = compute_kv_cache_bytes(
            **_BASE_KW,
            bit_width=4,
            residual_window=0,
            compressible_layers=None,
            compressible_heads=None,
            key_strategy="mse",
        )
        # Every element shrinks 2 bytes -> 0.5 bytes => 4x.
        assert out["kv_cache_bytes_baseline"] == 2560
        assert out["kv_cache_bytes_policy"] == 640
        assert out["kv_cache_compression_ratio"] == 4.0
        assert out["kv_cache_bytes_saved"] == 1920

    def test_window_covers_full_sequence(self) -> None:
        out = compute_kv_cache_bytes(
            **_BASE_KW,
            bit_width=4,
            residual_window=10,  # equals seq_len
            compressible_layers=None,
            compressible_heads=None,
            key_strategy="mse",
        )
        assert out["kv_cache_bytes_policy"] == out["kv_cache_bytes_baseline"]
        assert out["kv_cache_compression_ratio"] == 1.0
        assert out["kv_cache_bytes_saved"] == 0

    def test_window_larger_than_sequence(self) -> None:
        out = compute_kv_cache_bytes(
            **_BASE_KW,
            bit_width=4,
            residual_window=99,
            compressible_layers=None,
            compressible_heads=None,
            key_strategy="mse",
        )
        assert out["kv_cache_compression_ratio"] == 1.0

    def test_only_one_layer_compressible(self) -> None:
        # Per-layer baseline = 2560 / 4 = 640 bytes. One layer at 4-bit
        # shrinks to 160; the other three stay at 640 each. Total 160 +
        # 3*640 = 2080. Ratio 2560 / 2080.
        out = compute_kv_cache_bytes(
            **_BASE_KW,
            bit_width=4,
            residual_window=0,
            compressible_layers=(0,),
            compressible_heads=None,
            key_strategy="mse",
        )
        assert out["kv_cache_bytes_baseline"] == 2560
        assert out["kv_cache_bytes_policy"] == 2080
        assert abs(out["kv_cache_compression_ratio"] - 2560 / 2080) < 1e-9

    def test_only_one_kv_head_compressible(self) -> None:
        # full_head_kv_bytes = seq_len * head_dim * 2 (K+V) * 2 bytes
        #                    = 10 * 8 * 2 * 2 = 320
        # compressed_head_kv_bytes (window=0, 4-bit) = 10 * 8 * 2 * 0.5 = 80
        # per layer = 80 (head 0) + 320 (head 1) = 400
        # 4 layers = 1600. Ratio 2560 / 1600 = 1.6.
        out = compute_kv_cache_bytes(
            **_BASE_KW,
            bit_width=4,
            residual_window=0,
            compressible_layers=None,
            compressible_heads=(0,),
            key_strategy="mse",
        )
        assert out["kv_cache_bytes_policy"] == 1600
        assert abs(out["kv_cache_compression_ratio"] - 1.6) < 1e-9
        assert 1.0 < out["kv_cache_compression_ratio"] < 2.0

    def test_qjl_overhead_makes_policy_larger_than_pure_mse(self) -> None:
        kw = dict(
            **_BASE_KW,
            bit_width=4,
            residual_window=0,
            compressible_layers=None,
            compressible_heads=None,
        )
        mse_only = compute_kv_cache_bytes(key_strategy="mse", **kw)
        with_qjl = compute_kv_cache_bytes(key_strategy="mse+qjl", **kw)
        assert with_qjl["kv_cache_bytes_policy"] > mse_only["kv_cache_bytes_policy"]
        assert with_qjl["kv_cache_compression_ratio"] < mse_only["kv_cache_compression_ratio"]

    def test_zero_seq_len_returns_unit_ratio(self) -> None:
        out = compute_kv_cache_bytes(
            prompt_tokens=0,
            output_tokens=0,
            num_layers=4,
            num_kv_heads=2,
            head_dim=8,
            bit_width=4,
            residual_window=0,
            compressible_layers=None,
            compressible_heads=None,
            key_strategy="mse",
            value_strategy="mse",
        )
        assert out["kv_cache_bytes_baseline"] == 0
        assert out["kv_cache_bytes_policy"] == 0
        # Helper falls back to ratio 1.0 when policy bytes are zero.
        assert out["kv_cache_compression_ratio"] == 1.0
