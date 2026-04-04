from __future__ import annotations

from turboquant_workflow_eval.scoring import (
    check_reference_answer,
    compute_verdict,
    extract_final_number,
    extract_numbers,
)


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
        # 20.5 is within 5% of 20.51
        assert check_reference_answer("Result: 20.5", "20.51") is True

    def test_exact_integer(self) -> None:
        assert check_reference_answer("12 * 170 = 2040 tokens total.", "2040") is True

    def test_weighted_average(self) -> None:
        assert check_reference_answer("The weighted average is 86.2.", "86.2") is True


class TestComputeVerdict:
    def test_baseline_is_green(self) -> None:
        row = {"latency_s": 1.0, "output_tokens": 100}
        assert compute_verdict(row, None) == "green"

    def test_green_within_thresholds(self) -> None:
        baseline = {"latency_s": 1.0, "output_tokens": 100}
        row = {"latency_s": 1.05, "output_tokens": 102}
        assert compute_verdict(row, baseline) == "green"

    def test_yellow_latency(self) -> None:
        baseline = {"latency_s": 1.0, "output_tokens": 100}
        row = {"latency_s": 1.15, "output_tokens": 100}
        assert compute_verdict(row, baseline) == "yellow"

    def test_red_latency(self) -> None:
        baseline = {"latency_s": 1.0, "output_tokens": 100}
        row = {"latency_s": 1.30, "output_tokens": 100}
        assert compute_verdict(row, baseline) == "red"

    def test_red_math_failure(self) -> None:
        baseline = {"latency_s": 1.0, "output_tokens": 100}
        row = {"latency_s": 1.0, "output_tokens": 100, "math_correct": False}
        assert compute_verdict(row, baseline) == "red"

    def test_red_code_failure(self) -> None:
        baseline = {"latency_s": 1.0, "output_tokens": 100}
        row = {"latency_s": 1.0, "output_tokens": 100, "code_verdict": "fail"}
        assert compute_verdict(row, baseline) == "red"

    def test_yellow_similarity(self) -> None:
        baseline = {"latency_s": 1.0, "output_tokens": 100}
        row = {"latency_s": 1.0, "output_tokens": 100, "semantic_similarity": 0.88}
        assert compute_verdict(row, baseline) == "yellow"

    def test_red_output_length_delta(self) -> None:
        baseline = {"latency_s": 1.0, "output_tokens": 100}
        row = {"latency_s": 1.0, "output_tokens": 140}
        assert compute_verdict(row, baseline) == "red"
