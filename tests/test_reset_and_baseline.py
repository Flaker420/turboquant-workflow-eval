from __future__ import annotations

from types import SimpleNamespace

import pytest

from turboquant_workflow_eval.adapters.base import CompressionAdapter
from turboquant_workflow_eval.schema import (
    AdapterSpec,
    PolicyConfig,
    PolicySettings,
    RuntimeConfig,
)
from turboquant_workflow_eval.study import _run_single_prompt, score_results


def _baseline_policy() -> PolicyConfig:
    return PolicyConfig(
        name="baseline",
        adapter=AdapterSpec(import_path="m:C"),
        settings=PolicySettings(),
    )


_RUNTIME = RuntimeConfig(max_input_tokens=512, max_new_tokens=64)


class CountingAdapter(CompressionAdapter):
    name = "counting"

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset_generation_state(self) -> None:
        self.reset_calls += 1


def _fake_generate_one(model, tokenizer, prompt_text, runtime_cfg, turns=None):
    return {
        "rendered_prompt": prompt_text,
        "prompt_tokens": 5,
        "output_tokens": 10,
        "latency_s": 0.1,
        "tokens_per_second": 100.0,
        "peak_vram_gb": None,
        "output_text": "ok",
    }


def _make_prompt(pid="p1"):
    return SimpleNamespace(
        id=pid,
        category="reasoning",
        title="t",
        prompt="hello",
        watch_for="",
        reference_answer=None,
        test_cases=None,
        turns=None,
    )


# ---------- reset_generation_state ----------

class TestResetGenerationState:
    def test_noop_base_adapter(self) -> None:
        # Base class implementation must not raise.
        CompressionAdapter().reset_generation_state()

    def test_reset_count_with_repetitions(self, monkeypatch) -> None:
        import turboquant_workflow_eval.study as study_mod
        monkeypatch.setattr(study_mod, "generate_one", _fake_generate_one)

        adapter = CountingAdapter()
        _run_single_prompt(
            model=None,
            tokenizer=None,
            prompt=_make_prompt("p1"),
            policy=_baseline_policy(),
            adapter=adapter,
            runtime=_RUNTIME,
            repetitions=3,
        )
        assert adapter.reset_calls == 3

    def test_reset_count_two_prompts(self, monkeypatch) -> None:
        import turboquant_workflow_eval.study as study_mod
        monkeypatch.setattr(study_mod, "generate_one", _fake_generate_one)

        adapter = CountingAdapter()
        for pid in ("p1", "p2"):
            _run_single_prompt(
                model=None,
                tokenizer=None,
                prompt=_make_prompt(pid),
                policy=_baseline_policy(),
                adapter=adapter,
                runtime=_RUNTIME,
                repetitions=2,
            )
        assert adapter.reset_calls == 4


# ---------- score_results baseline selection ----------

def _row(policy: str, pid: str, tokens: int = 10, ids: list[int] | None = None) -> dict:
    if ids is None:
        ids = list(range(tokens))
    return {
        "policy_name": policy,
        "prompt_id": pid,
        "prompt_tokens": 5,
        "output_tokens": tokens,
        "output_text": "x",
        "output_token_ids": ids,
        # Policy settings the new score_results path expects on every row.
        "bit_width": 16,
        "residual_window": 0,
        "key_strategy": "mse",
        "value_strategy": "mse",
        "compressible_layers": None,
        "compressible_heads": None,
    }


class TestScoreResultsBaseline:
    def test_single_policy_defaults(self) -> None:
        rows = [_row("only", "p1"), _row("only", "p2")]
        score_results(rows)
        for row in rows:
            # Single-policy run: every row IS its own baseline, sentinel
            # divergence values land on the row.
            assert row["exact_match"] is True
            assert row["first_divergence_token"] == -1
            assert row["token_edit_distance"] == 0

    def test_explicit_baseline_used(self) -> None:
        rows = [
            _row("compressed", "p1", tokens=20, ids=list(range(10)) + list(range(100, 110))),
            _row("baseline", "p1", tokens=10),
        ]
        score_results(rows, baseline_policy_name="baseline")
        compressed = next(r for r in rows if r["policy_name"] == "compressed")
        # Same first 10 ids, then 10 differing ids.
        assert compressed["exact_match"] is False
        assert compressed["first_divergence_token"] == 10
        assert compressed["common_prefix_tokens"] == 10
        assert compressed["output_length_delta_tokens"] == 10
        baseline = next(r for r in rows if r["policy_name"] == "baseline")
        assert baseline["exact_match"] is True

    def test_stable_under_row_order(self) -> None:
        rows_a = [
            _row("baseline", "p1", tokens=10),
            _row("compressed", "p1", tokens=15, ids=list(range(8)) + list(range(200, 207))),
        ]
        rows_b = list(reversed([dict(r) for r in rows_a]))
        score_results(rows_a, baseline_policy_name="baseline")
        score_results(rows_b, baseline_policy_name="baseline")

        def by_policy(rs):
            return {r["policy_name"]: r for r in rs}

        a = by_policy(rows_a)
        b = by_policy(rows_b)
        for name in ("baseline", "compressed"):
            assert a[name]["first_divergence_token"] == b[name]["first_divergence_token"]
            assert a[name]["token_edit_distance"] == b[name]["token_edit_distance"]
            assert a[name]["exact_match"] == b[name]["exact_match"]

    def test_missing_baseline_name_multi_policy(self) -> None:
        rows = [_row("a", "p1"), _row("b", "p1")]
        with pytest.raises(ValueError, match="baseline_policy_name"):
            score_results(rows)

    def test_named_baseline_not_in_rows(self) -> None:
        rows = [_row("a", "p1"), _row("b", "p1")]
        with pytest.raises(ValueError, match="not found"):
            score_results(rows, baseline_policy_name="nonexistent")

    def test_baseline_missing_for_some_prompt(self) -> None:
        rows = [
            _row("baseline", "p1"),
            _row("compressed", "p1"),
            _row("compressed", "p2"),
        ]
        with pytest.raises(ValueError, match="no baseline row"):
            score_results(rows, baseline_policy_name="baseline")
