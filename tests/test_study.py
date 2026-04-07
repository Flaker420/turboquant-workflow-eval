from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from turboquant_workflow_eval.schema import (
    AdapterSpec,
    ModelConfig,
    PolicyConfig,
    PolicySettings,
    RuntimeConfig,
    StudyConfig,
)
from turboquant_workflow_eval.study import _aggregate_stats, run_workflow_study


class _FakeModelConfig:
    """Stand-in for a HF transformers config with the four fields the
    study driver reads to populate ``model_info``."""

    num_hidden_layers = 4
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 8
    hidden_size = 32


def _make_mock_model() -> MagicMock:
    """A MagicMock model whose ``.config`` exposes real ints so that the
    KV-cache-bytes accounting in score_results doesn't trip on
    ``int(MagicMock())``."""
    model = MagicMock()
    model.config = _FakeModelConfig()
    return model


class TestAggregateStats:
    def test_empty(self) -> None:
        result = _aggregate_stats([])
        assert result == {"mean": 0.0, "std": 0.0}

    def test_single_value(self) -> None:
        result = _aggregate_stats([5.0])
        assert result["mean"] == pytest.approx(5.0)
        assert result["std"] == pytest.approx(0.0)

    def test_known_values(self) -> None:
        result = _aggregate_stats([2.0, 4.0, 6.0])
        assert result["mean"] == pytest.approx(4.0)
        # sample std with n-1: sqrt(((2-4)^2 + (0)^2 + (2)^2) / 2) = sqrt(4) = 2
        assert result["std"] == pytest.approx(2.0)

    def test_identical_values(self) -> None:
        result = _aggregate_stats([3.0, 3.0, 3.0])
        assert result["mean"] == pytest.approx(3.0)
        assert result["std"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration tests for run_workflow_study with mocked model loading
# ---------------------------------------------------------------------------


def _make_baseline_policy() -> PolicyConfig:
    return PolicyConfig(
        name="baseline",
        adapter=AdapterSpec(
            import_path="turboquant_workflow_eval.adapters.none:NoCompressionAdapter"
        ),
        settings=PolicySettings(),
    )


def _make_compressed_policy() -> PolicyConfig:
    return PolicyConfig(
        name="compressed",
        adapter=AdapterSpec(
            import_path="turboquant_workflow_eval.adapters.none:NoCompressionAdapter"
        ),
        settings=PolicySettings(),
    )


def _make_study(tmp_path: Path, policies: tuple[PolicyConfig, ...]) -> StudyConfig:
    """Build a minimal but valid StudyConfig with one prompt fixture."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    prompt_pack_path = prompts_dir / "test_prompts.yaml"
    prompt_pack_path.write_text(
        "prompts:\n"
        "  - id: test_01\n"
        "    category: reasoning\n"
        "    title: test\n"
        "    prompt: Say hello.\n"
        "    watch_for: greeting\n"
    )

    return StudyConfig(
        name="test_study",
        model=ModelConfig(
            model_name="test/tiny",
            dtype="bfloat16",
            device_map="cpu",
            trust_remote_code=False,
        ),
        prompt_pack=(prompt_pack_path,),
        policies=policies,
        baseline_policy_name="baseline",
        runtime=RuntimeConfig(max_input_tokens=512, max_new_tokens=64),
    )


def _mock_generate_one(model, tokenizer, prompt_text, runtime_cfg, turns=None):
    return {
        "rendered_prompt": prompt_text,
        "prompt_tokens": 10,
        "output_tokens": 20,
        "latency_s": 0.5,
        "tokens_per_second": 40.0,
        "peak_vram_gb": None,
        "output_text": "Hello! This is a test response.",
        # New required field for the divergence-metrics scoring path.
        "output_token_ids": list(range(20)),
    }


class TestRunWorkflowStudy:
    @patch("turboquant_workflow_eval.study.load_model_and_tokenizer")
    @patch("turboquant_workflow_eval.study.generate_one")
    def test_full_study_run(self, mock_gen, mock_load, tmp_path: Path) -> None:
        mock_load.return_value = (_make_mock_model(), MagicMock(), "MockLoader")
        mock_gen.side_effect = _mock_generate_one

        study = _make_study(tmp_path, (_make_baseline_policy(),))
        output_dir = tmp_path / "outputs"

        summary = run_workflow_study(study, output_dir)

        assert summary["study_name"] == "test_study"
        assert summary["policy_count"] == 1
        assert summary["prompt_count"] == 1
        assert summary["row_count"] == 1
        # Single-policy run defaults baseline to that policy for provenance.
        assert summary["baseline_policy_name"] == "baseline"
        assert (output_dir / "rows.jsonl").exists()
        assert (output_dir / "workflow_compare.csv").exists()
        assert (output_dir / "run_summary.json").exists()
        assert (output_dir / "examples.md").exists()
        on_disk = json.loads((output_dir / "run_summary.json").read_text())
        assert on_disk["baseline_policy_name"] == "baseline"

    @patch("turboquant_workflow_eval.study.load_model_and_tokenizer")
    @patch("turboquant_workflow_eval.study.generate_one")
    def test_baseline_policy_name_in_summary_multi_policy(
        self, mock_gen, mock_load, tmp_path: Path
    ) -> None:
        mock_load.return_value = (_make_mock_model(), MagicMock(), "MockLoader")
        mock_gen.side_effect = _mock_generate_one

        study = _make_study(
            tmp_path, (_make_baseline_policy(), _make_compressed_policy())
        )
        output_dir = tmp_path / "outputs"
        summary = run_workflow_study(study, output_dir)

        assert summary["baseline_policy_name"] == "baseline"
        on_disk = json.loads((output_dir / "run_summary.json").read_text())
        assert on_disk["baseline_policy_name"] == "baseline"

    @patch("turboquant_workflow_eval.study.load_model_and_tokenizer")
    @patch("turboquant_workflow_eval.study.generate_one")
    def test_error_recovery(self, mock_gen, mock_load, tmp_path: Path) -> None:
        """A failing prompt does not crash the whole study."""
        mock_load.return_value = (_make_mock_model(), MagicMock(), "MockLoader")

        call_count = 0

        def gen_with_error(model, tokenizer, prompt_text, runtime_cfg, turns=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_generate_one(model, tokenizer, prompt_text, runtime_cfg)
            raise RuntimeError("GPU OOM")

        mock_gen.side_effect = gen_with_error

        study = _make_study(tmp_path, (_make_baseline_policy(),))
        output_dir = tmp_path / "outputs"
        summary = run_workflow_study(study, output_dir)
        assert summary["row_count"] == 1
        row = json.loads((output_dir / "rows.jsonl").read_text().strip())
        assert "error" in row
        assert "GPU OOM" in row["error"]

    @patch("turboquant_workflow_eval.study.load_model_and_tokenizer")
    @patch("turboquant_workflow_eval.study.generate_one")
    def test_divergence_summary(self, mock_gen, mock_load, tmp_path: Path) -> None:
        mock_load.return_value = (_make_mock_model(), MagicMock(), "MockLoader")
        mock_gen.side_effect = _mock_generate_one

        study = _make_study(tmp_path, (_make_baseline_policy(),))
        output_dir = tmp_path / "outputs"
        summary = run_workflow_study(study, output_dir)
        assert "divergence_summary" in summary
        assert "verdict_summary" not in summary
        # Single-policy run: the only policy is its own baseline, so the
        # divergence_summary excludes it and is empty.
        assert summary["divergence_summary"] == {}
        # model_info is captured at first model load and persisted.
        assert summary["model_info"] is not None
        assert summary["model_info"]["num_hidden_layers"] == 4
