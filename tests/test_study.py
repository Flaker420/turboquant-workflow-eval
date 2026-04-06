from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from turboquant_workflow_eval.study import _aggregate_stats, _load_policy_configs, run_workflow_study


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


class TestLoadPolicyConfigs:
    def test_from_arg_string(self, tmp_path: Path) -> None:
        paths = _load_policy_configs(tmp_path / "study.yaml", {}, "a.yaml,b.yaml")
        assert len(paths) == 2
        assert paths[0] == Path("a.yaml")
        assert paths[1] == Path("b.yaml")

    def test_from_study_config(self, tmp_path: Path) -> None:
        study_cfg = {"policy_configs": ["../policies/baseline.yaml"]}
        paths = _load_policy_configs(tmp_path / "studies" / "default.yaml", study_cfg, None)
        assert len(paths) == 1

    def test_empty_arg(self, tmp_path: Path) -> None:
        paths = _load_policy_configs(tmp_path / "study.yaml", {}, None)
        assert paths == []


class TestRunWorkflowStudy:
    """Integration-style tests with mocked model loading and generation."""

    def _create_study_files(self, tmp_path: Path) -> Path:
        """Create a minimal but valid study config tree."""
        model_dir = tmp_path / "configs" / "model"
        policy_dir = tmp_path / "configs" / "policies"
        study_dir = tmp_path / "configs" / "studies"
        prompts_dir = tmp_path / "prompts"
        for d in (model_dir, policy_dir, study_dir, prompts_dir):
            d.mkdir(parents=True)

        model_cfg = {
            "model_name": "test/tiny",
            "dtype": "bfloat16",
            "device_map": "cpu",
            "trust_remote_code": False,
        }
        (model_dir / "tiny.yaml").write_text(yaml.dump(model_cfg))

        policy_cfg = {
            "name": "baseline",
            "enabled": True,
            "comparison_label": "baseline",
            "adapter": {
                "import_path": "turboquant_workflow_eval.adapters.none:NoCompressionAdapter",
            },
            "settings": {},
        }
        (policy_dir / "baseline.yaml").write_text(yaml.dump(policy_cfg))

        prompts = {
            "prompts": [
                {
                    "id": "test_01",
                    "category": "reasoning",
                    "title": "test",
                    "prompt": "Say hello.",
                    "watch_for": "greeting",
                },
            ]
        }
        (prompts_dir / "test_prompts.yaml").write_text(yaml.dump(prompts))

        study_cfg = {
            "name": "test_study",
            "model_config": "../model/tiny.yaml",
            "prompt_pack": "../../prompts/test_prompts.yaml",
            "policy_configs": ["../policies/baseline.yaml"],
            "runtime": {
                "max_input_tokens": 512,
                "max_new_tokens": 64,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "use_cache": True,
                "repetitions": 1,
            },
            "thresholds": {},
            "outputs": {"write_individual_text_files": True, "truncate_csv_output_to_chars": 80},
        }
        study_path = study_dir / "test.yaml"
        study_path.write_text(yaml.dump(study_cfg))
        return study_path

    def _mock_generate_one(self, model, tokenizer, prompt_text, runtime_cfg, turns=None):
        return {
            "rendered_prompt": prompt_text,
            "prompt_tokens": 10,
            "output_tokens": 20,
            "latency_s": 0.5,
            "tokens_per_second": 40.0,
            "peak_vram_gb": None,
            "output_text": "Hello! This is a test response.",
        }

    @patch("turboquant_workflow_eval.study.load_model_and_tokenizer")
    @patch("turboquant_workflow_eval.study.generate_one")
    def test_full_study_run(self, mock_gen, mock_load, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer, "MockLoader")
        mock_gen.side_effect = self._mock_generate_one

        study_path = self._create_study_files(tmp_path)
        output_dir = tmp_path / "outputs"

        summary = run_workflow_study(study_path, output_dir)

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
        """Multi-policy study records the explicit baseline_policy_name."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer, "MockLoader")
        mock_gen.side_effect = self._mock_generate_one

        study_path = self._create_study_files(tmp_path)
        # Add a second enabled policy and an explicit baseline name.
        policy_dir = tmp_path / "configs" / "policies"
        second = {
            "name": "compressed",
            "enabled": True,
            "comparison_label": "compressed",
            "adapter": {
                "import_path": "turboquant_workflow_eval.adapters.none:NoCompressionAdapter",
            },
            "settings": {},
        }
        (policy_dir / "compressed.yaml").write_text(yaml.dump(second))
        study_cfg = yaml.safe_load(study_path.read_text())
        study_cfg["policy_configs"].append("../policies/compressed.yaml")
        study_cfg["baseline_policy_name"] = "baseline"
        study_path.write_text(yaml.dump(study_cfg))

        output_dir = tmp_path / "outputs"
        summary = run_workflow_study(study_path, output_dir)
        assert summary["baseline_policy_name"] == "baseline"
        on_disk = json.loads((output_dir / "run_summary.json").read_text())
        assert on_disk["baseline_policy_name"] == "baseline"

    @patch("turboquant_workflow_eval.study.load_model_and_tokenizer")
    @patch("turboquant_workflow_eval.study.generate_one")
    def test_error_recovery(self, mock_gen, mock_load, tmp_path: Path) -> None:
        """Test that a failing prompt doesn't crash the whole study."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer, "MockLoader")

        call_count = 0

        def gen_with_error(model, tokenizer, prompt_text, runtime_cfg, turns=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Warmup call succeeds
                return self._mock_generate_one(model, tokenizer, prompt_text, runtime_cfg)
            # Actual prompt call fails
            raise RuntimeError("GPU OOM")

        mock_gen.side_effect = gen_with_error

        study_path = self._create_study_files(tmp_path)
        output_dir = tmp_path / "outputs"

        summary = run_workflow_study(study_path, output_dir)
        # Study should complete despite the error
        assert summary["row_count"] == 1
        rows_file = output_dir / "rows.jsonl"
        row = json.loads(rows_file.read_text().strip())
        assert "error" in row
        assert "GPU OOM" in row["error"]

    @patch("turboquant_workflow_eval.study.load_model_and_tokenizer")
    @patch("turboquant_workflow_eval.study.generate_one")
    def test_verdict_summary(self, mock_gen, mock_load, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer, "MockLoader")
        mock_gen.side_effect = self._mock_generate_one

        study_path = self._create_study_files(tmp_path)
        output_dir = tmp_path / "outputs"

        summary = run_workflow_study(study_path, output_dir)
        assert "verdict_summary" in summary
        assert all(k in summary["verdict_summary"] for k in ("green", "yellow", "red"))

    def test_invalid_study_config(self, tmp_path: Path) -> None:
        """Test that validation catches missing required fields."""
        bad_cfg = {"name": "bad"}
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(bad_cfg))

        with pytest.raises(Exception, match="missing required field"):
            run_workflow_study(path, tmp_path / "out")
