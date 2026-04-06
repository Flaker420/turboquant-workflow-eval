"""Default 3-policy workflow study targeting Qwen2.5-3B-Instruct."""

from pathlib import Path

from turboquant_workflow_eval.loader import load_model_module, load_policy_module
from turboquant_workflow_eval.schema import (
    OutputsConfig,
    RuntimeConfig,
    StudyConfig,
    ThresholdsConfig,
)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

STUDY = StudyConfig(
    name="turboquant_workflow_eval_qwen25_3b",
    model=load_model_module(_HERE.parent / "model" / "qwen25_3b.py"),
    prompt_pack=(_REPO_ROOT / "prompts" / "workflow_prompts.yaml",),
    policies=(
        load_policy_module(_HERE.parent / "policies" / "baseline.py"),
        load_policy_module(_HERE.parent / "policies" / "safe_template.py"),
        load_policy_module(_HERE.parent / "policies" / "aggressive_template.py"),
    ),
    baseline_policy_name="baseline",
    runtime=RuntimeConfig(
        max_input_tokens=4096,
        max_new_tokens=384,
        repetitions=3,
    ),
    thresholds=ThresholdsConfig(
        latency_yellow_pct=10.0,
        latency_red_pct=25.0,
        similarity_yellow=0.92,
        similarity_red=0.80,
        output_length_yellow_pct=15.0,
        output_length_red_pct=30.0,
    ),
    outputs=OutputsConfig(
        write_individual_text_files=True,
        truncate_csv_output_to_chars=180,
    ),
)
