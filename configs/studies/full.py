"""Full study: 3 policies x 2 prompt packs (workflow + long-context)."""

from pathlib import Path

from turboquant_workflow_eval.loader import load_model_module, load_policy_module
from turboquant_workflow_eval.schema import (
    OutputsConfig,
    RuntimeConfig,
    StudyConfig,
)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

STUDY = StudyConfig(
    name="turboquant_workflow_eval_full",
    model=load_model_module(_HERE.parent / "model" / "qwen35_9b_text_only.py"),
    prompt_pack=(
        _REPO_ROOT / "prompts" / "workflow_prompts.yaml",
        _REPO_ROOT / "prompts" / "generated_long_context.yaml",
    ),
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
        shuffle_policies=True,
    ),
    outputs=OutputsConfig(
        write_individual_text_files=True,
        truncate_csv_output_to_chars=180,
    ),
)
