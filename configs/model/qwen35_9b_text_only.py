"""Qwen3.5-9B text-only model config (vision tower stripped)."""

from turboquant_workflow_eval.schema import ContextConfig, LayoutConfig, ModelConfig

MODEL = ModelConfig(
    model_name="Qwen/Qwen3.5-9B",
    dtype="bfloat16",
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
    chat_template_mode="auto",
    language_model_only=True,
    layout=LayoutConfig(
        total_lm_layers=32,
        attention_blocks=8,
        deltanet_blocks=24,
        pattern="8 x (3 x GatedDeltaNet + 1 x GatedAttention)",
    ),
    context=ContextConfig(short=2048, medium=16384, long=None),
    notes=(
        "Designed for RunPod pods with a network volume mounted at /workspace.",
        "Long context should be chosen based on the GPU and real memory overhead.",
    ),
)
