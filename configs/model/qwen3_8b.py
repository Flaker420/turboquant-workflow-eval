"""Qwen3-8B dense attention model config."""

from turboquant_workflow_eval.schema import ContextConfig, LayoutConfig, ModelConfig

MODEL = ModelConfig(
    model_name="Qwen/Qwen3-8B",
    dtype="bfloat16",
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
    chat_template_mode="auto",
    language_model_only=False,
    layout=LayoutConfig(
        total_lm_layers=36,
        attention_blocks=36,
        deltanet_blocks=0,
        pattern="36 x dense attention",
    ),
    context=ContextConfig(short=2048, medium=16384, long=None),
    notes=(
        "Dense attention model — all 36 layers use standard attention.",
        "Uses Qwen3DenseKVBackend from turboquant-core for compression.",
    ),
)
