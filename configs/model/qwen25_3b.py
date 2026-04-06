"""Qwen2.5-3B-Instruct dense attention model config."""

from turboquant_workflow_eval.schema import ContextConfig, LayoutConfig, ModelConfig

MODEL = ModelConfig(
    model_name="Qwen/Qwen2.5-3B-Instruct",
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
        pattern="36 x dense attention (GQA: 16 Q heads / 2 KV heads, head_dim 128)",
    ),
    context=ContextConfig(short=2048, medium=16384, long=None),
    notes=(
        "Dense attention model — all 36 layers use standard attention.",
        "Uses Qwen25DenseKVBackend from turboquant-core for compression.",
        "GQA layout 16 Q / 2 KV heads, head_dim 128 (matches the backend defaults).",
    ),
)
