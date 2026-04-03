DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"

PHASE_0 = "preflight"
PHASE_1 = "workflow_study"

RUNPOD_WORKSPACE_ROOT = "/workspace"
RUNPOD_DEFAULT_VENV = "/workspace/venvs/turboquant-eval"
RUNPOD_DEFAULT_CACHE = "/workspace/.cache/huggingface"
RUNPOD_DEFAULT_OUTPUTS = "/workspace/outputs"

ATTENTION_NAME_HINTS = ("attention", "attn")
ATTENTION_EXCLUDE_HINTS = (
    "delta",
    "deltanet",
    "linear_attention",
    "vision",
    "visual",
    "image",
    "audio",
    "cross_attn",
    "cross_attention",
)
Q_PROJ_HINTS = ("q_proj", "query", "wq")
K_PROJ_HINTS = ("k_proj", "key", "wk")
V_PROJ_HINTS = ("v_proj", "value", "wv")
PACKED_QKV_HINTS = ("qkv", "c_attn", "qkv_proj")
