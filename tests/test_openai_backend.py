from qwen35_turboquant_workflow_study.backends.openai_compatible import OpenAICompatibleBackend


def test_backend_description_contains_model():
    cfg = {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "Qwen/Qwen3.5-9B",
        "request_defaults": {"max_tokens": 128},
    }
    backend = OpenAICompatibleBackend(cfg)
    desc = backend.describe()
    assert desc["backend"] == "openai-compatible"
    assert desc["model"] == "Qwen/Qwen3.5-9B"
