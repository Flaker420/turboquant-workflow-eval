from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from turboquant_workflow_eval.types import PromptSpec


@pytest.fixture
def sample_model_config() -> dict[str, Any]:
    return {
        "model_name": "test/tiny-model",
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": False,
    }


@pytest.fixture
def sample_policy_config() -> dict[str, Any]:
    return {
        "name": "baseline",
        "enabled": True,
        "comparison_label": "baseline",
        "adapter": {
            "import_path": "turboquant_workflow_eval.adapters.none:NoCompressionAdapter",
        },
        "settings": {},
    }


@pytest.fixture
def sample_runtime_config() -> dict[str, Any]:
    return {
        "max_input_tokens": 512,
        "max_new_tokens": 64,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "use_cache": True,
        "repetitions": 1,
    }


@pytest.fixture
def sample_prompt() -> PromptSpec:
    return PromptSpec(
        id="test_01",
        category="reasoning",
        title="test prompt",
        prompt="What is 2+2?",
        watch_for="correct answer",
    )


@pytest.fixture
def sample_math_prompt() -> PromptSpec:
    return PromptSpec(
        id="math_01",
        category="math",
        title="simple math",
        prompt="Compute 10 * 5.",
        watch_for="correct multiplication",
        reference_answer="50",
    )


@pytest.fixture
def sample_coding_prompt() -> PromptSpec:
    return PromptSpec(
        id="coding_01",
        category="coding",
        title="add function",
        prompt="Write a Python function add(a, b) that returns a + b.",
        watch_for="correct code",
        test_cases=({"input": "[1, 2]", "expected": "3"},),
    )


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """A mock tokenizer that returns predictable values."""
    tok = MagicMock()
    tok.chat_template = None  # No chat template by default

    def tokenize_call(text, return_tensors=None, truncation=False, max_length=None):
        import torch

        ids = torch.tensor([[1, 2, 3]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    tok.side_effect = tokenize_call
    tok.__call__ = tokenize_call
    tok.decode = MagicMock(return_value="mock output text")
    tok.apply_chat_template = MagicMock(return_value="<|user|>prompt<|assistant|>")
    return tok


@pytest.fixture
def mock_model() -> MagicMock:
    """A mock model that returns predictable generation output."""
    import torch

    model = MagicMock()
    model.eval = MagicMock()

    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters = MagicMock(return_value=iter([param]))

    # generate returns token ids: [prompt_tokens + new_tokens]
    model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 10, 20, 30]]))

    return model
