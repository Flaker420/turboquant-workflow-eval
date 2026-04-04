from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from turboquant_workflow_eval.generation import render_prompt, generate_one


class TestRenderPrompt:
    def test_no_chat_template(self) -> None:
        tok = MagicMock()
        tok.chat_template = None
        result = render_prompt(tok, "Hello world")
        assert result == "Hello world"

    def test_with_chat_template(self) -> None:
        tok = MagicMock()
        tok.chat_template = "some_template"
        tok.apply_chat_template = MagicMock(return_value="<|user|>Hello<|assistant|>")
        result = render_prompt(tok, "Hello")
        tok.apply_chat_template.assert_called_once()
        args = tok.apply_chat_template.call_args
        assert args[0][0] == [{"role": "user", "content": "Hello"}]
        assert result == "<|user|>Hello<|assistant|>"

    def test_with_turns(self) -> None:
        tok = MagicMock()
        tok.chat_template = "some_template"
        tok.apply_chat_template = MagicMock(return_value="rendered")
        turns = (
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        )
        result = render_prompt(tok, "", turns=turns)
        args = tok.apply_chat_template.call_args
        assert len(args[0][0]) == 3
        assert result == "rendered"

    def test_no_turns_no_template(self) -> None:
        tok = MagicMock()
        tok.chat_template = None
        result = render_prompt(tok, "Just text", turns=None)
        assert result == "Just text"


class TestGenerateOne:
    def test_basic_generation(self, mock_model, mock_tokenizer) -> None:
        runtime_cfg = {
            "max_input_tokens": 512,
            "max_new_tokens": 64,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "use_cache": True,
        }
        result = generate_one(mock_model, mock_tokenizer, "Test prompt", runtime_cfg)

        assert "latency_s" in result
        assert "output_text" in result
        assert "prompt_tokens" in result
        assert "output_tokens" in result
        assert "tokens_per_second" in result
        assert result["latency_s"] >= 0
        assert result["prompt_tokens"] == 3  # mock returns [1, 2, 3]
        assert result["output_tokens"] == 3  # 6 total - 3 prompt = 3 new

    def test_eval_mode_called(self, mock_model, mock_tokenizer) -> None:
        runtime_cfg = {"max_input_tokens": 512, "max_new_tokens": 64}
        generate_one(mock_model, mock_tokenizer, "Test", runtime_cfg)
        mock_model.eval.assert_called_once()

    def test_generate_called_with_params(self, mock_model, mock_tokenizer) -> None:
        runtime_cfg = {
            "max_input_tokens": 512,
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "use_cache": False,
        }
        generate_one(mock_model, mock_tokenizer, "Test", runtime_cfg)
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 128
        assert call_kwargs["do_sample"] is True
        assert call_kwargs["temperature"] == pytest.approx(0.7)
        assert call_kwargs["top_p"] == pytest.approx(0.9)
        assert call_kwargs["use_cache"] is False
