from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from turboquant_workflow_eval.model_loader import (
    _build_model_kwargs,
    infer_model_device,
    resolve_language_model_root,
    resolve_torch_dtype,
)


class TestResolveTorchDtype:
    def test_bfloat16_variants(self) -> None:
        assert resolve_torch_dtype("bf16") is torch.bfloat16
        assert resolve_torch_dtype("bfloat16") is torch.bfloat16

    def test_float16_variants(self) -> None:
        assert resolve_torch_dtype("fp16") is torch.float16
        assert resolve_torch_dtype("float16") is torch.float16

    def test_float32_variants(self) -> None:
        assert resolve_torch_dtype("fp32") is torch.float32
        assert resolve_torch_dtype("float32") is torch.float32

    def test_case_insensitive(self) -> None:
        assert resolve_torch_dtype("BFloat16") is torch.bfloat16
        assert resolve_torch_dtype("FP16") is torch.float16

    def test_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dtype"):
            resolve_torch_dtype("int8")


class TestBuildModelKwargs:
    def test_basic(self, sample_model_config) -> None:
        kwargs = _build_model_kwargs(sample_model_config)
        assert kwargs["dtype"] is torch.bfloat16
        assert kwargs["trust_remote_code"] is False
        assert kwargs["device_map"] == "auto"

    def test_attn_implementation(self) -> None:
        cfg = {"model_name": "test", "dtype": "bf16", "attn_implementation": "sdpa"}
        kwargs = _build_model_kwargs(cfg)
        assert kwargs["attn_implementation"] == "sdpa"

    def test_language_model_only(self) -> None:
        cfg = {"model_name": "test", "dtype": "bf16", "language_model_only": True}
        kwargs = _build_model_kwargs(cfg)
        assert kwargs["language_model_only"] is True

    def test_no_optional_fields(self) -> None:
        cfg = {"model_name": "test", "dtype": "fp32"}
        kwargs = _build_model_kwargs(cfg)
        assert "attn_implementation" not in kwargs
        assert "language_model_only" not in kwargs


class TestLoadModelLanguageModelOnly:
    """Verify that language_model_only skips the multimodal loader."""

    def test_skips_image_text_loader(self) -> None:
        from turboquant_workflow_eval.model_loader import load_model_and_tokenizer

        cfg = {"model_name": "test/model", "dtype": "bf16", "language_model_only": True}
        fake_model = MagicMock()
        fake_tokenizer = MagicMock()

        mock_tf = MagicMock()
        mock_tf.AutoTokenizer.from_pretrained.return_value = fake_tokenizer
        mock_tf.AutoModelForCausalLM.from_pretrained.return_value = fake_model
        mock_tf.AutoModelForImageTextToText = MagicMock()

        with patch.dict("sys.modules", {"transformers": mock_tf}):
            model, tokenizer, loader_name = load_model_and_tokenizer(cfg)

        mock_tf.AutoModelForImageTextToText.from_pretrained.assert_not_called()
        mock_tf.AutoModelForCausalLM.from_pretrained.assert_called_once()
        assert loader_name == "AutoModelForCausalLM"
        # language_model_only should NOT be in the kwargs passed to from_pretrained
        call_kwargs = mock_tf.AutoModelForCausalLM.from_pretrained.call_args[1]
        assert "language_model_only" not in call_kwargs


class TestResolveLanguageModelRoot:
    def test_language_model_attr(self) -> None:
        inner = MagicMock()
        inner.named_modules = MagicMock(return_value=[])
        outer = MagicMock()
        outer.language_model = inner
        assert resolve_language_model_root(outer) is inner

    def test_model_attr(self) -> None:
        inner = MagicMock()
        inner.named_modules = MagicMock(return_value=[])
        outer = MagicMock(spec=["model"])
        outer.model = inner
        assert resolve_language_model_root(outer) is inner

    def test_fallback_to_self(self) -> None:
        model = MagicMock(spec=[])
        assert resolve_language_model_root(model) is model


class TestInferModelDevice:
    def test_cpu_device(self) -> None:
        model = MagicMock()
        param = torch.nn.Parameter(torch.zeros(1))
        model.parameters = MagicMock(return_value=iter([param]))
        assert infer_model_device(model) == torch.device("cpu")

    def test_no_parameters(self) -> None:
        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([]))
        assert infer_model_device(model) == torch.device("cpu")
