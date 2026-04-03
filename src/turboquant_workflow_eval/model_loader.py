from __future__ import annotations

from typing import Any

import torch


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def _build_model_kwargs(model_cfg: dict) -> dict:
    kwargs = {
        "torch_dtype": resolve_torch_dtype(model_cfg["dtype"]),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "device_map": model_cfg.get("device_map", "auto"),
    }
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    if model_cfg.get("language_model_only"):
        kwargs["language_model_only"] = True
    return kwargs


def load_model_and_tokenizer(model_cfg: dict) -> tuple[Any, Any, str]:
    import transformers

    model_name = model_cfg["model_name"]
    model_kwargs = _build_model_kwargs(model_cfg)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    errors: list[str] = []
    for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(model_name, **model_kwargs)
            return model, tokenizer, loader_name
        except TypeError as exc:
            if "language_model_only" in str(exc):
                retry_kwargs = dict(model_kwargs)
                retry_kwargs.pop("language_model_only", None)
                try:
                    model = loader.from_pretrained(model_name, **retry_kwargs)
                    return model, tokenizer, loader_name + " (without language_model_only)"
                except Exception as retry_exc:  # noqa: BLE001
                    errors.append(f"{loader_name}: {retry_exc}")
            else:
                errors.append(f"{loader_name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{loader_name}: {exc}")

    detail = "\n".join(errors) if errors else "no loader was available"
    raise RuntimeError(f"Failed to load model {model_name}.\n{detail}")


def resolve_language_model_root(model: Any) -> Any:
    for attr in ("language_model", "model", "text_model"):
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, "named_modules"):
            return candidate
    return model


def infer_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
