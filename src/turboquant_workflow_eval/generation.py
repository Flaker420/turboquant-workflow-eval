from __future__ import annotations

import time
from typing import Any

import torch

from .model_loader import infer_model_device


def render_prompt(tokenizer: Any, prompt_text: str, turns: tuple[dict[str, str], ...] | None = None) -> str:
    """Render a prompt through the tokenizer's chat template.

    If *turns* is provided, they are used as a multi-turn conversation;
    otherwise *prompt_text* is wrapped in a single user message.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        if turns:
            messages = list(turns)
        else:
            messages = [{"role": "user", "content": prompt_text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt_text


def _peak_vram_gb(device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / 1e9)


def _reset_peak_vram(device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    runtime_cfg: dict,
    turns: tuple[dict[str, str], ...] | None = None,
) -> dict[str, Any]:
    model.eval()
    device = infer_model_device(model)

    rendered_prompt = render_prompt(tokenizer, prompt_text, turns=turns)
    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(runtime_cfg["max_input_tokens"]),
    )
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    encoded = {key: value.to(device) for key, value in encoded.items()}

    _reset_peak_vram(device)
    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=int(runtime_cfg["max_new_tokens"]),
            do_sample=bool(runtime_cfg.get("do_sample", False)),
            temperature=float(runtime_cfg.get("temperature", 0.0)),
            top_p=float(runtime_cfg.get("top_p", 1.0)),
            use_cache=bool(runtime_cfg.get("use_cache", True)),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_s = time.perf_counter() - start

    output_ids = generated[0][prompt_tokens:]
    output_tokens = int(output_ids.shape[-1])
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    tokens_per_second = (output_tokens / latency_s) if latency_s > 0 else None

    return {
        "rendered_prompt": rendered_prompt,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "latency_s": latency_s,
        "tokens_per_second": tokens_per_second,
        "peak_vram_gb": _peak_vram_gb(device),
        "output_text": output_text,
    }
