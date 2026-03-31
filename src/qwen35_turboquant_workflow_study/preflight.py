from __future__ import annotations

import torch

from .hooks import ProjectionHookBook, ProjectionHookManager
from .model_loader import infer_model_device


def run_preflight(model, tokenizer, language_model_root, attention_blocks, prompts, max_length: int, use_cache: bool, loader_name: str) -> dict:
    model.eval()
    device = infer_model_device(model)
    prompt_lengths: list[int] = []

    hook_book = ProjectionHookBook(attention_blocks)
    with ProjectionHookManager(language_model_root, attention_blocks, hook_book):
        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            prompt_lengths.append(int(encoded["input_ids"].shape[-1]))
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                model(**encoded, use_cache=use_cache)

    return {
        "loader": loader_name,
        "prompt_count": len(prompts),
        "prompt_lengths": prompt_lengths,
        "attention_blocks": [block.to_dict() for block in attention_blocks],
        "layer_stats": hook_book.to_dict(),
    }
