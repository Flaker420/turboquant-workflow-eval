# Current RunPod state

This repository has been validated on a RunPod environment with the following effective runtime stack:

- Python 3.12
- NVIDIA driver reporting CUDA 13.0 capability
- `nvcc` / system toolkit: CUDA 12.8
- `torch==2.10.0+cu128`
- `torch.version.cuda == 12.8`
- `triton==3.6.0`
- `flash-linear-attention` importable through the `fla` namespace
- `causal_conv1d` importable

## Supported models

- **Qwen3.5-9B** -- primary validated model (hybrid architecture)
- **Qwen3-8B** -- also supported via `configs/model/qwen3_8b.yaml` (dense attention)
- **Qwen2.5-3B-Instruct** -- also supported via `configs/model/qwen25_3b.yaml` (dense attention, GQA 16/2, head_dim 128)

## Important caveat

The Qwen 3.5 Transformers implementation may still emit:

> The fast path is not available because one of the required library is not installed.

Even when the module-level checks and imports succeed. Treat this as a likely false-positive warning in the current Transformers/Qwen code path unless profiling proves otherwise.

## What is validated

Validated and working in this environment:

- unit tests
- attention-block discovery
- preflight instrumentation
- baseline workflow study
- native Qwen3.5 non-thinking toggle via `enable_thinking=False` in local `apply_chat_template(...)` (the harness probes the tokenizer and falls back gracefully on Llama / Mistral / Gemma tokenizers that do not accept the kwarg)

## What this means for the repo

- prefer the provided RunPod bootstrap or the manual runbook exactly as written
- do not install generic latest torch into the venv on this Pod
- do not use `pip install -U causal-conv1d` casually; it may spawn an isolated build environment with a mismatched torch stack
