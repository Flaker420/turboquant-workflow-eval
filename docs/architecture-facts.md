# Architecture facts

This document records the frozen architecture priors for each supported model. The framework treats these as fixed -- it tests, implements, and benchmarks **compression policy**, not architecture design.

## Qwen3.5-9B (hybrid)

- 32 language-model layers
- hidden schedule: 8 x (3 x Gated DeltaNet + 1 x Gated Attention)
- one full-attention layer every 4 layers
- linear-attention convolution kernel = 4
- linear key head dimension = 128
- linear value head dimension = 128
- linear key-head count = 16
- linear value-head count = 32

TurboQuant-core backend: `Qwen35KVBackend` -- compresses 8 full-attention layers while leaving DeltaNet layers unchanged. Keys use TQ_prod quantization, values use TQ_MSE.

## Qwen3-8B (dense)

- 36 language-model layers
- all layers use standard attention (no DeltaNet)

TurboQuant-core backend: `Qwen3DenseKVBackend` -- applies compression uniformly across all 36 layers. Keys use TQ_prod quantization, values use TQ_MSE.

## Qwen2.5-3B-Instruct (dense)

- 36 language-model layers
- all layers use standard attention
- 16 query heads / 2 KV heads (GQA), head_dim 128

TurboQuant-core backend: `Qwen25DenseKVBackend` (hook helper: `patch_qwen25_with_tq`) -- applies compression uniformly across all 36 layers. Keys use TQ_prod quantization, values use TQ_MSE. Config: `configs/model/qwen25_3b.py`; bundled study: `configs/studies/default_qwen25_3b.py`.

## What this means for the evaluation

- Do **not** spend time tuning architecture internals
- Do **not** treat the layer schedule as a first-pass knob
- Do focus on **compression policy**, memory, latency, and output quality
- The framework benchmarks what works in practice and where the practical cliff appears for each model
