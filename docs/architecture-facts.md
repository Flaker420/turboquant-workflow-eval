# Architecture facts to keep fixed

This study is intentionally **not** exploring Qwen3.5-9B architecture design space.

The following are treated as **frozen priors** for the workflow study:

- 32 language-model layers
- hidden schedule: 8 × (3 × Gated DeltaNet + 1 × Gated Attention)
- one full-attention layer every 4 layers
- linear-attention convolution kernel = 4
- linear key head dimension = 128
- linear value head dimension = 128
- linear key-head count = 16
- for Qwen3.5-9B specifically, linear value-head count = 32

The implication is simple:

- do **not** spend time tuning the DeltaNet architecture
- do **not** treat the 3:1 schedule as a first-pass knob
- do focus on **compression policy**, memory, latency, and output quality

This repository studies:

- what works in a real workflow
- what fails under conservative and aggressive compression policies
- where the practical cliff appears
