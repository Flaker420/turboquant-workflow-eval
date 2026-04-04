# Study scope

This repository is designed for a **compact engineering study**, not a research-paper ablation.

## Primary question

Which compression policy is usable in my workflow, and what degrades when I push harder?

## Supported models

- **Qwen3.5-9B** -- hybrid architecture (8 full-attention + 24 DeltaNet layers), uses `Qwen35KVBackend`
- **Qwen3-8B** -- dense attention (36 layers), uses `Qwen3DenseKVBackend`

Select the model through the `--model-config` flag or the study config YAML.

## Non-goals

Not in scope for the first pass:

- architecture redesign
- exhaustive paper-style ablations
- full theoretical validation of QJL
- maximum-throughput serving optimization
- publication-grade benchmarking breadth

## Deliverables

A successful run should produce:

- `workflow_compare.csv`
- `rows.jsonl`
- `examples.md`
- `run_summary.json`
- per-prompt text files in `text_outputs/`

These are meant to be read side by side when deciding:

- green: good enough for workflow use
- yellow: usable with caution
- red: too lossy or unstable

These verdicts are now computed automatically by `scoring.py:compute_verdict()` using configurable thresholds defined in the study config (e.g. `configs/studies/default.yaml`). The thresholds cover latency regression, output-length delta, semantic similarity, math correctness, and code execution results.
