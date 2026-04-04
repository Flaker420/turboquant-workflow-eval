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

## Execution modes

| Mode | Command | GPU needed? | Purpose |
|------|---------|-------------|---------|
| **Dry run** | `--dry-run` | No | Validate all configs, paths, and adapters in <1 second |
| **Smoke test** | `--single` | Yes | Run 1 prompt with 1 policy, 1 repetition — fast end-to-end sanity check |
| **Full study** | (default) | Yes | Run all prompts x policies x repetitions |
| **Re-score** | `--rescore ROWS_JSONL` | No | Recompute verdicts on existing results with different thresholds |

All modes support `--set KEY=VALUE` overrides and prompt filtering (`--prompt-id`, `--prompt-category`, `--prompt-filter`).

## Deliverables

A successful run produces:

- `workflow_compare.csv`
- `rows.jsonl` (written incrementally -- partial results survive crashes)
- `examples.md`
- `run_summary.json`
- per-prompt text files in `text_outputs/`

These are meant to be read side by side when deciding:

- green: good enough for workflow use
- yellow: usable with caution
- red: too lossy or unstable

Verdicts are computed automatically by `scoring.py:compute_verdict()` using configurable thresholds defined in the study config (e.g. `configs/studies/default.yaml`). The thresholds cover latency regression, output-length delta, semantic similarity, math correctness, and code execution results. Thresholds can be set per-category (e.g. different latency tolerance for math vs coding prompts).

Results can be re-scored with `--rescore` or the **Re-Score** tab in the Gradio UI without re-running inference.
