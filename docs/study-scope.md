# Study scope

This repository is designed for a **compact engineering study**, not a research-paper ablation.

## Primary question

Which compression policy is usable in my workflow, and what degrades when I push harder?

## Supported models

- **Qwen3.5-9B** -- hybrid architecture (8 full-attention + 24 DeltaNet layers), uses `Qwen35KVBackend`. Config: `configs/model/qwen35_9b_text_only.py`. Default study: `configs/studies/default_qwen35_9b.py`.
- **Qwen3-8B** -- dense attention (36 layers), uses `Qwen3DenseKVBackend`. Config: `configs/model/qwen3_8b.py`. Default study: `configs/studies/default_qwen3_8b.py`.
- **Qwen2.5-3B-Instruct** -- dense attention (36 layers, GQA: 16 Q / 2 KV heads, head_dim 128), uses `Qwen25DenseKVBackend`. Config: `configs/model/qwen25_3b.py`. Default study: `configs/studies/default_qwen25_3b.py`.

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
| **Re-score** | `--rescore ROWS_JSONL` | No | Recompute verdicts on existing results with different thresholds; `--study` is optional and serves as a thresholds source |

All modes support per-knob flags (e.g. `--bit-width`, `--max-new-tokens`, `--latency-red-pct`), per-policy `--KNOB-for NAME=VALUE` overrides, the generic `--set DOT.PATH=VALUE` / `--set-policy NAME.DOT.KEY=VALUE` escape hatches, and prompt filtering (`--prompt-id`, `--prompt-category`, `--prompt-filter`). See `python -m turboquant_workflow_eval --help` for the full list.

## Study configuration

Studies are Python modules under `configs/studies/` that export a frozen `StudyConfig` instance (`src/turboquant_workflow_eval/schema.py`). Multi-policy studies must declare `baseline_policy_name=` — this names the policy whose rows are used as the comparison baseline for similarity, latency delta, and verdicts. `StudyConfig.__post_init__` raises `ConfigValidationError` immediately if the field is missing on a multi-policy run; single-policy studies auto-default it to that policy's name. All bundled studies declare `baseline_policy_name="baseline"`.

## Live verdicts and early stop

Verdicts are computed live as each prompt finishes — `run_policy` populates a shared baseline lookup as the baseline policy executes and stamps a real `verdict`, `semantic_similarity`, and `output_length_delta_pct` on every row before emitting `prompt_completed`. This means the early-stop controller can react to red verdicts as they happen. Two early-stop knobs live on `EarlyStopConfig`:

```python
from turboquant_workflow_eval.schema import EarlyStopConfig, StudyConfig

STUDY = StudyConfig(
    # ...
    early_stop=EarlyStopConfig(
        max_red_verdicts=5,   # stop after this many red verdicts across all policies
        max_error_rate=0.5,   # stop if errored / total exceeds this fraction
    ),
)
```

Both can also be set from the CLI: `--max-red-verdicts 5 --max-error-rate 0.5`.

`score_results` runs at the end of the study and re-stamps the same fields idempotently.

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
- error: the prompt did not complete (the row carries `error: <message>`, `verdict: "error"`, and zeroed metrics)

Verdicts are computed automatically by `scoring.py:compute_verdict()` using configurable thresholds defined in the study config (e.g. `configs/studies/default_qwen35_9b.py`). The thresholds cover latency regression, output-length delta, semantic similarity, math correctness, and code execution results. Thresholds can be set per-category (e.g. different latency tolerance for math vs coding prompts).

Results can be re-scored with `--rescore` or the **Re-Score** tab in the Gradio UI without re-running inference.
