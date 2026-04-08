# Project scope

This repository is a model-agnostic **test, implementation, and benchmarking framework** for TurboQuant-style KV-cache compression. It builds on the community's published findings as a starting point for its own exploration and is the practical counterpart to [turboquant-core](https://github.com/Flaker420/turboquant-core).

## What this framework is for

- test TurboQuant-style compression policies on any HuggingFace model
- implement new adapters and new policies against the stable contract in `docs/adapter-interface.md`
- benchmark compression configurations against an uncompressed baseline using deterministic divergence + theoretical KV-cache compression metrics

## Supported models

- **Qwen3.5-9B** -- hybrid architecture (8 full-attention + 24 DeltaNet layers), uses `Qwen35KVBackend`. Config: `configs/model/qwen35_9b_text_only.py`. Default study: `configs/studies/default_qwen35_9b.py`.
- **Qwen3-8B** -- dense attention (36 layers), uses `Qwen3DenseKVBackend`. Config: `configs/model/qwen3_8b.py`. Default study: `configs/studies/default_qwen3_8b.py`.
- **Qwen2.5-3B-Instruct** -- dense attention (36 layers, GQA: 16 Q / 2 KV heads, head_dim 128), uses `Qwen25DenseKVBackend`. Config: `configs/model/qwen25_3b.py`. Default study: `configs/studies/default_qwen25_3b.py`.

Select the model through the `--model-config` flag or by editing the study config module.

## Non-goals

This framework is opinionated about what it is **not** in scope for:

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
| **Re-score** | `--rescore ROWS_JSONL` | No | Recompute post-hoc divergence + KV-cache metrics from existing results without a GPU; takes no threshold knobs |

All modes support per-knob flags (e.g. `--bit-width`, `--max-new-tokens`, `--max-error-rate`), per-policy `--KNOB-for NAME=VALUE` overrides, the generic `--set DOT.PATH=VALUE` / `--set-policy NAME.DOT.KEY=VALUE` escape hatches, and prompt filtering (`--prompt-id`, `--prompt-category`, `--prompt-filter`). See `python -m turboquant_workflow_eval --help` for the full list.

## Study configuration

Studies are Python modules under `configs/studies/` that export a frozen `StudyConfig` instance (`src/turboquant_workflow_eval/schema.py`). Multi-policy studies must declare `baseline_policy_name=` — this names the policy whose rows are used as the comparison baseline for divergence + KV-cache-byte scoring. `StudyConfig.__post_init__` raises `ConfigValidationError` immediately if the field is missing on a multi-policy run; single-policy studies auto-default it to that policy's name. All bundled studies declare `baseline_policy_name="baseline"`.

## Post-hoc scoring and early stop

Scoring is post-hoc. `run_policy` no longer stamps a per-row verdict mid-run; the divergence and KV-cache columns are computed in the post-hoc `score_results` join after the study has finished, since both metrics need every prompt's baseline counterpart to exist before they make sense. The early-stop controller has only one live arm:

```python
from turboquant_workflow_eval.schema import EarlyStopConfig, StudyConfig

STUDY = StudyConfig(
    # ...
    early_stop=EarlyStopConfig(
        max_error_rate=0.5,    # stop if errored / total exceeds this fraction
    ),
)
```

The corresponding CLI flag is `--max-error-rate`. The previous `max_red_verdicts` knob is gone with the verdict pipeline.

## Deliverables

A successful run produces:

- `workflow_compare.csv`
- `rows.jsonl` (written incrementally -- partial results survive crashes)
- `examples.md`
- `run_summary.json`
- per-prompt text files in `text_outputs/`

Each row carries divergence-vs-baseline and theoretical KV-cache-compression columns populated by `score_results` after every policy has finished. See the `## Row schema` and `## Why these metrics` sections of `README.md` for the full field list and the rationale for each column. Failed rows carry `error: <message>`, `output_token_ids == []`, `output_tokens == 0`, and every divergence / KV-cache column written as `None` — filter on `error` (or `output_tokens == 0`) to find them.

Results can be re-scored with `--rescore` without re-running inference; the recomputation is deterministic and takes no threshold knobs.
