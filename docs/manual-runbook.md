# Manual runbook

This document is the step-by-step path for running the repository manually on RunPod.

> **Alternative:** Instead of following the CLI steps below, you can run `make ui` after setting up the environment (steps 1-2) to use the Gradio web UI at `http://0.0.0.0:7860`. The UI covers steps 3-8 interactively.

## 1. Create the environment

```bash
python -m venv /workspace/venvs/turboquant-eval
source /workspace/venvs/turboquant-eval/bin/activate
python -m pip install --upgrade pip "setuptools<82" wheel ninja
python -m pip install -r requirements-runpod-cu128.txt
python -m pip install -r requirements.txt
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export PYTHONPATH=$PWD/src
export TOKENIZERS_PARALLELISM=false
```

### Why this order matters

On the validated RunPod environment, the system CUDA toolkit is 12.8. Installing `torch==2.10.0+cu128` first keeps the Python stack aligned with that toolkit. Avoid installing a generic latest torch wheel into this venv.

### turboquant-core is vendored

`turboquant-core` is vendored in-tree at `vendor/turboquant-core/` (added via `git subtree`). It is no longer a pip dependency. To pick up upstream changes, just `git pull` like any other in-tree code — there is no SHA to bump in `requirements.txt` or `pyproject.toml`. The standalone `Flaker420/turboquant-core` repo is archived read-only.

## 2. Optional fast-path packages

```bash
python -m pip install -U "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention"
python -m pip install causal-conv1d
```

Notes:
- `fla` and `causal_conv1d` may import successfully while Qwen still prints a fast-path warning in Transformers.
- Treat that warning as non-blocking unless profiling proves otherwise.

## 3. Validate imports and GPU visibility

```bash
python scripts/validate_environment.py
```

## 4. Warm the Hugging Face cache

Download all configured models:

```bash
python scripts/download_model.py --all
```

Download a single model (Qwen3.5-9B, default):

```bash
python scripts/download_model.py --model-config configs/model/qwen35_9b_text_only.yaml
```

Download a single model (Qwen3-8B):

```bash
python scripts/download_model.py --model-config configs/model/qwen3_8b.yaml
```

Tokenizer only:

```bash
python scripts/download_model.py --model-config configs/model/qwen35_9b_text_only.yaml --tokenizer-only
```

Check what is already cached:

```bash
python scripts/download_model.py --check-only
```

Parameters:
- `--all` -- download all models in `configs/model/`
- `--model-config PATH` -- single model config
- `--check-only` -- show cache status without downloading
- `--tokenizer-only` -- skip full model weights
- `--skip-cached` / `--no-skip-cached` -- skip or force re-download of cached models
- `--max-retries N` -- retry count on failure (default: 3)
- `--fallback-tokenizer-only` / `--no-fallback` -- auto-fallback to tokenizer-only on model download failure
- `--output PATH` -- write JSON summary

## 5. Run tests

```bash
pytest -q
```

## 6. Dry-run validation (no GPU)

Before committing GPU hours, verify all configs, file paths, and adapter imports resolve correctly:

```bash
python -m turboquant_workflow_eval --study-config configs/studies/default_qwen35_9b.yaml --dry-run
```

This runs in <1 second and prints an execution plan:

```
Study:       default_workflow_study
Model:       Qwen/Qwen3.5-9B
Policies:    3
Prompts:     14
Repetitions: 3
Total gens:  126
All configs valid. Ready to run.
```

You can combine `--dry-run` with prompt filtering and `--set` overrides to validate specific scenarios.

## 7. List discovered attention blocks

```bash
python scripts/list_attention_blocks.py --model-config configs/model/qwen35_9b_text_only.yaml
```

Optional parameters:
- `--model-config PATH`
- `--output PATH`

## 8. Run preflight instrumentation

```bash
python scripts/run_preflight_stats.py \
  --experiment-config configs/experiments/preflight_stats.yaml \
  --output-dir outputs/preflight_smoke
```

Optional parameters:
- `--experiment-config PATH`
- `--output-dir PATH`
- `--prompts-file PATH`

## 9. Run the workflow study

### Quick smoke test (one prompt, one policy)

Before running the full matrix, verify that a single prompt works end-to-end:

```bash
python scripts/run_workflow_study.py \
  --study-config configs/studies/default_qwen35_9b.yaml \
  --single --prompt-id math_01
```

### Full study runs

Baseline only:

```bash
python scripts/run_workflow_study.py \
  --study-config configs/studies/default_qwen35_9b.yaml \
  --policy-configs configs/policies/baseline.yaml \
  --output-dir outputs/study_baseline
```

Baseline + safe policy:

```bash
python scripts/run_workflow_study.py \
  --study-config configs/studies/default_qwen35_9b.yaml \
  --policy-configs configs/policies/baseline.yaml,configs/policies/safe_template.yaml \
  --output-dir outputs/study_compare
```

### CLI overrides (no YAML editing needed)

Override runtime parameters directly from the command line:

```bash
python scripts/run_workflow_study.py \
  --study-config configs/studies/default_qwen35_9b.yaml \
  --set runtime.max_new_tokens=64 --repetitions 5
```

Run only coding prompts:

```bash
python scripts/run_workflow_study.py \
  --study-config configs/studies/default_qwen35_9b.yaml \
  --prompt-category coding
```

Filter prompts by regex:

```bash
python scripts/run_workflow_study.py \
  --study-config configs/studies/default_qwen35_9b.yaml \
  --prompt-filter "math|cagr"
```

### Parameters
- `--study-config PATH`
- `--policy-configs PATH1,PATH2,...`
- `--model-config PATH`
- `--output-dir PATH`
- `--set KEY=VALUE` (repeatable, dot-notation)
- `--repetitions N`
- `--prompt-id ID` (repeatable)
- `--prompt-category CAT` (repeatable)
- `--prompt-filter REGEX`
- `--single`
- `--dry-run`

### Notes

- Results are written incrementally to `rows.jsonl` as each prompt completes. If the run crashes, partial results are preserved on disk.
- The model is loaded once and reused across policies when the adapter supports `can_revert()`. This saves 30-60s per policy for a 9B model.

## 10. Re-score existing results (no GPU)

After reviewing study outputs, you can re-score with different thresholds without re-running inference:

```bash
python -m turboquant_workflow_eval \
  --rescore outputs/study_compare/rows.jsonl \
  --set thresholds.latency_red_pct=50
```

`--study-config` is **optional** in `--rescore` mode. When supplied, the study YAML's `thresholds:` block is used as the base and `--set` overrides are layered on top:

```bash
python -m turboquant_workflow_eval \
  --study-config configs/studies/default_qwen35_9b.yaml \
  --rescore outputs/study_compare/rows.jsonl \
  --set thresholds.latency_red_pct=50
```

Bare `--set latency_red_pct=50` (without the `thresholds.` prefix) is also accepted in rescore mode and is normalized to a `thresholds.` override automatically.

The baseline policy is taken from the sibling `run_summary.json` next to the rows JSONL, so you do not need to repeat `baseline_policy_name` on the command line. A refreshed `run_summary.json` is always written next to the rescored rows with `rescored: true`, `rescore_thresholds`, and `rescore_verdicts_changed` recording exactly what was applied.

This recomputes verdicts on existing `rows.jsonl` and rewrites the CSV and markdown files. The Gradio UI also has a **Re-Score** tab with threshold sliders for interactive tuning.

Per-category thresholds are supported (see `scoring._resolve_thresholds`):

```bash
python -m turboquant_workflow_eval \
  --rescore outputs/study_compare/rows.jsonl \
  --set thresholds.default.latency_red_pct=25 \
  --set thresholds.math.latency_red_pct=50
```

## 11. Generate additional prompts (optional)

Generate long-context evaluation prompts using the target model:

```bash
python scripts/generate_prompts.py \
  --model-config configs/model/qwen35_9b_text_only.yaml \
  --output prompts/generated_long_context.yaml \
  --max-new-tokens 2048
```

Optional parameters:
- `--model-config PATH`
- `--output PATH`
- `--max-new-tokens N`

The generated YAML follows the same schema as the fixed prompt pack. To use it in a study, pass `configs/studies/full.yaml` as the study config (which references both the fixed and generated prompt packs) or use `make study-full`.

## 12. Expected outputs

Workflow study outputs:
- `workflow_compare.csv`
- `rows.jsonl`
- `examples.md`
- `run_summary.json`
- `text_outputs/` directory with one file per prompt/policy
