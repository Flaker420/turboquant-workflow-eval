# Manual runbook

This document is the step-by-step path for running the repository manually on RunPod.

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
python scripts/download_model.py --model configs/model/qwen35_9b_text_only.py
```

Download a single model (Qwen3-8B):

```bash
python scripts/download_model.py --model configs/model/qwen3_8b.py
```

Tokenizer only:

```bash
python scripts/download_model.py --model configs/model/qwen35_9b_text_only.py --tokenizer-only
```

Check what is already cached:

```bash
python scripts/download_model.py --check-only
```

Parameters:
- `--all` -- download all models in `configs/model/`
- `--model PATH` -- single model config
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
python -m turboquant_workflow_eval --study configs/studies/default_qwen35_9b.py --dry-run
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
python scripts/list_attention_blocks.py --model configs/model/qwen35_9b_text_only.py
```

Optional parameters:
- `--model PATH`
- `--output PATH`

## 8. Run preflight instrumentation

```bash
python scripts/run_preflight_stats.py \
  --experiment-config configs/experiments/preflight_stats.py \
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
  --study configs/studies/default_qwen35_9b.py \
  --single --prompt-id math_01
```

### Full study runs

Baseline only:

```bash
python scripts/run_workflow_study.py \
  --study configs/studies/default_qwen35_9b.py \
  --policies configs/policies/baseline.py \
  --output-dir outputs/study_baseline
```

Baseline + safe policy:

```bash
python scripts/run_workflow_study.py \
  --study configs/studies/default_qwen35_9b.py \
  --policies configs/policies/baseline.py,configs/policies/safe_template.py \
  --output-dir outputs/study_compare
```

### CLI overrides (no YAML editing needed)

Override runtime parameters directly from the command line:

```bash
python scripts/run_workflow_study.py \
  --study configs/studies/default_qwen35_9b.py \
  --set runtime.max_new_tokens=64 --repetitions 5
```

Run only coding prompts:

```bash
python scripts/run_workflow_study.py \
  --study configs/studies/default_qwen35_9b.py \
  --prompt-category coding
```

Filter prompts by regex:

```bash
python scripts/run_workflow_study.py \
  --study configs/studies/default_qwen35_9b.py \
  --prompt-filter "math|cagr"
```

### Parameters
- `--study PATH`
- `--policies PATH1,PATH2,...`
- `--model PATH`
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

After reviewing study outputs, you can recompute the divergence + KV-cache columns of an existing `rows.jsonl` without re-running inference:

```bash
python -m turboquant_workflow_eval \
  --rescore outputs/study_compare/rows.jsonl
```

The recomputation is deterministic and takes no threshold knobs — the post-hoc metrics are pure functions of each row's `output_token_ids` and the policy/model bookkeeping carried alongside it. The rescore path:

1. Reads the sibling `run_summary.json` for `model_info` (`num_hidden_layers`, `num_key_value_heads`, `head_dim`); if those are absent, the divergence columns are still populated but the KV-cache columns end up empty.
2. Hard-fails with one actionable error if any non-error row is missing `output_token_ids`. Rows produced before the divergence-metrics schema bump cannot be cold-rescored — rerun the study to regenerate them.
3. Writes a refreshed `run_summary.json` (with `rescored: true` and the new `divergence_summary` block), `workflow_compare.csv`, and `examples.md` next to the rows file.

The baseline policy is taken from the sibling `run_summary.json`, so you do not need to repeat `baseline_policy_name` on the command line.

## 11. Generate additional prompts (optional)

Generate long-context evaluation prompts using the target model:

```bash
python scripts/generate_prompts.py \
  --model configs/model/qwen35_9b_text_only.py \
  --output prompts/generated_long_context.yaml \
  --max-new-tokens 2048
```

Optional parameters:
- `--model PATH`
- `--output PATH`
- `--max-new-tokens N`

The generated YAML follows the same schema as the fixed prompt pack. To use it in a study, pass `configs/studies/full.py` as the study config (which references both the fixed and generated prompt packs) or use `make study-full`.

## 12. Expected outputs

Workflow study outputs:
- `workflow_compare.csv`
- `rows.jsonl`
- `examples.md`
- `run_summary.json`
- `text_outputs/` directory with one file per prompt/policy
