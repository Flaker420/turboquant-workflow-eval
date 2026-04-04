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

## 6. List discovered attention blocks

```bash
python scripts/list_attention_blocks.py --model-config configs/model/qwen35_9b_text_only.yaml
```

Optional parameters:
- `--model-config PATH`
- `--output PATH`

## 7. Run preflight instrumentation

```bash
python scripts/run_preflight_stats.py \
  --experiment-config configs/experiments/preflight_stats.yaml \
  --output-dir outputs/preflight_smoke
```

Optional parameters:
- `--experiment-config PATH`
- `--output-dir PATH`
- `--prompts-file PATH`

## 8. Run the workflow study

Baseline only:

```bash
python scripts/run_workflow_study.py \
  --study-config configs/studies/default.yaml \
  --policy-configs configs/policies/baseline.yaml \
  --output-dir outputs/study_baseline
```

Baseline + safe policy:

```bash
python scripts/run_workflow_study.py \
  --study-config configs/studies/default.yaml \
  --policy-configs configs/policies/baseline.yaml,configs/policies/safe_template.yaml \
  --output-dir outputs/study_compare
```

Optional parameters:
- `--study-config PATH`
- `--policy-configs PATH1,PATH2,...`
- `--output-dir PATH`

## 9. Generate additional prompts (optional)

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

## 10. Expected outputs

Workflow study outputs:
- `workflow_compare.csv`
- `rows.jsonl`
- `examples.md`
- `run_summary.json`
- `text_outputs/` directory with one file per prompt/policy
