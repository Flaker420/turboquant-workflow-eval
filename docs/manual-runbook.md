# Manual runbook

This document is the step-by-step path for running the repository manually on RunPod.

## 1. Create the environment

```bash
python -m venv /workspace/venvs/qwen35-turboquant-study
source /workspace/venvs/qwen35-turboquant-study/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export PYTHONPATH=$PWD/src
export TOKENIZERS_PARALLELISM=false
```

## 2. Validate imports and GPU visibility

```bash
python scripts/validate_environment.py
```

## 3. Warm the Hugging Face cache

Model + tokenizer:

```bash
python scripts/download_model.py --model-config configs/model/qwen35_9b_text_only.yaml
```

Tokenizer only:

```bash
python scripts/download_model.py --model-config configs/model/qwen35_9b_text_only.yaml --tokenizer-only
```

Optional parameters:
- `--model-config PATH`
- `--tokenizer-only`
- `--output PATH`

## 4. Run tests

```bash
pytest -q
```

## 5. List discovered attention blocks

```bash
python scripts/list_attention_blocks.py --model-config configs/model/qwen35_9b_text_only.yaml
```

Optional parameters:
- `--model-config PATH`
- `--output PATH`

## 6. Run preflight instrumentation

```bash
python scripts/run_preflight_stats.py \
  --experiment-config configs/experiments/preflight_stats.yaml \
  --output-dir outputs/preflight_smoke
```

Optional parameters:
- `--experiment-config PATH`
- `--output-dir PATH`
- `--prompts-file PATH`

## 7. Run the workflow study

Baseline only:

```bash
python scripts/run_workflow_study.py \
  --study-config configs/studies/default.yaml \
  --policy-configs configs/policies/baseline.yaml \
  --output-dir outputs/study_baseline
```

Multiple policies:

```bash
python scripts/run_workflow_study.py \
  --study-config configs/studies/default.yaml \
  --policy-configs configs/policies/baseline.yaml,configs/policies/safe_template.yaml,configs/policies/aggressive_template.yaml \
  --output-dir outputs/study_compare
```

Optional parameters:
- `--study-config PATH`
- `--policy-configs PATH1,PATH2,...`
- `--output-dir PATH`

## 8. Expected outputs

Workflow study outputs:
- `workflow_compare.csv`
- `rows.jsonl`
- `examples.md`
- `run_summary.json`
- `texts/` directory with one file per prompt/policy
