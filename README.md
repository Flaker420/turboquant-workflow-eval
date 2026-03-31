# Qwen3.5-9B TurboQuant Workflow Study

A professional, **RunPod-first** repository for answering a practical question:

> Which compression policy is usable in my workflow, and what breaks when I push it harder?

This repository is **not** a paper-style ablation project. It is a compact evaluation harness for **Qwen3.5-9B**, with:

- model discovery and preflight instrumentation
- a fixed prompt pack for reasoning, math, coding, and retrieval
- reproducible CSV / JSONL / Markdown outputs
- a pluggable adapter interface for integrating a real TurboQuant backend
- RunPod-oriented bootstrap and storage conventions

## Ready-to-run status

The scaffold is **ready to run** for:
- environment setup
- dependency installation
- explicit model/tokenizer cache warmup
- preflight instrumentation
- baseline workflow study

It is **not** a fake production TurboQuant package. To compare real compression policies, you still need to point the `safe` and `aggressive` policy configs at a real backend adapter.

## What is built in

Built in and ready to run:

- RunPod bootstrap with optional cache warmup
- environment validation
- Qwen3.5-9B model loading helpers
- language-model root discovery
- attention-block discovery
- Q / K / V projection capture for preflight checks
- baseline pass-through adapter
- workflow study runner and report generation
- unit tests for the scaffold

## What you still need to wire

This repository does **not** pretend to ship a production TurboQuant kernel.

To compare actual compression policies, you need to point the `safe` and `aggressive` policies at a real backend by implementing or importing an adapter class. The adapter interface is intentionally small and documented in `docs/adapter-interface.md`.

## RunPod-first design

The default scripts assume a **network volume mounted at `/workspace`** and keep caches, environments, and outputs there.

Recommended directories on the mounted volume:

- `/workspace/venvs/qwen35-turboquant-study`
- `/workspace/.cache/huggingface`
- `/workspace/outputs`

## Fastest path on RunPod

### 1) Bootstrap, install requirements, and optionally warm the model cache

Install only:

```bash
bash scripts/bootstrap_runpod.sh
source .env.runpod
source /workspace/venvs/qwen35-turboquant-study/bin/activate
```

Install + explicit full model download:

```bash
bash scripts/bootstrap_runpod.sh --download-model
source .env.runpod
source /workspace/venvs/qwen35-turboquant-study/bin/activate
```

Install + tokenizer-only cache warmup:

```bash
bash scripts/bootstrap_runpod.sh --download-model --tokenizer-only
source .env.runpod
source /workspace/venvs/qwen35-turboquant-study/bin/activate
```

### 2) Validate the scaffold

```bash
python scripts/validate_environment.py
pytest -q
```

### 3) Inspect the discovered attention blocks

```bash
make list-attention
```

### 4) Run the preflight instrumentation pass

```bash
make preflight
```

### 5) Run the workflow study

Baseline only:

```bash
make study POLICY_CONFIGS=configs/policies/baseline.yaml OUTPUT_DIR=outputs/study_baseline
```

Baseline plus your real policies after you wire adapters:

```bash
make study \
  POLICY_CONFIGS=configs/policies/baseline.yaml,configs/policies/safe_template.yaml,configs/policies/aggressive_template.yaml \
  OUTPUT_DIR=outputs/study_compare
```

## Manual step-by-step path

The manual runbook is in:

- `docs/manual-runbook.md`

That document includes each step separately plus the parameters for each script.

## Script usage summary

### `scripts/bootstrap_runpod.sh`

```bash
bash scripts/bootstrap_runpod.sh [--download-model] [--tokenizer-only] [--model-config PATH]
```

Environment overrides:
- `WORKSPACE_ROOT`
- `VENV_DIR`
- `CACHE_ROOT`
- `OUTPUT_ROOT`

### `scripts/validate_environment.py`

```bash
python scripts/validate_environment.py
```

### `scripts/download_model.py`

```bash
python scripts/download_model.py [--model-config PATH] [--tokenizer-only] [--output PATH]
```

### `scripts/list_attention_blocks.py`

```bash
python scripts/list_attention_blocks.py [--model-config PATH] [--output PATH]
```

### `scripts/run_preflight_stats.py`

```bash
python scripts/run_preflight_stats.py [--experiment-config PATH] [--output-dir PATH] [--prompts-file PATH]
```

### `scripts/run_workflow_study.py`

```bash
python scripts/run_workflow_study.py [--study-config PATH] [--policy-configs PATH1,PATH2,...] [--output-dir PATH]
```

## Outputs

For each policy you test, the repo produces concrete artifacts you can compare:

- `workflow_compare.csv`
- `rows.jsonl`
- `examples.md`
- per-prompt text files
- `run_summary.json`

That gives you a direct **works / degrades / fails** view instead of a research-heavy sweep.

## Repository layout

- `configs/` – model, policy, and study configs
- `docs/` – architecture facts, scope, RunPod setup, manual runbook, and adapter contract
- `prompts/` – fixed workflow prompt pack
- `scripts/` – RunPod bootstrap and entrypoints
- `src/` – reusable package code
- `tests/` – scaffold tests

## Recommended first decision path

Do not start with a large sweep.

1. run the baseline
2. wire one conservative compression policy
3. compare outputs in `examples.md` and `workflow_compare.csv`
4. only then try a more aggressive policy

## Notes

This repository is intentionally opinionated:

- freeze the Qwen3.5 architecture priors
- vary compression **policy**, not model architecture
- optimize for **workflow decisions**, not paper completeness
