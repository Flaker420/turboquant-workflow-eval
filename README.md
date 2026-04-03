# Qwen3.5-9B TurboQuant Workflow Study

A practical, **RunPod-first** repository for answering a practical question:

> Which compression policy is usable in my workflow, and what breaks when I push it harder?

This repository is **not** a paper-style ablation project. It is a compact evaluation harness for **Qwen3.5-9B**, with:

- model discovery and preflight instrumentation
- a fixed prompt pack for reasoning, math, coding, and retrieval
- reproducible CSV / JSONL / Markdown outputs
- a pluggable adapter interface for integrating a real TurboQuant backend
- RunPod-oriented bootstrap and storage conventions

## Ready-to-run status

The scaffold is **ready to run** for:
- RunPod environment setup with the validated cu128 torch stack
- dependency installation
- explicit model/tokenizer cache warmup
- preflight instrumentation
- baseline workflow study

It now includes a **local Transformers-side patch adapter** for a conservative full-attention-only workflow comparison. This adapter is a behavioral proxy that quantizes K/V projection outputs in the 8 full-attention layers. It is useful for practical evaluation, but it does **not** claim true KV-cache memory savings.

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

What is included now:
- a pass-through baseline adapter
- a local Transformers-side safe adapter for conservative full-attention-only comparison

If you later add a true TurboQuant backend, you can still point the policies at a different adapter class. The adapter interface is intentionally small and documented in `docs/adapter-interface.md`.

## RunPod-first design

The default scripts assume a **network volume mounted at `/workspace`** and keep caches, environments, and outputs there.

Recommended directories on the mounted volume:

- `/workspace/venvs/qwen35-turboquant-study`
- `/workspace/.cache/huggingface`
- `/workspace/outputs`

## Current validated RunPod state

The current validated environment for this repository is:

- system CUDA toolkit / `nvcc`: **12.8**
- PyTorch: **2.10.0+cu128**
- Triton: **3.6.0**
- native Qwen3.5 non-thinking toggle enabled through local `apply_chat_template(..., enable_thinking=False)`

This is why the RunPod bootstrap installs torch separately from the base requirements. See `docs/current-runpod-state.md` for the full note and caveats.

## Fastest path on RunPod

### 1) Bootstrap, install requirements, and optionally warm the model cache

Install the validated RunPod environment only:

```bash
bash scripts/bootstrap_runpod.sh
source .env.runpod
source /workspace/venvs/qwen35-turboquant-study/bin/activate
```

Install the validated environment and warm the model cache:

```bash
bash scripts/bootstrap_runpod.sh --download-model
source .env.runpod
source /workspace/venvs/qwen35-turboquant-study/bin/activate
```

Install the validated environment and attempt the optional fast-path packages:

```bash
bash scripts/bootstrap_runpod.sh --download-model --fast-path
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

Baseline plus the built-in safe policy:

```bash
make study \
  POLICY_CONFIGS=configs/policies/baseline.yaml,configs/policies/safe_template.yaml \
  OUTPUT_DIR=outputs/study_compare
```

After that, you can optionally enable the aggressive template for a second-pass stress comparison.

## Manual step-by-step path

The manual runbook is in:

- `docs/manual-runbook.md`

That document includes each step separately plus the parameters for each script.

## Script usage summary

### `scripts/bootstrap_runpod.sh`

```bash
bash scripts/bootstrap_runpod.sh [--download-model] [--tokenizer-only] [--fast-path] [--model-config PATH]
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
- treat the built-in local patch as a practical proxy, not as proof of TurboQuant-equivalent memory compression
