# TurboQuant Workflow Evaluation

A practical, **RunPod-first** evaluation harness for answering a simple question:

> Which compression policy is usable in my workflow, and what breaks when I push it harder?

This repository is the evaluation counterpart to [turboquant-core](https://github.com/Flaker420/turboquant-core), the TurboQuant KV-cache compression library. It is **not** a paper-style ablation project. It is a compact harness with:

- model discovery and preflight instrumentation
- a fixed prompt pack for reasoning, math, coding, and retrieval
- reproducible CSV / JSONL / Markdown outputs
- a pluggable adapter interface for integrating TurboQuant backends
- RunPod-oriented bootstrap and storage conventions

## Supported models

| Model | Config | turboquant-core backend |
|-------|--------|------------------------|
| Qwen3.5-9B | `configs/model/qwen35_9b_text_only.yaml` | `Qwen35KVBackend` |
| Qwen3-8B | `configs/model/qwen3_8b.yaml` | `Qwen3DenseKVBackend` |

Both models work with the same evaluation pipeline. Select the model by passing a different `--model-config` to the scripts or by editing the study/experiment config.

## Ready-to-run status

The scaffold is **ready to run** for:
- RunPod environment setup with the validated cu128 torch stack
- dependency installation
- explicit model/tokenizer cache warmup
- preflight instrumentation
- baseline workflow study

It includes a **pluggable adapter interface** and disabled template configs (`safe_template.yaml`, `aggressive_template.yaml`) for wiring a real compression backend from [turboquant-core](https://github.com/Flaker420/turboquant-core). These templates point to a stub adapter that raises an error until you replace the import path with your own backend class.

## What is built in

Built in and ready to run:

- RunPod bootstrap with optional cache warmup
- environment validation
- model loading helpers (any HuggingFace model via config)
- language-model root discovery
- attention-block discovery
- Q / K / V projection capture for preflight checks
- baseline pass-through adapter
- workflow study runner and report generation
- unit tests for the scaffold

## What you still need to wire

This repository does **not** ship a production TurboQuant kernel.

What is included now:
- a pass-through baseline adapter (runs the model with no compression)
- stub-backed safe and aggressive policy templates (disabled by default; wire a real backend to enable)

To test real compression, install [turboquant-core](https://github.com/Flaker420/turboquant-core), write an adapter class, and point a policy config at it. See `docs/adapter-interface.md` for the full contract.

## Wiring turboquant-core

This evaluation harness is designed to work with [turboquant-core](https://github.com/Flaker420/turboquant-core). To wire it:

1. Install turboquant-core in the same environment.
2. Write an adapter class that calls `Qwen35KVBackend` (for Qwen3.5-9B) or `Qwen3DenseKVBackend` (for Qwen3-8B) inside `prepare_model`.
3. Set the adapter's `import_path` in `safe_template.yaml` or `aggressive_template.yaml`.
4. Set `enabled: true` in the policy config.

See `docs/adapter-interface.md` for the full adapter contract and a concrete example.

## RunPod-first design

The default scripts assume a **network volume mounted at `/workspace`** and keep caches, environments, and outputs there.

Recommended directories on the mounted volume:

- `/workspace/venvs/turboquant-eval`
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
source /workspace/venvs/turboquant-eval/bin/activate
```

Install the validated environment and warm the model cache:

```bash
bash scripts/bootstrap_runpod.sh --download-model
source .env.runpod
source /workspace/venvs/turboquant-eval/bin/activate
```

Install the validated environment and attempt the optional fast-path packages:

```bash
bash scripts/bootstrap_runpod.sh --download-model --fast-path
source .env.runpod
source /workspace/venvs/turboquant-eval/bin/activate
```

To use Qwen3-8B instead of the default Qwen3.5-9B:

```bash
bash scripts/bootstrap_runpod.sh --download-model --model-config configs/model/qwen3_8b.yaml
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

Baseline plus the built-in safe policy (after wiring a real backend):

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
bash scripts/bootstrap_runpod.sh [--download-model] [--download-all] [--tokenizer-only] [--fast-path] [--model-config PATH]
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
python scripts/download_model.py [--model-config PATH] [--all] [--check-only] [--tokenizer-only] [--skip-cached] [--max-retries N] [--output PATH]
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
- per-prompt text files in `text_outputs/`
- `run_summary.json`

That gives you a direct **works / degrades / fails** view instead of a research-heavy sweep.

## Repository layout

- `configs/` -- model, policy, and study configs
- `docs/` -- architecture facts, scope, RunPod setup, manual runbook, and adapter contract
- `prompts/` -- fixed workflow prompt pack
- `scripts/` -- RunPod bootstrap and entrypoints
- `src/` -- reusable package code
- `tests/` -- scaffold tests

## Recommended first decision path

Do not start with a large sweep.

1. run the baseline
2. wire one conservative compression policy
3. compare outputs in `examples.md` and `workflow_compare.csv`
4. only then try a more aggressive policy

## Notes

This repository is intentionally opinionated:

- vary compression **policy**, not model architecture
- optimize for **workflow decisions**, not paper completeness
- the safe and aggressive templates are stubs; wire [turboquant-core](https://github.com/Flaker420/turboquant-core) or your own backend to produce real compression comparisons
