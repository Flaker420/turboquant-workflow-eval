# TurboQuant Workflow Evaluation

A practical, **RunPod-first** evaluation harness for answering a simple question:

> Which compression policy is usable in my workflow, and what breaks when I push it harder?

This repository is the evaluation counterpart to [turboquant-core](https://github.com/Flaker420/turboquant-core), the TurboQuant KV-cache compression library. It is **not** a paper-style ablation project. It is a compact harness with:

- model discovery and preflight instrumentation
- a fixed prompt pack for reasoning, math, coding, and retrieval
- reproducible CSV / JSONL / Markdown outputs
- a pluggable adapter interface for integrating TurboQuant backends
- a Gradio web UI for interactive workflow execution
- RunPod-oriented bootstrap and storage conventions

## Supported models

| Model | Config | turboquant-core backend |
|-------|--------|------------------------|
| Qwen3.5-9B | `configs/model/qwen35_9b_text_only.yaml` | `Qwen35KVBackend` |
| Qwen3-8B | `configs/model/qwen3_8b.yaml` | `Qwen3DenseKVBackend` |
| Qwen2.5-3B-Instruct | `configs/model/qwen25_3b.yaml` | `Qwen25DenseKVBackend` |

All three models work with the same evaluation pipeline. Select the model by passing a different `--model-config` to the scripts or by editing the study/experiment config. Bundled study configs: `configs/studies/default.yaml` (Qwen3.5-9B), `configs/studies/default_qwen3_8b.yaml`, `configs/studies/default_qwen25_3b.yaml`.

## Ready-to-run status

The scaffold is **ready to run** for:
- RunPod environment setup with the validated cu128 torch stack
- dependency installation
- explicit model/tokenizer cache warmup
- preflight instrumentation
- baseline workflow study

It includes a **pluggable adapter interface** and ready-to-use template configs (`safe_template.yaml`, `aggressive_template.yaml`) that use the built-in `TurboQuantAdapter` to delegate to [turboquant-core](https://github.com/Flaker420/turboquant-core). Both templates are enabled by default.

## What is built in

Built in and ready to run:

- RunPod bootstrap with optional cache warmup
- environment validation
- model loading helpers (any HuggingFace model via config)
- language-model root discovery
- attention-block discovery
- Q / K / V projection capture for preflight checks
- baseline pass-through adapter with revert support for model reuse
- composable study runner (`prepare_study` / `run_policy` / `score_results` / `write_results`)
- automated quality scoring (math reference checking, code execution, semantic similarity)
- configurable green/yellow/red verdict system with per-prompt and per-category verdicts
- repetition support with mean/std aggregation for stable benchmarking
- prompt generation script for long-context evaluation prompts
- Gradio web UI (`app.py`) with 8 tabs for interactive workflow execution
- `--dry-run` validation without GPU (validates configs, paths, adapters in <1 second)
- `--single` smoke-test mode (one prompt, one policy, one repetition)
- `--set key=value` CLI config overrides with dot-notation (e.g. `--set runtime.max_new_tokens=128`)
- prompt filtering from CLI (`--prompt-id`, `--prompt-category`, `--prompt-filter`)
- `--rescore` existing results with new thresholds (no GPU, pure computation)
- event-driven study loop with pause/resume/stop and configurable early stopping
- incremental JSONL output (partial results survive crashes)
- environment variable expansion in YAML configs (`${VAR:-default}`)
- unit tests for the scaffold

## What is wired

[turboquant-core](https://github.com/Flaker420/turboquant-core) is installed automatically via `requirements.txt`. What is included:

- a pass-through baseline adapter (runs the model with no compression)
- a `TurboQuantAdapter` that bridges turboquant-core to the eval harness
- safe and aggressive policy templates (both enabled, pointing to `TurboQuantAdapter`)

To run a compression comparison, no additional wiring is needed -- just run the study with the desired policy configs. See `docs/adapter-interface.md` for the adapter contract if you want to write a custom adapter.

## turboquant-core integration

[turboquant-core](https://github.com/Flaker420/turboquant-core) is listed in `requirements.txt` and installed automatically. The built-in `TurboQuantAdapter` (`src/turboquant_workflow_eval/adapters/turboquant.py`) wraps the core library's adapter, handling the `model_name` → `name` field normalization between the eval harness and the core library, and validating that `model_name` is set on the model config (raises `ValueError` early if not).

The dependency is pinned to a specific commit SHA in both `requirements.txt` and `pyproject.toml`. To bump the pin, replace the SHA after `@` in both files with the new core commit and re-run `pip install -e .`. The same procedure applies on RunPod (see `docs/manual-runbook.md` step 1).

Both `safe_template.yaml` (bit_width 4) and `aggressive_template.yaml` (bit_width 2) are pre-configured and enabled. They forward `bit_width`, `seed`, `residual_window`, and `key_strategy` from `settings:` straight into turboquant-core; all four are recorded in `describe()` and on every result row, so they are visible in `run_summary.json` and the per-row CSV/JSONL outputs. See `docs/adapter-interface.md` for the adapter contract if you need to write a custom adapter.

## Study configuration

Study YAMLs (`configs/studies/*.yaml`) collect the model config, prompt pack, policy list, runtime parameters, and thresholds. A multi-policy study **must** declare `baseline_policy_name:` -- this names the policy whose rows are used as the comparison baseline for similarity, latency delta, and verdicts. `validate_study_config` raises a `ConfigValidationError` at load time if the field is missing on a multi-policy run. All bundled studies declare `baseline_policy_name: baseline`.

Single-policy studies do not require the field; the only present policy is used as its own baseline (this is just provenance, since there is nothing to compare against).

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
- native Qwen3.5 non-thinking toggle enabled through local `apply_chat_template(..., enable_thinking=False)` (the harness probes for tokenizer support and falls back gracefully on tokenizers that do not accept the kwarg)

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

Fast-path packages (flash-linear-attention, causal-conv1d) are now installed by default. To skip them:

```bash
bash scripts/bootstrap_runpod.sh --download-model --no-fast-path
source .env.runpod
source /workspace/venvs/turboquant-eval/bin/activate
```

To use Qwen3-8B instead of the default Qwen3.5-9B:

```bash
bash scripts/bootstrap_runpod.sh --download-model --model-config configs/model/qwen3_8b.yaml
```

Or Qwen2.5-3B-Instruct:

```bash
bash scripts/bootstrap_runpod.sh --download-model --model-config configs/model/qwen25_3b.yaml
```

Alternatively, after bootstrapping, launch the web UI to run the entire workflow from your browser:

```bash
make ui
```

### 2) Validate the scaffold

```bash
python scripts/validate_environment.py
pytest -q
```

### 3) Dry-run to validate configs (no GPU)

Before committing GPU hours, verify that all configs, paths, and adapters resolve correctly:

```bash
python -m turboquant_workflow_eval --study-config configs/studies/default.yaml --dry-run
```

This runs in <1 second and prints an execution plan showing how many prompts, policies, and total generations would run.

### 4) Inspect the discovered attention blocks

```bash
make list-attention
```

### 5) Run the preflight instrumentation pass

```bash
make preflight
```

### 6) Smoke-test a single prompt (optional)

Before running the full matrix, test one prompt end-to-end:

```bash
make smoke-test
```

Or with the CLI directly:

```bash
python -m turboquant_workflow_eval \
  --study-config configs/studies/default.yaml \
  --single --prompt-id math_01
```

### 7) Run the workflow study

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

## Web UI (Gradio)

The repository includes a browser-based Gradio UI that covers the full workflow. This is especially useful on RunPod, where you can access it through the pod's exposed port.

### Launching the UI

```bash
make ui
```

Or directly:

```bash
python app.py
```

The UI starts on `http://0.0.0.0:7860`. On RunPod, access it through your pod's proxy URL at port 7860.

### UI tabs

| Tab | Purpose |
|-----|---------|
| **Environment** | Validate CUDA/torch setup, check HuggingFace cache, download models |
| **Model Inspection** | Load a model into memory, discover and inspect attention blocks |
| **Preflight** | Run Q/K/V tensor statistics on loaded model, view results as JSON |
| **Study Runner** | Select study and policy configs, filter prompts, run study with pause/resume/stop controls, live verdict summary |
| **Results** | Browse completed study outputs: comparison table, per-prompt text, run metadata |
| **Quick Test** | Run a single prompt with parameter sliders (max tokens, temperature, repetitions), see instant results |
| **Re-Score** | Adjust verdict thresholds with sliders and re-score existing results instantly (no GPU) |
| **Comparison** | Side-by-side diff of two study runs with verdict highlighting |

The UI calls the same Python library functions as the CLI scripts. A model loaded in the Model Inspection tab stays in memory and is reused by the Preflight and Quick Test tabs.

## Manual step-by-step path

The manual runbook is in:

- `docs/manual-runbook.md`

That document includes each step separately plus the parameters for each script.

## Script usage summary

### `app.py` (Gradio UI)

```bash
python app.py
```

Launches the web UI on port 7860. See the [Web UI](#web-ui-gradio) section above.

### `scripts/bootstrap_runpod.sh`

```bash
bash scripts/bootstrap_runpod.sh [--download-model] [--download-all] [--tokenizer-only] [--no-fast-path] [--model-config PATH]
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
python scripts/run_workflow_study.py \
  [--study-config PATH] [--policy-configs PATH1,PATH2,...] [--output-dir PATH] \
  [--model-config PATH] \
  [--set KEY=VALUE] [--repetitions N] \
  [--prompt-id ID] [--prompt-category CAT] [--prompt-filter REGEX] \
  [--single] [--dry-run]
```

### `scripts/generate_prompts.py`

```bash
python scripts/generate_prompts.py [--model-config PATH] [--output PATH] [--max-new-tokens N]
```

Generates long-context evaluation prompts using the target model. The output YAML follows the same schema as `prompts/workflow_prompts.yaml` and can be used with `configs/studies/full.yaml`.

## CLI quick reference

The `python -m turboquant_workflow_eval` entry point (and `scripts/run_workflow_study.py`) supports these flags:

| Flag | Description |
|------|-------------|
| `--study-config PATH` | Path to study YAML. Required for normal and `--dry-run` modes; **optional in `--rescore` mode**, where it serves as a thresholds source. |
| `--output-dir PATH` | Output directory (default: `outputs/`) |
| `--policy-configs P1,P2` | Override policy configs from study YAML |
| `--model-config PATH` | Override model config from study YAML |
| `--set KEY=VALUE` | Override any **study** config value with dot-notation (repeatable). In `--rescore` mode, both `thresholds.latency_red_pct=50` and the bare `latency_red_pct=50` forms are accepted. |
| `--set-policy NAME.KEY=VALUE` | Override a key inside a **policy** YAML at load time (repeatable). Format: `<policy_name\|*>.<dot.key>=<value>`. Examples: `--set-policy turboquant_safe.settings.key_strategy=mse`, `--set-policy '*.settings.bit_width=8'`, `--set-policy baseline.enabled=false`. |
| `--repetitions N` | Shorthand for `--set runtime.repetitions=N` |
| `--prompt-id ID` | Run only this prompt (repeatable for multiple IDs) |
| `--prompt-category CAT` | Run only prompts in this category (repeatable) |
| `--prompt-filter REGEX` | Filter prompts by regex on id/title |
| `--single` | Quick smoke test: 1st prompt, 1st policy, 1 rep |
| `--dry-run` | Validate all configs without GPU (runs in <1s) |
| `--rescore ROWS_JSONL` | Re-score existing results with new thresholds (no GPU). Reads `baseline_policy_name` from a sibling `run_summary.json` if present, and writes a refreshed `run_summary.json` with `rescored: true`, `rescore_thresholds`, and `rescore_verdicts_changed`. |

### Common examples

```bash
# Validate before committing GPU hours
python -m turboquant_workflow_eval --study-config configs/studies/default.yaml --dry-run

# Quick smoke test with one math prompt
python -m turboquant_workflow_eval --study-config configs/studies/default.yaml \
  --single --prompt-id math_01

# Override max tokens and repetitions from CLI
python -m turboquant_workflow_eval --study-config configs/studies/default.yaml \
  --set runtime.max_new_tokens=64 --repetitions 5

# Run only coding prompts
python -m turboquant_workflow_eval --study-config configs/studies/default.yaml \
  --prompt-category coding

# Re-score existing results with looser latency threshold
python -m turboquant_workflow_eval \
  --rescore outputs/study_run/rows.jsonl \
  --set thresholds.latency_red_pct=50

# Force every policy to use plain MSE (no QJL) without editing YAML
python -m turboquant_workflow_eval --study-config configs/studies/default.yaml \
  --set-policy '*.settings.key_strategy=mse'

# Tweak only the safe policy: 8-bit, MSE-only
python -m turboquant_workflow_eval --study-config configs/studies/default.yaml \
  --set-policy turboquant_safe.settings.bit_width=8 \
  --set-policy turboquant_safe.settings.key_strategy=mse
```

The Gradio UI's **Study Runner** tab has a matching **Policy Overrides** accordion: paste one override per line and they are applied to every loaded policy YAML before the run.

## Outputs

For each policy you test, the repo produces concrete artifacts you can compare:

- `workflow_compare.csv`
- `rows.jsonl` (written incrementally -- partial results survive crashes)
- `examples.md`
- per-prompt text files in `text_outputs/`
- `run_summary.json`

That gives you a direct **works / degrades / fails** view instead of a research-heavy sweep.

### Row schema

Every row carries the same set of fields whether the prompt succeeded or failed. A successful row has `verdict ∈ {green, yellow, red}` plus `output_text`, `output_tokens`, `latency_s`, `tokens_per_second`, `peak_vram_gb`, `semantic_similarity`, `output_length_delta_pct`, and (for code prompts) `code_verdict` / `code_passed` / `code_failed` / `code_errors`. A failed row has `error` set, `verdict == "error"`, `code_verdict == "error"`, `output_tokens == 0`, and `semantic_similarity == None`. Filter on `verdict == "error"` to find failed prompts; `score_results` skips them when computing baseline deltas, and refuses to use an error row as the baseline reference for a multi-policy study.

### Live verdicts and early stop

`run_policy` populates a shared baseline lookup as the baseline policy executes and stamps a real `verdict` (plus `semantic_similarity` and `output_length_delta_pct`) on every row before emitting the `prompt_completed` event. This means the early-stop controller can react to red verdicts as they happen instead of only after the full study finishes. Two early-stop knobs are read from `early_stop:` in the study YAML:

```yaml
early_stop:
  max_red_verdicts: 5      # stop after this many red verdicts across all policies
  max_error_rate: 0.5      # stop if errored / total exceeds this fraction
```

`score_results` runs at the end of the study and re-stamps the same fields idempotently, so the final outputs match what the live event stream reported.

### Re-scoring

Results can be re-scored with different thresholds using `--rescore` without re-running inference. `--rescore` accepts an optional `--study-config` that supplies a `thresholds:` block as the base, then layers `--set thresholds.<key>=<value>` (or the bare `--set <key>=<value>`) overrides on top. The baseline policy is taken from the sibling `run_summary.json` next to the rows JSONL, falling back to single-policy auto-detection. A refreshed `run_summary.json` is always written next to the rescored rows with `rescored: true`, `rescore_thresholds`, and `rescore_verdicts_changed` fields recording exactly what was applied.

## Repository layout

- `app.py` -- Gradio web UI (launch with `make ui`)
- `configs/` -- model, policy, and study configs
- `docs/` -- architecture facts, scope, RunPod setup, manual runbook, adapter contract, data-workflow review, and current RunPod state
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
- the safe and aggressive templates use the built-in `TurboQuantAdapter` backed by [turboquant-core](https://github.com/Flaker420/turboquant-core); write a custom adapter class if you need a different backend
