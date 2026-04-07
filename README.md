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
| Qwen3.5-9B | `configs/model/qwen35_9b_text_only.py` | `Qwen35KVBackend` |
| Qwen3-8B | `configs/model/qwen3_8b.py` | `Qwen3DenseKVBackend` |
| Qwen2.5-3B-Instruct | `configs/model/qwen25_3b.py` | `Qwen25DenseKVBackend` |

All three models work with the same evaluation pipeline. Select the model by passing a different `--model-config` to the scripts or by editing the study/experiment config. Bundled study configs: `configs/studies/default_qwen35_9b.py` (Qwen3.5-9B), `configs/studies/default_qwen3_8b.py`, `configs/studies/default_qwen25_3b.py`.

## Ready-to-run status

The scaffold is **ready to run** for:
- RunPod environment setup with the validated cu128 torch stack
- dependency installation
- explicit model/tokenizer cache warmup
- preflight instrumentation
- baseline workflow study

It includes a **pluggable adapter interface** and ready-to-use template configs (`safe_template.py`, `aggressive_template.py`) that use the built-in `TurboQuantAdapter` to delegate to [turboquant-core](https://github.com/Flaker420/turboquant-core). Both templates are enabled by default.

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

`turboquant-core` is **vendored in-tree** at `vendor/turboquant-core/` (added via `git subtree`). It is no longer a pip dependency — `pyproject.toml` adds `vendor/turboquant-core/src` to `pythonpath` and `vendor/turboquant-core/tests` to `testpaths`, so `from turboquant_core import …` and `pytest` both pick up the vendored copy automatically. Update it with `git pull` like any other in-tree code; there is no SHA pin to bump. The standalone `Flaker420/turboquant-core` repo is archived read-only.

The built-in `TurboQuantAdapter` (`src/turboquant_workflow_eval/adapters/turboquant.py`) wraps the core library's adapter, handling the `model_name` → `name` field normalization between the eval harness and the core library, and validating that `model_name` is set on the model config (raises `ValueError` early if not).

Both `safe_template.py` (bit_width 4) and `aggressive_template.py` (bit_width 2) are pre-configured and enabled. They forward `bit_width`, `seed`, `residual_window`, and `key_strategy` from `settings:` straight into turboquant-core; all four are recorded in `describe()` and on every result row, so they are visible in `run_summary.json` and the per-row CSV/JSONL outputs. See `docs/adapter-interface.md` for the adapter contract if you need to write a custom adapter.

## Study configuration

Study, model, and policy configs are **Python modules** under `configs/studies/`, `configs/model/`, and `configs/policies/`. Each module exports a single frozen `@dataclass` instance (`STUDY`, `MODEL`, or `POLICY`) defined in `src/turboquant_workflow_eval/schema.py`. The loader (`src/turboquant_workflow_eval/loader.py`) imports the file via `importlib.util` and returns the dataclass directly — there is no YAML in the configuration path.

Field validation lives in `__post_init__`: invalid `dtype`, missing `model_name`, out-of-range `bit_width`, malformed `import_path`, etc. all raise `ConfigValidationError` at construction time. A multi-policy `StudyConfig` **must** declare `baseline_policy_name=`; single-policy studies auto-set it to that policy's name. All bundled studies declare `baseline_policy_name="baseline"`.

The CLI (`python -m turboquant_workflow_eval`) loads a study module via `--study configs/studies/default_qwen35_9b.py` and exposes every dataclass field as a typed argparse flag. See `--help` for the full list, or `## CLI quick reference` below.

Authoring a new study is just writing Python:

```python
# configs/studies/my_run.py
from pathlib import Path
from turboquant_workflow_eval.loader import load_model_module, load_policy_module
from turboquant_workflow_eval.schema import StudyConfig, RuntimeConfig, ThresholdsConfig

_HERE = Path(__file__).resolve().parent

STUDY = StudyConfig(
    name="my_run",
    model=load_model_module(_HERE.parent / "model" / "qwen25_3b.py"),
    prompt_pack=(_HERE.parent.parent / "prompts" / "workflow_prompts.yaml",),
    policies=(
        load_policy_module(_HERE.parent / "policies" / "baseline.py"),
        load_policy_module(_HERE.parent / "policies" / "safe_template.py"),
    ),
    baseline_policy_name="baseline",
    runtime=RuntimeConfig(max_input_tokens=2048, max_new_tokens=256),
    thresholds=ThresholdsConfig(latency_red_pct=50.0),
)
```

> **Note:** prompt packs (`prompts/*.yaml`) remain YAML — they are content, not configuration, and live in a separate location from the dataclass-based configs. Only the configuration tree (`configs/{model,policies,studies,experiments}/`) was migrated.

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
bash scripts/bootstrap_runpod.sh --download-model --model-config configs/model/qwen3_8b.py
```

Or Qwen2.5-3B-Instruct:

```bash
bash scripts/bootstrap_runpod.sh --download-model --model-config configs/model/qwen25_3b.py
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
python -m turboquant_workflow_eval --study configs/studies/default_qwen35_9b.py --dry-run
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
  --study configs/studies/default_qwen35_9b.py \
  --single --prompt-id math_01
```

### 7) Run the workflow study

Baseline only:

```bash
make study POLICY_CONFIGS=configs/policies/baseline.py OUTPUT_DIR=outputs/study_baseline
```

Baseline plus the built-in safe policy (after wiring a real backend):

```bash
make study \
  POLICY_CONFIGS=configs/policies/baseline.py,configs/policies/safe_template.py \
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
  [--study PATH] [--policies PATH1,PATH2,...] [--output-dir PATH] \
  [--model-config PATH] \
  [--set KEY=VALUE] [--repetitions N] \
  [--prompt-id ID] [--prompt-category CAT] [--prompt-filter REGEX] \
  [--single] [--dry-run]
```

### `scripts/generate_prompts.py`

```bash
python scripts/generate_prompts.py [--model-config PATH] [--output PATH] [--max-new-tokens N]
```

Generates long-context evaluation prompts using the target model. The output YAML follows the same schema as `prompts/workflow_prompts.yaml` and can be used with `configs/studies/full.py`.

## CLI quick reference

The `python -m turboquant_workflow_eval` entry point (and `scripts/run_workflow_study.py`) loads a Python study module and exposes every dataclass knob as a typed argparse flag. The tables below are auto-generated from the argparse parser by `scripts/generate_cli_docs.py`; run that script (or `make cli-docs`) after adding a flag.

<!-- cli-docs:start -->
<!-- Auto-generated by scripts/generate_cli_docs.py. Do not edit by hand. -->

### options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--study STUDY` | str | — | Path to a Python study module (e.g. configs/studies/default_qwen35_9b.py) or a 'package.module:STUDY' import path. Required unless --rescore. |
| `--output-dir OUTPUT-DIR` | str | `'outputs'` | Directory for result artefacts (default: outputs/) |
| `--model MODEL` | str | — | Override the study's model with a different model module path. |
| `--policies POLICIES` | str | — | Comma-separated list of policy module paths that overrides study.policies. |
| `--single` | flag | False | Quick smoke test: first matching prompt, first enabled policy, 1 repetition |
| `--dry-run` | flag | False | Validate all configs and print execution plan without touching the GPU |
| `--rescore ROWS_JSONL` | str | — | Recompute divergence + KV-cache compression metrics over an existing rows.jsonl (no GPU; requires output_token_ids in the file). |
| `--prompt-id PROMPT-IDS` | str (repeatable) | — | Run only the specified prompt ID (repeatable) |
| `--prompt-category PROMPT-CATEGORIES` | str (repeatable) | — | Run only prompts in the specified category (repeatable) |
| `--prompt-filter REGEX` | str | — | Filter prompts by regex on id/title |
| `--set DOT.PATH=VALUE` | str (repeatable) | — | Escape hatch: override any StudyConfig field via dot-path (e.g. --set runtime.max_new_tokens=128). Repeatable. Applied via dataclasses.replace recursively. |
| `--set-policy NAME.DOT.KEY=VALUE` | str (repeatable) | — | Escape hatch: per-policy dot-path override. First segment is a policy name (or '*' for every policy). Examples: --set-policy turboquant_safe.settings.bit_width=8, --set-policy '*.settings.key_strategy=mse', --set-policy baseline.enabled=false. Repeatable. |

### global policy settings (apply to every policy)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--bit-width BIT-WIDTH` | int | — | Override PolicySettings.bit_width on every policy. |
| `--seed SEED` | int | — | Override PolicySettings.seed on every policy. |
| `--residual-window RESIDUAL-WINDOW` | int | — | Override PolicySettings.residual_window on every policy. |
| `--key-strategy KEY-STRATEGY` | {mse,mse+qjl} | — | Override PolicySettings.key_strategy on every policy. |
| `--value-strategy VALUE-STRATEGY` | {mse} | — | Override PolicySettings.value_strategy on every policy. |
| `--compressible-layers 3,7,11,...` | parse_int_list | — | Override PolicySettings.compressible_layers on every policy (comma-separated integer indices). |
| `--compressible-heads 0,2,...` | parse_int_list | — | Override PolicySettings.compressible_heads on every policy (comma-separated KV-head indices). Runtime wiring is pending: supported by Qwen*KVBackend directly; prepare_model raises NotImplementedError when set. |
| `--profile PROFILE` | str | — | Override PolicySettings.profile on every policy. |

### per-policy overrides (NAME=VALUE, repeatable)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--bit-width-for NAME=VALUE` | str | — | Override PolicySettings.bit_width on a single policy by name. |
| `--seed-for NAME=VALUE` | str | — | Override PolicySettings.seed on a single policy by name. |
| `--residual-window-for NAME=VALUE` | str | — | Override PolicySettings.residual_window on a single policy by name. |
| `--key-strategy-for NAME=VALUE` | str | — | Override PolicySettings.key_strategy on a single policy by name. |
| `--value-strategy-for NAME=VALUE` | str | — | Override PolicySettings.value_strategy on a single policy by name. |
| `--compressible-layers-for NAME=VALUE` | str | — | Override PolicySettings.compressible_layers on a single policy by name. |
| `--compressible-heads-for NAME=VALUE` | str | — | Override PolicySettings.compressible_heads on a single policy by name. |
| `--profile-for NAME=VALUE` | str | — | Override PolicySettings.profile on a single policy by name. |

### runtime

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-input-tokens MAX-INPUT-TOKENS` | int | — | Override runtime.max_input_tokens. |
| `--max-new-tokens MAX-NEW-TOKENS` | int | — | Override runtime.max_new_tokens. |
| `--temperature TEMPERATURE` | float | — | Override runtime.temperature. |
| `--top-p TOP-P` | float | — | Override runtime.top_p. |
| `--repetitions REPETITIONS` | int | — | Override runtime.repetitions. |
| `--no-cache` | flag | False | Set runtime.use_cache=False. |
| `--shuffle-policies` | flag | False | Set runtime.shuffle_policies=True. |
| `--shuffle-seed SHUFFLE-SEED` | int | — | Override runtime.shuffle_seed. |
| `--baseline-policy BASELINE-POLICY` | str | — | Override study.baseline_policy_name. |

### early stop

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-error-rate MAX-ERROR-RATE` | float | — | Override early_stop.max_error_rate. |
<!-- cli-docs:end -->

### Common examples

```bash
# Validate before committing GPU hours
python -m turboquant_workflow_eval --study configs/studies/default_qwen35_9b.py --dry-run

# Quick smoke test with one math prompt
python -m turboquant_workflow_eval --study configs/studies/default_qwen35_9b.py \
  --single --prompt-id math_01

# Override max tokens and repetitions from CLI
python -m turboquant_workflow_eval --study configs/studies/default_qwen35_9b.py \
  --max-new-tokens 64 --repetitions 5

# Force every policy to use plain MSE (no QJL)
python -m turboquant_workflow_eval --study configs/studies/default_qwen35_9b.py \
  --key-strategy mse

# Compress only a subset of GatedAttn layers in the safe policy
python -m turboquant_workflow_eval --study configs/studies/default_qwen35_9b.py \
  --compressible-layers-for turboquant_safe=7,15,23,31

# Run only coding prompts
python -m turboquant_workflow_eval --study configs/studies/default_qwen35_9b.py \
  --prompt-category coding

# Re-score existing results with looser latency threshold
python -m turboquant_workflow_eval --rescore outputs/study_run/rows.jsonl \
  --latency-red-pct 50
```

> **Note:** the Gradio UI is **temporarily broken** between this PR and the UI rework PR. `app.py` still imports the deleted YAML loaders; launching it will fail at import. Use the CLI in the meantime.

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

Results can be re-scored with different thresholds using `--rescore` without re-running inference. `--rescore` accepts an optional `--study` argument that supplies the study's `ThresholdsConfig` as the base, then layers the dedicated threshold flags (e.g. `--latency-red-pct 50`, `--similarity-red 0.75`) on top. The legacy `--set thresholds.latency_red_pct=50` form is still accepted as an escape hatch. The baseline policy is taken from the sibling `run_summary.json` next to the rows JSONL, falling back to single-policy auto-detection. A refreshed `run_summary.json` is always written next to the rescored rows with `rescored: true`, `rescore_thresholds`, and `rescore_verdicts_changed` fields recording exactly what was applied.

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
