# RunPod setup notes

This repository is designed around the common RunPod pattern:

- Secure Cloud Pod
- network volume attached at deploy time
- default volume mount path `/workspace`
- PyTorch-oriented interactive workflow

## Validated environment

Current validated working state on RunPod:

- `nvcc` / system toolkit: CUDA 12.8
- `torch==2.10.0+cu128`
- `triton==3.6.0`
- optional fast-path packages may install successfully while Qwen still prints a stale warning

Use the bootstrap script or the manual runbook exactly as written so the torch build matches the system CUDA toolkit.

## Suggested layout

- repository: `/workspace/<repo-name>`
- virtual environment: `/workspace/venvs/turboquant-eval`
- Hugging Face cache: `/workspace/.cache/huggingface`
- outputs: `/workspace/outputs`

## Automated setup

Base environment only:

```bash
bash scripts/bootstrap_runpod.sh
source .env.runpod
source /workspace/venvs/turboquant-eval/bin/activate
```

Base environment + model cache warmup:

```bash
bash scripts/bootstrap_runpod.sh --download-model
source .env.runpod
source /workspace/venvs/turboquant-eval/bin/activate
```

Fast-path packages (flash-linear-attention, causal-conv1d) are installed by default. To skip them:

```bash
bash scripts/bootstrap_runpod.sh --download-model --no-fast-path
source .env.runpod
source /workspace/venvs/turboquant-eval/bin/activate
```

## Web UI

After bootstrapping, launch the Gradio web UI for browser-based interaction:

```bash
make ui
```

The UI starts on port 7860. Access it through your pod's proxy URL (e.g., `https://<pod-id>-7860.proxy.runpod.net`). It covers environment validation, model loading, preflight instrumentation, study execution, and results exploration.

## Manual setup

The detailed manual sequence is in `docs/manual-runbook.md`.

## Why the scripts assume `/workspace`

Using the network volume as the working root means:

- models do not need to be re-downloaded after Pod restarts
- outputs persist independently from the container disk
- the environment is easy to recreate on another Pod in the same volume-backed workflow
