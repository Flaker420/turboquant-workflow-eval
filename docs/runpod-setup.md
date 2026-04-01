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
- virtual environment: `/workspace/venvs/qwen35-turboquant-study`
- Hugging Face cache: `/workspace/.cache/huggingface`
- outputs: `/workspace/outputs`

## Automated setup

Base environment only:

```bash
bash scripts/bootstrap_runpod.sh
source .env.runpod
source /workspace/venvs/qwen35-turboquant-study/bin/activate
```

Base environment + model cache warmup:

```bash
bash scripts/bootstrap_runpod.sh --download-model
source .env.runpod
source /workspace/venvs/qwen35-turboquant-study/bin/activate
```

Base environment + optional fast-path package attempt:

```bash
bash scripts/bootstrap_runpod.sh --download-model --fast-path
source .env.runpod
source /workspace/venvs/qwen35-turboquant-study/bin/activate
```

## Manual setup

The detailed manual sequence is in `docs/manual-runbook.md`.

## Why the scripts assume `/workspace`

Using the network volume as the working root means:

- models do not need to be re-downloaded after Pod restarts
- outputs persist independently from the container disk
- the environment is easy to recreate on another Pod in the same volume-backed workflow
