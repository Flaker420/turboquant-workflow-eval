# RunPod setup notes

This repository is designed around the common RunPod pattern:

- Secure Cloud Pod
- network volume attached at deploy time
- default volume mount path `/workspace`
- PyTorch-oriented interactive workflow

## Suggested layout

- repository: `/workspace/<repo-name>`
- virtual environment: `/workspace/venvs/qwen35-turboquant-study`
- Hugging Face cache: `/workspace/.cache/huggingface`
- outputs: `/workspace/outputs`

## Automated setup

Install dependencies only:

```bash
bash scripts/bootstrap_runpod.sh
source .env.runpod
source /workspace/venvs/qwen35-turboquant-study/bin/activate
```

Install dependencies and warm the model cache:

```bash
bash scripts/bootstrap_runpod.sh --download-model
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
