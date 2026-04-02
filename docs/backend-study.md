# Backend study path

This document describes the server-side backend evaluation path for the repository.

## Goal

Compare a real serving backend against the local baseline and local safe proxy using the same prompt packs and the same report artifacts.

## Current backend

The repository includes an OpenAI-compatible backend client:

- `src/qwen35_turboquant_workflow_study/backends/openai_compatible.py`
- `configs/backends/qwen35_openai_server.yaml`
- `scripts/probe_openai_backend.py`
- `scripts/run_backend_study.py`

## Review point

The backend layer is ready for review after:

1. the probe script returns a real generation response
2. the backend study runner writes the standard artifacts
3. the long-context prompt pack is exercised successfully

## Standard artifacts

The backend study writes the same outputs as the local workflow study:

- `run_summary.json`
- `workflow_compare.csv`
- `rows.jsonl`
- `examples.md`
- per-prompt text outputs

## Long-context prompt pack

Use `prompts/long_context_workflow_prompts.yaml` when the goal is to expose KV pressure and compare long-context behavior.
