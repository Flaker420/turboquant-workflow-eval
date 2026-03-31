# Study scope

This repository is designed for a **compact engineering study**, not a research-paper ablation.

## Primary question

Which policy is usable in my workflow on Qwen3.5-9B, and what degrades when I push harder?

## Non-goals

Not in scope for the first pass:

- architecture redesign
- exhaustive paper-style ablations
- full theoretical validation of QJL
- maximum-throughput serving optimization
- publication-grade benchmarking breadth

## Deliverables

A successful run should produce:

- `workflow_compare.csv`
- `rows.jsonl`
- `examples.md`
- `run_summary.json`
- per-prompt text files

These are meant to be read side by side when deciding:

- green: good enough for workflow use
- yellow: usable with caution
- red: too lossy or unstable
