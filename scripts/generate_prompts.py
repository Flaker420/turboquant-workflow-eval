#!/usr/bin/env python3
"""Generate long-context and multi-turn evaluation prompts using the target model.

This script loads the target model (e.g. Qwen3.5-9B) and uses it to generate
diverse long-context prompts programmatically.  The generated prompts are written
to a YAML file that follows the same schema as ``prompts/workflow_prompts.yaml``
and can be committed for reproducibility.

Usage::

    python scripts/generate_prompts.py \
        --model-config configs/model/qwen35_9b_text_only.yaml \
        --output prompts/generated_long_context.yaml \
        --max-new-tokens 2048

The script is idempotent: re-running with the same seed produces identical output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import yaml

# Ensure the package is importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turboquant_workflow_eval.loader import load_model_module
from turboquant_workflow_eval.schema import model_to_legacy_dict
from turboquant_workflow_eval.generation import render_prompt
from turboquant_workflow_eval.model_loader import load_model_and_tokenizer, infer_model_device

import torch

# ---------------------------------------------------------------------------
# Meta-prompt templates
# ---------------------------------------------------------------------------

# Each template asks the model to produce a structured JSON block that we parse.
# The ``{target_tokens}`` placeholder is replaced with the desired token range.

_META_PROMPTS: list[dict] = [
    # --- Long-context retrieval (4 prompts) ---
    {
        "id_prefix": "gen_retrieval",
        "category": "retrieval",
        "meta_prompt": textwrap.dedent("""\
            You are a test-data generator for an LLM evaluation harness.

            Generate a RETRIEVAL evaluation prompt that contains a long context passage
            (approximately {target_tokens} words) followed by a specific question that
            requires faithful extraction from the context.  The context should be a
            realistic technical document — for example, a project status report,
            infrastructure specification, experiment log, or API documentation.

            Return your answer as a single JSON object with these exact keys:
            - "title": a short descriptive title (3-6 words)
            - "watch_for": what a human reviewer should check (one sentence)
            - "prompt": the full evaluation prompt including the context and question

            Return ONLY the JSON object, no other text.
        """),
        "count": 4,
        "target_tokens_range": (300, 600),
    },
    # --- Long-context reasoning (3 prompts) ---
    {
        "id_prefix": "gen_reasoning",
        "category": "reasoning",
        "meta_prompt": textwrap.dedent("""\
            You are a test-data generator for an LLM evaluation harness.

            Generate a REASONING evaluation prompt that presents a detailed scenario
            (approximately {target_tokens} words) requiring multi-step analysis.
            The scenario should involve trade-offs, constraints, and a clear question
            at the end.  Good domains: system design decisions, resource allocation,
            risk assessment, engineering prioritization.

            Return your answer as a single JSON object with these exact keys:
            - "title": a short descriptive title (3-6 words)
            - "watch_for": what a human reviewer should check (one sentence)
            - "prompt": the full evaluation prompt including the scenario and question

            Return ONLY the JSON object, no other text.
        """),
        "count": 3,
        "target_tokens_range": (250, 500),
    },
    # --- Long-context coding (2 prompts) ---
    {
        "id_prefix": "gen_coding",
        "category": "coding",
        "meta_prompt": textwrap.dedent("""\
            You are a test-data generator for an LLM evaluation harness.

            Generate a CODING evaluation prompt that includes a substantial code listing
            (approximately {target_tokens} words / lines of Python code) and asks the
            reader to modify, debug, or extend it.  The code should be realistic — for
            example a data pipeline, an API client, a configuration parser, or a test
            harness.  Include specific instructions about what to change.

            Return your answer as a single JSON object with these exact keys:
            - "title": a short descriptive title (3-6 words)
            - "watch_for": what a human reviewer should check (one sentence)
            - "prompt": the full evaluation prompt including the code and task

            Return ONLY the JSON object, no other text.
        """),
        "count": 2,
        "target_tokens_range": (250, 400),
    },
    # --- Multi-turn conversations (3 prompts) ---
    {
        "id_prefix": "gen_multiturn",
        "category": "reasoning",
        "meta_prompt": textwrap.dedent("""\
            You are a test-data generator for an LLM evaluation harness.

            Generate a MULTI-TURN conversation evaluation prompt.  Produce a JSON object
            representing a 3-turn conversation where:
            - Turn 1 (user): asks an initial question or sets up a scenario ({target_tokens} words)
            - Turn 2 (assistant): provides a detailed response (you write this)
            - Turn 3 (user): asks a follow-up that requires remembering details from turns 1-2

            The conversation should test whether the model maintains context across turns.
            Good topics: iterative design review, debugging session, incremental specification.

            Return your answer as a single JSON object with these exact keys:
            - "title": a short descriptive title (3-6 words)
            - "watch_for": what a human reviewer should check (one sentence)
            - "turns": a list of objects, each with "role" ("user" or "assistant") and "content"

            Return ONLY the JSON object, no other text.
        """),
        "count": 3,
        "target_tokens_range": (150, 300),
    },
]


def _generate_text(model, tokenizer, prompt_text: str, max_new_tokens: int, device) -> str:
    """Run a single generation with the model."""
    rendered = render_prompt(tokenizer, prompt_text)
    encoded = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=4096)
    prompt_len = encoded["input_ids"].shape[-1]
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )

    output_ids = generated[0][prompt_len:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from the model's output."""
    # Try to find JSON block in code fence
    match = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


def _extract_json_with_turns(text: str) -> dict | None:
    """Extract JSON that may contain nested arrays (for multi-turn prompts)."""
    # More permissive: find outermost { ... }
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def generate_prompts(model_config_path: str, output_path: str, max_new_tokens: int = 2048) -> list[dict]:
    """Generate long-context evaluation prompts using the model."""
    model_cfg = model_to_legacy_dict(load_model_module(model_config_path))
    model, tokenizer, _ = load_model_and_tokenizer(model_cfg)
    device = infer_model_device(model)
    model.eval()

    # Warmup
    _generate_text(model, tokenizer, "Say hello.", 32, device)

    all_prompts: list[dict] = []
    prompt_counter = 0

    for spec in _META_PROMPTS:
        for i in range(spec["count"]):
            lo, hi = spec["target_tokens_range"]
            # Vary target tokens across prompts for diversity
            target = lo + (hi - lo) * i // max(spec["count"] - 1, 1)
            meta = spec["meta_prompt"].format(target_tokens=target)

            # Add diversity hint to avoid repetitive outputs
            meta += f"\n\nMake this prompt #{i + 1} of {spec['count']} — ensure it is distinct from others in the set."

            raw_output = _generate_text(model, tokenizer, meta, max_new_tokens, device)

            is_multiturn = "turns" in spec["meta_prompt"]
            parsed = _extract_json_with_turns(raw_output) if is_multiturn else _extract_json(raw_output)

            if parsed is None:
                print(f"WARNING: Failed to parse JSON for {spec['id_prefix']}_{i + 1:02d}. Raw output:")
                print(raw_output[:500])
                print("---")
                continue

            prompt_counter += 1
            entry: dict = {
                "id": f"{spec['id_prefix']}_{i + 1:02d}",
                "category": spec["category"],
                "title": parsed.get("title", f"generated {spec['category']} {i + 1}"),
                "watch_for": parsed.get("watch_for", "faithful and coherent response"),
            }

            if is_multiturn and "turns" in parsed:
                entry["turns"] = parsed["turns"]
                # For multi-turn, prompt field is empty (turns carry the content)
                entry["prompt"] = ""
            else:
                entry["prompt"] = parsed.get("prompt", "")

            # Validate: check that the generated prompt has substance
            content_length = len(entry.get("prompt", "")) + sum(
                len(t.get("content", "")) for t in entry.get("turns", [])
            )
            if content_length < 100:
                print(f"WARNING: {entry['id']} has very short content ({content_length} chars), skipping.")
                continue

            all_prompts.append(entry)
            print(f"Generated: {entry['id']} ({spec['category']}, ~{content_length} chars)")

    # Write YAML
    output = {"prompts": all_prompts}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"\nWrote {len(all_prompts)} prompts to {output_path}")
    return all_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate long-context evaluation prompts using the target model.")
    parser.add_argument("--model-config", required=True, help="Path to model config YAML")
    parser.add_argument("--output", default="prompts/generated_long_context.yaml", help="Output YAML path")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max tokens for generation")
    args = parser.parse_args()

    generate_prompts(args.model_config, args.output, args.max_new_tokens)


if __name__ == "__main__":
    main()
