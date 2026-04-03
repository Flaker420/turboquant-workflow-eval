from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant_workflow_eval.config import load_yaml
from turboquant_workflow_eval.model_loader import load_model_and_tokenizer, resolve_language_model_root
from turboquant_workflow_eval.module_discovery import discover_attention_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="List discovered attention blocks for the configured model.")
    parser.add_argument("--model-config", default="configs/model/qwen35_9b_text_only.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)
    model, _tokenizer, loader_name = load_model_and_tokenizer(model_cfg)
    lm_root = resolve_language_model_root(model)
    blocks = discover_attention_blocks(lm_root, expected_count=model_cfg["layout"]["attention_blocks"])

    payload = {
        "model_name": model_cfg["model_name"],
        "loader": loader_name,
        "language_model_root": lm_root.__class__.__name__,
        "attention_blocks": [block.to_dict() for block in blocks],
    }

    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
