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
from turboquant_workflow_eval.model_loader import load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Warm the Hugging Face cache by downloading the configured model and tokenizer.",
    )
    parser.add_argument("--model-config", default="configs/model/qwen35_9b_text_only.yaml")
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Download only the tokenizer/config side and skip full model weights.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write a small JSON summary.",
    )
    args = parser.parse_args()

    model_cfg = load_yaml(args.model_config)

    if args.tokenizer_only:
        import transformers

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_cfg["model_name"],
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )
        summary = {
            "model_name": model_cfg["model_name"],
            "downloaded": "tokenizer_only",
            "tokenizer_class": tokenizer.__class__.__name__,
        }
    else:
        model, tokenizer, loader_name = load_model_and_tokenizer(model_cfg)
        summary = {
            "model_name": model_cfg["model_name"],
            "downloaded": "model_and_tokenizer",
            "loader_name": loader_name,
            "model_class": model.__class__.__name__,
            "tokenizer_class": tokenizer.__class__.__name__,
        }

    text = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
