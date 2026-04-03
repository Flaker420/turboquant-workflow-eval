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
from turboquant_workflow_eval.download import (
    check_cache_status,
    discover_model_configs,
    download_all,
    download_one,
    format_summary_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download models and tokenizers into the HuggingFace cache.",
    )
    parser.add_argument("--model-config", default="configs/model/qwen35_9b_text_only.yaml",
                        help="Path to a single model config YAML.")
    parser.add_argument("--all", action="store_true", dest="download_all",
                        help="Download all models found in configs/model/.")
    parser.add_argument("--check-only", action="store_true",
                        help="Check cache status without downloading.")
    parser.add_argument("--tokenizer-only", action="store_true",
                        help="Download only tokenizer/config, skip full model weights.")
    parser.add_argument("--skip-cached", action="store_true", default=True,
                        help="Skip models already in the HuggingFace cache (default: true).")
    parser.add_argument("--no-skip-cached", action="store_false", dest="skip_cached",
                        help="Force re-download even if cached.")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Number of retry attempts on failure (default: 3).")
    parser.add_argument("--fallback-tokenizer-only", action="store_true", default=True,
                        help="On full-model failure, fall back to tokenizer-only (default: true).")
    parser.add_argument("--no-fallback", action="store_false", dest="fallback_tokenizer_only",
                        help="Do not fall back to tokenizer-only on failure.")
    parser.add_argument("--output", default=None,
                        help="Optional path to write a JSON summary.")
    args = parser.parse_args()

    if args.check_only:
        configs = discover_model_configs()
        results = []
        for cfg in configs:
            model_name = cfg["model_name"]
            cache = check_cache_status(model_name)
            results.append({
                "model_name": model_name,
                "config_path": cfg.get("_config_path", "unknown"),
                **cache,
            })
        print(f"\n{'Model':<30} {'Model cached':<15} {'Tokenizer cached'}")
        print(f"{'-' * 30} {'-' * 15} {'-' * 17}")
        for r in results:
            print(f"{r['model_name']:<30} {str(r['model_cached']):<15} {r['tokenizer_cached']}")
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            print(f"\nwrote {args.output}")
        return

    if args.download_all:
        print("Discovering model configs...")
        results = download_all(
            tokenizer_only=args.tokenizer_only,
            skip_cached=args.skip_cached,
            max_retries=args.max_retries,
            fallback_tokenizer_only=args.fallback_tokenizer_only,
        )
        print(f"\n{format_summary_table(results)}")

        failed = [r for r in results if r["status"] == "failed"]
        if failed:
            print(f"\n{len(failed)} model(s) failed to download.")

        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            print(f"\nwrote {args.output}")

        if failed:
            sys.exit(1)
        return

    # Single model download (backward compatible)
    model_cfg = load_yaml(args.model_config)
    result = download_one(
        model_cfg,
        tokenizer_only=args.tokenizer_only,
        max_retries=args.max_retries,
        fallback_tokenizer_only=args.fallback_tokenizer_only,
    )

    if result["status"] == "failed":
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
