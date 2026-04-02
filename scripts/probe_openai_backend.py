from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen35_turboquant_workflow_study.config import load_yaml
from qwen35_turboquant_workflow_study.backends.openai_compatible import OpenAICompatibleBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe an OpenAI-compatible backend with one prompt.")
    parser.add_argument("--backend-config", default="configs/backends/qwen35_openai_server.yaml")
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    backend_cfg = load_yaml(args.backend_config)
    backend = OpenAICompatibleBackend(backend_cfg)

    print("===== backend description =====")
    print(json.dumps(backend.describe(), indent=2))
    print()

    runtime_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": backend_cfg.get("request_defaults", {}).get("temperature", 0.0),
        "top_p": backend_cfg.get("request_defaults", {}).get("top_p", 1.0),
    }

    result = backend.generate_one(args.prompt, runtime_cfg)

    print("===== generation result =====")
    print(
        json.dumps(
            {
                "prompt_tokens": result["prompt_tokens"],
                "output_tokens": result["output_tokens"],
                "latency_s": result["latency_s"],
                "tokens_per_second": result["tokens_per_second"],
                "peak_vram_gb": result["peak_vram_gb"],
            },
            indent=2,
        )
    )
    print()
    print("===== output_text =====")
    print(result["output_text"])


if __name__ == "__main__":
    main()
