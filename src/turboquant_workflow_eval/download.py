from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .config import load_yaml


def discover_model_configs(config_dir: str | Path = "configs/model") -> list[dict]:
    """Discover all model config YAMLs in the given directory."""
    config_dir = Path(config_dir)
    results = []
    for path in sorted(config_dir.glob("*.yaml")):
        cfg = load_yaml(path)
        cfg["_config_path"] = str(path)
        results.append(cfg)
    return results


def check_cache_status(model_name: str, cache_dir: str | None = None) -> dict:
    """Check whether a model and its tokenizer are already in the HuggingFace cache."""
    cache_dir = cache_dir or os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    repo_dir = Path(cache_dir) / "hub" / f"models--{model_name.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return {"model_cached": False, "tokenizer_cached": False}
    for snap in sorted(snapshots.iterdir(), reverse=True):
        if not snap.is_dir():
            continue
        has_model = (
            (snap / "model.safetensors.index.json").exists()
            or (snap / "pytorch_model.bin.index.json").exists()
            or (snap / "model.safetensors").exists()
            or (snap / "pytorch_model.bin").exists()
        )
        has_tokenizer = (snap / "tokenizer_config.json").exists()
        if has_model or has_tokenizer:
            return {"model_cached": has_model, "tokenizer_cached": has_tokenizer}
    return {"model_cached": False, "tokenizer_cached": False}


def _download_tokenizer(model_cfg: dict) -> dict[str, Any]:
    import transformers

    model_name = model_cfg["model_name"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    return {
        "model_name": model_name,
        "status": "ok",
        "downloaded": "tokenizer_only",
        "tokenizer_class": tokenizer.__class__.__name__,
        "error": None,
    }


def _download_model_and_tokenizer(model_cfg: dict) -> dict[str, Any]:
    from .model_loader import load_model_and_tokenizer

    model_name = model_cfg["model_name"]
    model, tokenizer, loader_name = load_model_and_tokenizer(model_cfg)
    return {
        "model_name": model_name,
        "status": "ok",
        "downloaded": "model_and_tokenizer",
        "loader_name": loader_name,
        "model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "error": None,
    }


def download_one(
    model_cfg: dict,
    tokenizer_only: bool = False,
    max_retries: int = 3,
    fallback_tokenizer_only: bool = True,
) -> dict:
    """Download a single model with retry logic and optional tokenizer-only fallback."""
    model_name = model_cfg["model_name"]
    download_fn = _download_tokenizer if tokenizer_only else _download_model_and_tokenizer
    last_error = None

    for attempt in range(max_retries):
        try:
            return download_fn(model_cfg)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [{model_name}] attempt {attempt + 1}/{max_retries} failed, retry in {wait}s: {exc}")
                time.sleep(wait)

    if fallback_tokenizer_only and not tokenizer_only:
        print(f"  [{model_name}] full model failed after {max_retries} attempts, falling back to tokenizer-only")
        try:
            result = _download_tokenizer(model_cfg)
            result["status"] = "tokenizer_only_fallback"
            return result
        except Exception as fallback_exc:
            return {
                "model_name": model_name,
                "status": "failed",
                "downloaded": None,
                "error": f"model: {last_error}; tokenizer fallback: {fallback_exc}",
            }

    return {
        "model_name": model_name,
        "status": "failed",
        "downloaded": None,
        "error": str(last_error),
    }


def download_all(
    config_dir: str | Path = "configs/model",
    tokenizer_only: bool = False,
    skip_cached: bool = True,
    max_retries: int = 3,
    fallback_tokenizer_only: bool = True,
) -> list[dict]:
    """Discover all model configs and download them, skipping cached models if requested."""
    configs = discover_model_configs(config_dir)
    if not configs:
        print(f"No model configs found in {config_dir}")
        return []

    results = []
    for cfg in configs:
        model_name = cfg["model_name"]
        config_path = cfg.get("_config_path", "unknown")

        if skip_cached:
            cache = check_cache_status(model_name)
            need_model = not tokenizer_only and not cache["model_cached"]
            need_tokenizer = not cache["tokenizer_cached"]
            if not need_model and not need_tokenizer:
                results.append({
                    "model_name": model_name,
                    "status": "cached",
                    "downloaded": None,
                    "config_path": config_path,
                    "error": None,
                })
                print(f"  [{model_name}] already cached, skipping")
                continue

        print(f"  [{model_name}] downloading ({'tokenizer only' if tokenizer_only else 'model + tokenizer'})...")
        result = download_one(
            cfg,
            tokenizer_only=tokenizer_only,
            max_retries=max_retries,
            fallback_tokenizer_only=fallback_tokenizer_only,
        )
        result["config_path"] = config_path
        results.append(result)

    return results


def format_summary_table(results: list[dict]) -> str:
    """Format download results as a human-readable table."""
    if not results:
        return "No models processed."
    lines = [
        f"{'Model':<30} {'Status':<25} {'Details'}",
        f"{'-' * 30} {'-' * 25} {'-' * 30}",
    ]
    for r in results:
        name = r["model_name"]
        status = r["status"]
        if status == "ok":
            details = r.get("downloaded", "")
        elif status == "cached":
            details = "already in cache"
        elif status == "tokenizer_only_fallback":
            details = "model failed, tokenizer cached"
        else:
            details = r.get("error", "unknown error")[:50]
        lines.append(f"{name:<30} {status:<25} {details}")
    return "\n".join(lines)
