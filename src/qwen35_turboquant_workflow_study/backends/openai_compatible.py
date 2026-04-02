from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import GenerationBackend


class OpenAICompatibleBackend(GenerationBackend):
    name = "openai-compatible"

    def __init__(self, backend_cfg: dict):
        self.backend_cfg = backend_cfg
        self.base_url = backend_cfg["base_url"].rstrip("/")
        self.model = backend_cfg["model"]
        self.api_key_env = backend_cfg.get("api_key_env")
        self.request_defaults = backend_cfg.get("request_defaults", {})
        self.request_extra_json = backend_cfg.get("request_extra_json", {})

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "request_defaults": self.request_defaults,
            "request_extra_json": self.request_extra_json,
        }

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            api_key = os.getenv(self.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def generate_one(self, prompt_text: str, runtime_cfg: dict) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt_text}],
            "stream": False,
            "max_tokens": int(runtime_cfg.get("max_new_tokens", self.request_defaults.get("max_tokens", 256))),
            "temperature": float(runtime_cfg.get("temperature", self.request_defaults.get("temperature", 0.0))),
            "top_p": float(runtime_cfg.get("top_p", self.request_defaults.get("top_p", 1.0))),
        }
        payload.update(self.request_extra_json)

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers=self._headers(),
            method="POST",
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from backend: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach backend at {self.base_url}: {exc}") from exc

        latency_s = time.perf_counter() - start
        choice = raw["choices"][0]
        message = choice.get("message", {})
        output_text = message.get("content", "")
        usage = raw.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        if output_tokens is None:
            output_tokens = len(output_text.split())
        tokens_per_second = (output_tokens / latency_s) if latency_s > 0 else None

        return {
            "rendered_prompt": None,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "latency_s": latency_s,
            "tokens_per_second": tokens_per_second,
            "peak_vram_gb": None,
            "output_text": output_text,
            "raw_response": raw,
        }
