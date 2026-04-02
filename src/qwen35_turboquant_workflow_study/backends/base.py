from __future__ import annotations


class GenerationBackend:
    name = "backend-base"

    def describe(self) -> dict:
        return {"backend": self.name}

    def generate_one(self, prompt_text: str, runtime_cfg: dict) -> dict:
        raise NotImplementedError
