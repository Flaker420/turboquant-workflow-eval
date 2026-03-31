from __future__ import annotations

from .stats import ProjectionTensorStats


class LayerCaptureStats:
    def __init__(self) -> None:
        self.q = ProjectionTensorStats()
        self.k = ProjectionTensorStats()
        self.v = ProjectionTensorStats()

    def to_dict(self) -> dict:
        return {
            "q": self.q.to_dict(),
            "k": self.k.to_dict(),
            "v": self.v.to_dict(),
        }


class ProjectionHookBook:
    def __init__(self, attention_blocks) -> None:
        self._stats = {block.module_path: LayerCaptureStats() for block in attention_blocks}

    def update(self, block_path: str, kind: str, output) -> None:
        layer_stats = self._stats[block_path]
        getattr(layer_stats, kind).update_from_output(output)

    def to_dict(self) -> dict:
        return {key: value.to_dict() for key, value in self._stats.items()}


class ProjectionHookManager:
    def __init__(self, model_root, attention_blocks, book: ProjectionHookBook) -> None:
        self.model_root = model_root
        self.attention_blocks = attention_blocks
        self.book = book
        self.handles = []

    def _register_one(self, block, kind: str, path: str | None) -> None:
        if not path:
            return
        module = self.model_root.get_submodule(path)
        handle = module.register_forward_hook(
            lambda _module, _inputs, output, block_path=block.module_path, tensor_kind=kind: self.book.update(
                block_path,
                tensor_kind,
                output,
            )
        )
        self.handles.append(handle)

    def register(self):
        for block in self.attention_blocks:
            self._register_one(block, "q", block.q_proj_path)
            self._register_one(block, "k", block.k_proj_path)
            self._register_one(block, "v", block.v_proj_path)
        return self

    def close(self) -> None:
        while self.handles:
            self.handles.pop().remove()

    def __enter__(self):
        return self.register()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
