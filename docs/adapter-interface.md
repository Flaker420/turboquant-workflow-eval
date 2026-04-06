# Adapter interface

A compression policy is represented by an adapter class.

## Purpose

The repository ships with a **baseline pass-through adapter** and a **TurboQuantAdapter** that wraps turboquant-core (`src/turboquant_workflow_eval/adapters/turboquant.py`). The safe and aggressive policy templates use this adapter by default. To use a different backend, implement a class that follows the adapter contract below and point a policy config at it.

## Contract

Your class should inherit from `CompressionAdapter` and implement:

### Required methods

- `name` -- human-readable adapter name
- `prepare_model(model, tokenizer, model_cfg, policy_cfg)` -- return the prepared model and tokenizer
- `describe(policy_cfg)` -- return a small dictionary for metadata
- `cleanup(model)` -- optional teardown

### Optional methods (model reuse and inspection)

These methods enable the study runner to reuse a single model across multiple policies instead of reloading from scratch each time. All have safe default implementations in `CompressionAdapter` -- subclasses that do not override them keep working.

- `can_revert() -> bool` -- return `True` if `revert()` can undo `prepare_model()` in-place. Default: `True`
- `revert(model) -> bool` -- undo the changes made by `prepare_model()`. Return `True` if the model is clean and reusable, `False` if a full reload is needed. Default: no-op, returns `True`
- `get_state() -> dict` -- return current compression parameters for inspection (used by the Quick Test UI tab). Default: empty dict
- `update_params(params) -> bool` -- hot-update compression params without full revert + reapply. Return `True` if applied, `False` if not supported. Default: returns `False`. **The built-in `TurboQuantAdapter` does not support hot updates and raises `NotImplementedError`** -- callers must `revert(model)` and `prepare_model(...)` again with the new settings.
- `reset_generation_state() -> None` -- clear per-generation state (e.g. KV cache, internal buffers) before each generation attempt. Called by the study runner before every `generate_one`, including each repetition, so each attempt starts from a clean state. Default: no-op. Override this if your adapter accumulates state across generations.

When `can_revert()` returns `True`, the study runner loads the model once and calls `prepare_model()` / `revert()` for each policy. When it returns `False`, the model is reloaded from scratch before each policy.

> **Note on the built-in `TurboQuantAdapter` wrapper:** the wrapper at `src/turboquant_workflow_eval/adapters/turboquant.py` delegates every method directly to `turboquant_core.TurboQuantAdapter` without `hasattr` fallbacks. If a future core release renames or removes one of these methods, the wrapper will raise `AttributeError` loudly rather than silently returning a fake state. The wrapper additionally validates that `model_cfg["model_name"]` is set and raises `ValueError` early if not.

## Minimal example

```python
from turboquant_workflow_eval.adapters.base import CompressionAdapter

class MyTurboQuantAdapter(CompressionAdapter):
    name = "my-turboquant"

    def prepare_model(self, model, tokenizer, model_cfg, policy_cfg):
        # apply your compression backend here
        self._original_weights = save_weights(model)  # for revert
        return model, tokenizer

    def describe(self, policy_cfg):
        return {
            "backend": "custom",
            "policy": policy_cfg.get("name"),
        }

    def can_revert(self):
        return True

    def revert(self, model):
        restore_weights(model, self._original_weights)
        return True

    def get_state(self):
        return {"backend": "custom", "bit_width": 4}
```

## Policy config example

```yaml
name: turboquant_safe
enabled: true
adapter:
  import_path: turboquant_workflow_eval.adapters.turboquant:TurboQuantAdapter
settings:
  bit_width: 4
  seed: 42
  residual_window: 0          # optional, forwarded to turboquant-core
  key_strategy: "mse+qjl"     # optional, forwarded to turboquant-core
```

`adapter.import_path` must match `module.path:ClassName` -- this is enforced at config load time by `validate_policy_config`. The `settings` block is passed unchanged into the adapter; for the built-in `TurboQuantAdapter` the supported keys are `bit_width`, `seed`, `residual_window`, `key_strategy`, and (for advanced use) `model_variant`. All of them are recorded in `describe()` so they appear in `run_summary.json` and per-row CSV/JSONL output.

### Overriding policy settings without editing the YAML

Any key inside a policy YAML can be overridden at load time without touching the file on disk. From the CLI:

```bash
python -m turboquant_workflow_eval --study-config configs/studies/default_qwen35_9b.yaml \
  --set-policy turboquant_safe.settings.key_strategy=mse \
  --set-policy '*.settings.bit_width=8' \
  --set-policy baseline.enabled=false
```

The first segment is matched against `policy_cfg["name"]`; `*` matches every policy. The remaining dot-path is applied via `apply_dot_overrides`. The Gradio UI exposes the same mechanism through the **Policy Overrides** accordion on the Study Runner tab — one override per line. Overrides are applied before `validate_policy_config`, so an override that produces an invalid policy fails fast with a clear error.

## Important

This repository does not assume every backend mutates the model the same way. The adapter layer exists so you can integrate:

- a direct Transformers patch
- a local experimental implementation
- a thin wrapper around a separate package
- a staging backend you are evaluating internally

## Wiring turboquant-core

> **Note:** turboquant-core is already wired. The built-in `TurboQuantAdapter` at `src/turboquant_workflow_eval/adapters/turboquant.py` wraps the core library and is used by both policy templates. You should normally just point your policy at `turboquant_workflow_eval.adapters.turboquant:TurboQuantAdapter` -- write a custom adapter only if you need different behavior.

[turboquant-core](https://github.com/Flaker420/turboquant-core) provides model-specific backends for KV-cache compression:

| Model | Backend | Patch helper |
|-------|---------|--------------|
| Qwen3.5-9B (hybrid) | `Qwen35KVBackend` | `patch_qwen35_with_tq` |
| Qwen3-8B (dense) | `Qwen3DenseKVBackend` | `patch_qwen3_with_tq` |
| Qwen2.5-3B-Instruct (dense) | `Qwen25DenseKVBackend` | `patch_qwen25_with_tq` |

If you really do want to write your own adapter from scratch, the patch helpers are the entry point:

```python
from turboquant_workflow_eval.adapters.base import CompressionAdapter
from turboquant_core.backends.qwen_hook import patch_qwen35_with_tq, unpatch_model


class MyTurboQuantAdapter(CompressionAdapter):
    name = "my-turboquant"

    def __init__(self) -> None:
        self._cache = None
        self._patched = False

    def prepare_model(self, model, tokenizer, model_cfg, policy_cfg):
        settings = policy_cfg.get("settings", {})
        self._cache = patch_qwen35_with_tq(
            model,
            bit_width=settings.get("bit_width", 4),
            seed=settings.get("seed", 42),
            residual_window=settings.get("residual_window", 0),
            key_strategy=settings.get("key_strategy", "mse+qjl"),
        )
        self._patched = True
        return model, tokenizer

    def can_revert(self) -> bool:
        return self._patched

    def revert(self, model) -> bool:
        if not self._patched:
            return False
        unpatch_model(model)
        if self._cache is not None:
            self._cache.clear()
            self._cache = None
        self._patched = False
        return True

    def describe(self, policy_cfg):
        s = policy_cfg.get("settings", {})
        return {
            "adapter": self.name,
            "bit_width": s.get("bit_width", 4),
            "seed": s.get("seed", 42),
            "residual_window": s.get("residual_window", 0),
            "key_strategy": s.get("key_strategy", "mse+qjl"),
        }
```

This is essentially what `turboquant_core.TurboQuantAdapter` already does -- the `TurboQuantAdapter` wrapper in this repo delegates straight to it.
