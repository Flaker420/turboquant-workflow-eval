# Adapter interface

A compression policy is represented by an adapter class.

## Purpose

The repository ships with a **baseline pass-through adapter** and a **TurboQuantAdapter** that wraps turboquant-core (`src/turboquant_workflow_eval/adapters/turboquant.py`). The safe and aggressive policy templates use this adapter by default. To use a different backend, implement a class that follows the adapter contract below and point a policy config at it.

## Contract

Your class should inherit from `CompressionAdapter` and implement:

- `name` -- human-readable adapter name
- `prepare_model(model, tokenizer, model_cfg, policy_cfg)` -- return the prepared model and tokenizer
- `describe(policy_cfg)` -- return a small dictionary for metadata
- `cleanup(model)` -- optional teardown

## Minimal example

```python
from turboquant_workflow_eval.adapters.base import CompressionAdapter

class MyTurboQuantAdapter(CompressionAdapter):
    name = "my-turboquant"

    def prepare_model(self, model, tokenizer, model_cfg, policy_cfg):
        # apply your compression backend here
        return model, tokenizer

    def describe(self, policy_cfg):
        return {
            "backend": "custom",
            "policy": policy_cfg.get("name"),
        }
```

## Policy config example

```yaml
name: turboquant_safe
enabled: true
adapter:
  import_path: my_package.my_module:MyTurboQuantAdapter
settings:
  note: conservative full-attention policy
```

## Important

This repository does not assume every backend mutates the model the same way. The adapter layer exists so you can integrate:

- a direct Transformers patch
- a local experimental implementation
- a thin wrapper around a separate package
- a staging backend you are evaluating internally

## Wiring turboquant-core

> **Note:** turboquant-core is already wired. The built-in `TurboQuantAdapter` at `src/turboquant_workflow_eval/adapters/turboquant.py` wraps the core library and is used by both policy templates. The example below is for reference if you need to write a custom adapter with different behavior.

[turboquant-core](https://github.com/Flaker420/turboquant-core) provides model-specific backends for KV-cache compression:

| Model | Backend |
|-------|---------|
| Qwen3.5-9B | `Qwen35KVBackend` |
| Qwen3-8B | `Qwen3DenseKVBackend` |

A minimal adapter wrapping turboquant-core would look like:

```python
from turboquant_workflow_eval.adapters.base import CompressionAdapter
from turboquant_core.backends import Qwen35KVBackend

class TurboQuantSafeAdapter(CompressionAdapter):
    name = "turboquant-safe"

    def prepare_model(self, model, tokenizer, model_cfg, policy_cfg):
        backend = Qwen35KVBackend(
            bit_width=policy_cfg["settings"].get("bit_width", 4),
        )
        model = backend.patch(model)
        return model, tokenizer

    def describe(self, policy_cfg):
        return {
            "adapter": self.name,
            "bit_width": policy_cfg["settings"].get("bit_width", 4),
        }
```

Adjust the backend constructor arguments to match your turboquant-core version. The `prepare_model` call receives the full `policy_cfg` dict, so all `settings:` keys from the YAML are available.
