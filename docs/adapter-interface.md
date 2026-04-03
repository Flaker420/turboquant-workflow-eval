# Adapter interface

A compression policy is represented by an adapter class.

## Purpose

The repository ships with a **baseline pass-through adapter** only. To test a real TurboQuant policy, implement a class that follows the adapter contract and point a policy config at it.

## Contract

Your class should inherit from `CompressionAdapter` and implement:

- `name` – human-readable adapter name
- `prepare_model(model, tokenizer, model_cfg, policy_cfg)` – return the prepared model and tokenizer
- `describe(policy_cfg)` – return a small dictionary for metadata
- `cleanup(model)` – optional teardown

## Minimal example

```python
from qwen35_turboquant_workflow_study.adapters.base import CompressionAdapter

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
