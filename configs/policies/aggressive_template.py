"""Aggressive TurboQuant policy: 2-bit MSE keys."""

from turboquant_workflow_eval.schema import (
    AdapterSpec,
    PolicyConfig,
    PolicySettings,
)

POLICY = PolicyConfig(
    name="turboquant_aggressive",
    enabled=True,
    comparison_label="aggressive",
    adapter=AdapterSpec(
        import_path="turboquant_workflow_eval.adapters.turboquant:TurboQuantAdapter"
    ),
    settings=PolicySettings(
        bit_width=2,
        key_strategy="mse",
        profile="aggressive",
    ),
    notes=("Stress profile — run after the safe policy is validated.",),
)
