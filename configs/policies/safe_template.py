"""Conservative TurboQuant policy: 4-bit MSE keys."""

from turboquant_workflow_eval.schema import (
    AdapterSpec,
    PolicyConfig,
    PolicySettings,
)

POLICY = PolicyConfig(
    name="turboquant_safe",
    enabled=True,
    comparison_label="safe",
    adapter=AdapterSpec(
        import_path="turboquant_workflow_eval.adapters.turboquant:TurboQuantAdapter"
    ),
    settings=PolicySettings(
        bit_width=4,
        key_strategy="mse",
        profile="safe",
    ),
    notes=("Conservative compression profile for first workflow comparison.",),
)
