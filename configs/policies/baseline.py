"""Pass-through baseline policy (no compression)."""

from turboquant_workflow_eval.schema import (
    AdapterSpec,
    PolicyConfig,
    PolicySettings,
)

POLICY = PolicyConfig(
    name="baseline",
    enabled=True,
    comparison_label="baseline",
    adapter=AdapterSpec(
        import_path="turboquant_workflow_eval.adapters.none:NoCompressionAdapter"
    ),
    settings=PolicySettings(),  # all defaults — baseline does not use them
    notes=("Pass-through baseline for workflow comparison.",),
)
