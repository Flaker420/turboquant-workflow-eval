"""Frozen dataclass schema for turboquant-workflow-eval study configurations.

Replaces the legacy YAML + ``validate_*_config`` dict pipeline. Studies, models,
and policies are now authored as Python modules that construct the dataclasses
defined here. Validation lives in ``__post_init__`` so configs can never be
constructed in an invalid state.

A small ``replace_path`` helper provides dot-path overrides on nested frozen
dataclasses, used by the CLI's ``--set`` / ``--set-policy`` escape hatches and
by the per-knob ``--KNOB`` / ``--KNOB-for`` flag handlers.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, TypeVar

__all__ = [
    "LayoutConfig",
    "ContextConfig",
    "ModelConfig",
    "PolicySettings",
    "AdapterSpec",
    "PolicyConfig",
    "RuntimeConfig",
    "ThresholdsConfig",
    "OutputsConfig",
    "EarlyStopConfig",
    "StudyConfig",
    "ConfigValidationError",
    "replace_path",
    "policy_to_legacy_dict",
    "model_to_legacy_dict",
]


class ConfigValidationError(ValueError):
    """Raised when a dataclass field violates its schema constraints."""


_VALID_DTYPES = frozenset(
    {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}
)
_VALID_KEY_STRATEGIES = frozenset({"mse", "mse+qjl"})
_VALID_VALUE_STRATEGIES = frozenset({"mse"})
_IMPORT_PATH_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LayoutConfig:
    total_lm_layers: int
    attention_blocks: int
    deltanet_blocks: int = 0
    pattern: str = ""

    def __post_init__(self) -> None:
        if self.total_lm_layers <= 0:
            raise ConfigValidationError(
                f"LayoutConfig.total_lm_layers must be > 0, got {self.total_lm_layers}"
            )
        if self.attention_blocks < 0:
            raise ConfigValidationError(
                f"LayoutConfig.attention_blocks must be >= 0, got {self.attention_blocks}"
            )
        if self.deltanet_blocks < 0:
            raise ConfigValidationError(
                f"LayoutConfig.deltanet_blocks must be >= 0, got {self.deltanet_blocks}"
            )


@dataclass(frozen=True, slots=True)
class ContextConfig:
    short: int
    medium: int
    long: int | None = None

    def __post_init__(self) -> None:
        for label, value in (("short", self.short), ("medium", self.medium)):
            if value <= 0:
                raise ConfigValidationError(
                    f"ContextConfig.{label} must be > 0, got {value}"
                )
        if self.long is not None and self.long <= 0:
            raise ConfigValidationError(
                f"ContextConfig.long must be > 0 or None, got {self.long}"
            )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_name: str
    dtype: str
    device_map: str | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    chat_template_mode: str = "auto"
    language_model_only: bool = False
    layout: LayoutConfig | None = None
    context: ContextConfig | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ConfigValidationError("ModelConfig.model_name must be non-empty")
        if self.dtype not in _VALID_DTYPES:
            raise ConfigValidationError(
                f"ModelConfig.dtype {self.dtype!r} not in {sorted(_VALID_DTYPES)}"
            )
        # Coerce notes to a tuple even if a list snuck in.
        if not isinstance(self.notes, tuple):
            object.__setattr__(self, "notes", tuple(self.notes))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PolicySettings:
    bit_width: int = 4
    seed: int = 42
    residual_window: int = 0
    key_strategy: str = "mse+qjl"
    value_strategy: str = "mse"
    profile: str | None = None
    model_variant: str | None = None
    compressible_layers: tuple[int, ...] | None = None
    compressible_heads: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.bit_width < 1:
            raise ConfigValidationError(
                f"PolicySettings.bit_width must be >= 1, got {self.bit_width}"
            )
        if self.key_strategy not in _VALID_KEY_STRATEGIES:
            raise ConfigValidationError(
                f"PolicySettings.key_strategy {self.key_strategy!r} not in "
                f"{sorted(_VALID_KEY_STRATEGIES)}"
            )
        if self.value_strategy not in _VALID_VALUE_STRATEGIES:
            raise ConfigValidationError(
                f"PolicySettings.value_strategy {self.value_strategy!r} not in "
                f"{sorted(_VALID_VALUE_STRATEGIES)}"
            )
        if self.residual_window < 0:
            raise ConfigValidationError(
                f"PolicySettings.residual_window must be >= 0, got {self.residual_window}"
            )
        for field_name in ("compressible_layers", "compressible_heads"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, tuple):
                value = tuple(value)
                object.__setattr__(self, field_name, value)
            seen: set[int] = set()
            for idx in value:
                if isinstance(idx, bool) or not isinstance(idx, int):
                    raise ConfigValidationError(
                        f"PolicySettings.{field_name} must contain ints, got {idx!r}"
                    )
                if idx < 0:
                    raise ConfigValidationError(
                        f"PolicySettings.{field_name} indices must be >= 0, got {idx}"
                    )
                if idx in seen:
                    raise ConfigValidationError(
                        f"PolicySettings.{field_name} contains duplicate index {idx}"
                    )
                seen.add(idx)
            if not seen:
                raise ConfigValidationError(
                    f"PolicySettings.{field_name} must be non-empty when set"
                )


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    import_path: str

    def __post_init__(self) -> None:
        if not _IMPORT_PATH_RE.match(self.import_path):
            raise ConfigValidationError(
                f"AdapterSpec.import_path {self.import_path!r} must match "
                f"'module.path:ClassName'"
            )


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    name: str
    adapter: AdapterSpec
    enabled: bool = True
    comparison_label: str | None = None
    settings: PolicySettings = field(default_factory=PolicySettings)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigValidationError("PolicyConfig.name must be non-empty")
        if self.comparison_label is None:
            object.__setattr__(self, "comparison_label", self.name)
        if not isinstance(self.notes, tuple):
            object.__setattr__(self, "notes", tuple(self.notes))


# ---------------------------------------------------------------------------
# Runtime / thresholds / outputs / early-stop
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    max_input_tokens: int
    max_new_tokens: int
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    use_cache: bool = True
    cleanup_between_policies: bool = True
    repetitions: int = 1
    shuffle_policies: bool = False
    shuffle_seed: int = 42

    def __post_init__(self) -> None:
        if self.max_input_tokens <= 0:
            raise ConfigValidationError(
                f"RuntimeConfig.max_input_tokens must be > 0, got {self.max_input_tokens}"
            )
        if self.max_new_tokens <= 0:
            raise ConfigValidationError(
                f"RuntimeConfig.max_new_tokens must be > 0, got {self.max_new_tokens}"
            )
        if self.temperature < 0.0:
            raise ConfigValidationError(
                f"RuntimeConfig.temperature must be >= 0.0, got {self.temperature}"
            )
        if not (0.0 < self.top_p <= 1.0):
            raise ConfigValidationError(
                f"RuntimeConfig.top_p must be in (0.0, 1.0], got {self.top_p}"
            )
        if self.repetitions < 1:
            raise ConfigValidationError(
                f"RuntimeConfig.repetitions must be >= 1, got {self.repetitions}"
            )


@dataclass(frozen=True, slots=True)
class ThresholdsConfig:
    latency_yellow_pct: float | None = None
    latency_red_pct: float | None = None
    similarity_yellow: float | None = None
    similarity_red: float | None = None
    output_length_yellow_pct: float | None = None
    output_length_red_pct: float | None = None
    per_category: Mapping[str, "ThresholdsConfig"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce mutable defaults to immutable forms.
        if not isinstance(self.per_category, dict):
            object.__setattr__(self, "per_category", dict(self.per_category))


@dataclass(frozen=True, slots=True)
class OutputsConfig:
    write_individual_text_files: bool = True
    truncate_csv_output_to_chars: int = 180

    def __post_init__(self) -> None:
        if self.truncate_csv_output_to_chars < 0:
            raise ConfigValidationError(
                "OutputsConfig.truncate_csv_output_to_chars must be >= 0, "
                f"got {self.truncate_csv_output_to_chars}"
            )


@dataclass(frozen=True, slots=True)
class EarlyStopConfig:
    max_red_verdicts: int | None = None
    max_error_rate: float | None = None

    def __post_init__(self) -> None:
        if self.max_red_verdicts is not None and self.max_red_verdicts < 0:
            raise ConfigValidationError(
                f"EarlyStopConfig.max_red_verdicts must be >= 0 or None, "
                f"got {self.max_red_verdicts}"
            )
        if self.max_error_rate is not None and not (0.0 <= self.max_error_rate <= 1.0):
            raise ConfigValidationError(
                f"EarlyStopConfig.max_error_rate must be in [0.0, 1.0] or None, "
                f"got {self.max_error_rate}"
            )


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StudyConfig:
    name: str
    model: ModelConfig
    prompt_pack: tuple[Path, ...]
    policies: tuple[PolicyConfig, ...]
    runtime: RuntimeConfig
    baseline_policy_name: str | None = None
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigValidationError("StudyConfig.name must be non-empty")

        # Coerce prompt_pack to a tuple of Path. Accept str | Path | iterable.
        if isinstance(self.prompt_pack, (str, Path)):
            packs: tuple[Path, ...] = (Path(self.prompt_pack),)
        else:
            packs = tuple(Path(p) for p in self.prompt_pack)
        if not packs:
            raise ConfigValidationError("StudyConfig.prompt_pack must be non-empty")
        object.__setattr__(self, "prompt_pack", packs)

        # Coerce policies to tuple.
        if not isinstance(self.policies, tuple):
            object.__setattr__(self, "policies", tuple(self.policies))
        if not self.policies:
            raise ConfigValidationError("StudyConfig.policies must be non-empty")
        for p in self.policies:
            if not isinstance(p, PolicyConfig):
                raise ConfigValidationError(
                    "StudyConfig.policies entries must be PolicyConfig instances, "
                    f"got {type(p).__name__}"
                )

        names = [p.name for p in self.policies]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ConfigValidationError(
                f"StudyConfig.policies contains duplicate names: {dupes}"
            )

        if len(self.policies) > 1:
            if not self.baseline_policy_name:
                raise ConfigValidationError(
                    "StudyConfig.baseline_policy_name is required when len(policies) > 1"
                )
            if self.baseline_policy_name not in names:
                raise ConfigValidationError(
                    f"StudyConfig.baseline_policy_name {self.baseline_policy_name!r} "
                    f"not in policies {names}"
                )
        else:
            # Single-policy: auto-set baseline to that policy.
            if not self.baseline_policy_name:
                object.__setattr__(self, "baseline_policy_name", names[0])


# ---------------------------------------------------------------------------
# Path-based override helper
# ---------------------------------------------------------------------------


T = TypeVar("T")


def replace_path(obj: T, path: str, value: Any) -> T:
    """Return a new dataclass with the dot-path field set to ``value``.

    Recurses through nested frozen dataclasses, calling ``dataclasses.replace``
    at each level. Tuples and lists are addressed by integer index. Raises
    ``ConfigValidationError`` (so the CLI can present a clean message) on any
    invalid path or value.
    """
    if not path:
        raise ConfigValidationError("replace_path: empty path")
    parts = path.split(".")
    return _replace_path(obj, parts, value, full_path=path)


def _replace_path(obj: Any, parts: list[str], value: Any, *, full_path: str) -> Any:
    head, *rest = parts
    if isinstance(obj, tuple):
        try:
            idx = int(head)
        except ValueError as exc:
            raise ConfigValidationError(
                f"replace_path({full_path!r}): tuple element index must be int, "
                f"got {head!r}"
            ) from exc
        if idx < 0 or idx >= len(obj):
            raise ConfigValidationError(
                f"replace_path({full_path!r}): tuple index {idx} out of range "
                f"(length {len(obj)})"
            )
        if rest:
            new_item = _replace_path(obj[idx], rest, value, full_path=full_path)
        else:
            new_item = value
        return obj[:idx] + (new_item,) + obj[idx + 1 :]

    if not dataclasses.is_dataclass(obj):
        raise ConfigValidationError(
            f"replace_path({full_path!r}): cannot descend into non-dataclass "
            f"{type(obj).__name__}"
        )

    field_names = {f.name for f in dataclasses.fields(obj)}
    if head not in field_names:
        raise ConfigValidationError(
            f"replace_path({full_path!r}): {type(obj).__name__} has no field {head!r}. "
            f"Valid fields: {sorted(field_names)}"
        )

    current = getattr(obj, head)
    if rest:
        new_value = _replace_path(current, rest, value, full_path=full_path)
    else:
        new_value = value
    return replace(obj, **{head: new_value})


# ---------------------------------------------------------------------------
# Legacy dict bridges (used at the core adapter seam)
# ---------------------------------------------------------------------------


def policy_to_legacy_dict(policy: PolicyConfig) -> dict[str, Any]:
    """Convert a PolicyConfig back to the legacy dict shape that
    ``turboquant_core.adapters.workflow_eval.TurboQuantAdapter`` expects.

    Core deliberately keeps a dict-shaped boundary so it can be vendored
    without depending on the harness's schema module.
    """
    settings_dict: dict[str, Any] = {}
    for f in dataclasses.fields(policy.settings):
        v = getattr(policy.settings, f.name)
        if v is None:
            continue
        if isinstance(v, tuple):
            v = list(v)
        settings_dict[f.name] = v
    return {
        "name": policy.name,
        "enabled": policy.enabled,
        "comparison_label": policy.comparison_label,
        "adapter": {"import_path": policy.adapter.import_path},
        "settings": settings_dict,
        "notes": list(policy.notes),
    }


def model_to_legacy_dict(model: ModelConfig) -> dict[str, Any]:
    """Convert a ModelConfig back to the legacy dict shape used by core."""
    out: dict[str, Any] = {
        "model_name": model.model_name,
        "dtype": model.dtype,
        "trust_remote_code": model.trust_remote_code,
        "chat_template_mode": model.chat_template_mode,
        "language_model_only": model.language_model_only,
        "notes": list(model.notes),
    }
    if model.device_map is not None:
        out["device_map"] = model.device_map
    if model.attn_implementation is not None:
        out["attn_implementation"] = model.attn_implementation
    if model.layout is not None:
        out["layout"] = {
            "total_lm_layers": model.layout.total_lm_layers,
            "attention_blocks": model.layout.attention_blocks,
            "deltanet_blocks": model.layout.deltanet_blocks,
            "pattern": model.layout.pattern,
        }
    if model.context is not None:
        out["context"] = {
            "short": model.context.short,
            "medium": model.context.medium,
            "long": model.context.long,
        }
    return out
