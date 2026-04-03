from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentArm:
    id: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentPlan:
    name: str
    phase: int | str
    description: str = ""
    arms: list[ExperimentArm] = field(default_factory=list)


@dataclass(frozen=True)
class AttentionBlockRef:
    index: int
    module_path: str
    class_name: str
    q_proj_path: str | None = None
    k_proj_path: str | None = None
    v_proj_path: str | None = None
    packed_qkv_path: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptSpec:
    id: str
    category: str
    title: str
    prompt: str
    watch_for: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
