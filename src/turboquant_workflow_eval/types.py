from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
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


@dataclass
class StudyContext:
    """Holds all resolved configuration for a study run (no GPU state)."""

    study_cfg: dict[str, Any]
    model_cfg: dict[str, Any]
    model_cfg_path: Path
    prompt_pack: list[Any]  # list[PromptSpec] — forward ref avoidance
    policy_paths: list[Path]
    runtime_cfg: dict[str, Any]
    thresholds_cfg: dict[str, Any]
    output_dir: Path
    repetitions: int = 1


@dataclass(frozen=True)
class PromptSpec:
    id: str
    category: str
    title: str
    prompt: str
    watch_for: str = ""
    reference_answer: str | None = None
    test_cases: tuple[dict[str, str], ...] | None = None
    turns: tuple[dict[str, str], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
