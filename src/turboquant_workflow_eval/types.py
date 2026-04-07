from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .schema import StudyConfig


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
    """Holds all resolved configuration for a study run (no GPU state).

    Wraps a fully-validated :class:`StudyConfig` plus the resolved prompt pack
    and output directory. The dataclass is intentionally mutable so the runner
    can swap in filtered prompts after construction.
    """

    study: "StudyConfig"
    prompt_pack: list[Any]  # list[PromptSpec] — forward ref avoidance
    output_dir: Path

    # Convenience accessors so existing call sites that read these fields
    # continue to work without needing to spell out ``ctx.study.<X>``.
    @property
    def baseline_policy_name(self) -> str | None:
        return self.study.baseline_policy_name

    @property
    def repetitions(self) -> int:
        return self.study.runtime.repetitions

    @property
    def runtime(self):
        return self.study.runtime

    @property
    def policies(self):
        return self.study.policies

    @property
    def model(self):
        return self.study.model


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
