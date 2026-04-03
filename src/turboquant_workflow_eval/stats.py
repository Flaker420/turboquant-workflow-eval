from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import torch


@dataclass
class OnlineScalarStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, values: Any) -> None:
        tensor = values if torch.is_tensor(values) else torch.tensor([values], dtype=torch.float32)
        flat = tensor.detach().float().reshape(-1)
        if flat.numel() == 0:
            return
        self.count += int(flat.numel())
        self.total += float(flat.sum().item())
        self.total_sq += float((flat * flat).sum().item())
        current_min = float(flat.min().item())
        current_max = float(flat.max().item())
        self.minimum = current_min if self.minimum is None else min(self.minimum, current_min)
        self.maximum = current_max if self.maximum is None else max(self.maximum, current_max)

    @property
    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count

    @property
    def std(self) -> float | None:
        if self.count == 0:
            return None
        mean = self.total / self.count
        variance = max(0.0, (self.total_sq / self.count) - (mean * mean))
        return sqrt(variance)

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass
class ProjectionTensorStats:
    vector_norm: OnlineScalarStats = field(default_factory=OnlineScalarStats)
    abs_value: OnlineScalarStats = field(default_factory=OnlineScalarStats)
    channel_rms_mean: OnlineScalarStats = field(default_factory=OnlineScalarStats)
    channel_rms_max: OnlineScalarStats = field(default_factory=OnlineScalarStats)
    shape_examples: list[list[int]] = field(default_factory=list)

    def update_from_output(self, output) -> None:
        tensor = take_first_tensor(output)
        if tensor is None:
            return
        tensor = tensor.detach().float()
        if tensor.numel() == 0:
            return
        self.abs_value.update(tensor.abs())
        last_dim = tensor.shape[-1]
        flat = tensor.reshape(-1, last_dim)
        self.vector_norm.update(torch.linalg.vector_norm(flat, dim=-1))
        channel_rms = torch.sqrt(torch.mean(flat * flat, dim=0))
        self.channel_rms_mean.update(channel_rms.mean())
        self.channel_rms_max.update(channel_rms.max())
        shape = list(tensor.shape)
        if shape not in self.shape_examples and len(self.shape_examples) < 5:
            self.shape_examples.append(shape)

    def to_dict(self) -> dict:
        return {
            "vector_norm": self.vector_norm.to_dict(),
            "abs_value": self.abs_value.to_dict(),
            "channel_rms_mean": self.channel_rms_mean.to_dict(),
            "channel_rms_max": self.channel_rms_max.to_dict(),
            "shape_examples": self.shape_examples,
        }


def take_first_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    if isinstance(output, dict):
        for item in output.values():
            if torch.is_tensor(item):
                return item
    return None
