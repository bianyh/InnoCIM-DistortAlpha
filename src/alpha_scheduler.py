from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class AlphaScheduler:
    mode: str

    def sample(self, epoch: int, step: int) -> float:
        raise NotImplementedError

    def state_dict(self) -> dict[str, float | str | int]:
        return {"mode": self.mode}


@dataclass
class NoAlphaScheduler(AlphaScheduler):
    def __init__(self) -> None:
        super().__init__(mode="none")

    def sample(self, epoch: int, step: int) -> float:
        return 0.0


@dataclass
class FixedAlphaScheduler(AlphaScheduler):
    alpha: float

    def __init__(self, alpha: float) -> None:
        super().__init__(mode="fixed")
        self.alpha = alpha

    def sample(self, epoch: int, step: int) -> float:
        return float(self.alpha)

    def state_dict(self) -> dict[str, float | str | int]:
        state = super().state_dict()
        state["alpha"] = self.alpha
        return state


@dataclass
class UniformAlphaScheduler(AlphaScheduler):
    low: float
    high: float

    def __init__(self, low: float, high: float) -> None:
        super().__init__(mode="uniform")
        self.low = low
        self.high = high

    def sample(self, epoch: int, step: int) -> float:
        return random.uniform(self.low, self.high)

    def state_dict(self) -> dict[str, float | str | int]:
        state = super().state_dict()
        state.update({"low": self.low, "high": self.high})
        return state


@dataclass
class CurriculumAlphaScheduler(AlphaScheduler):
    max_abs: float
    warmup_epochs: int
    min_abs: float = 0.0

    def __init__(self, max_abs: float, warmup_epochs: int, min_abs: float = 0.0) -> None:
        super().__init__(mode="curriculum")
        self.max_abs = max_abs
        self.warmup_epochs = max(1, warmup_epochs)
        self.min_abs = min_abs

    def current_abs(self, epoch: int) -> float:
        progress = min(1.0, max(0.0, float(epoch + 1) / float(self.warmup_epochs)))
        return self.min_abs + (self.max_abs - self.min_abs) * progress

    def sample(self, epoch: int, step: int) -> float:
        bound = self.current_abs(epoch)
        return random.uniform(-bound, bound)

    def state_dict(self) -> dict[str, float | str | int]:
        state = super().state_dict()
        state.update(
            {
                "max_abs": self.max_abs,
                "warmup_epochs": self.warmup_epochs,
                "min_abs": self.min_abs,
            }
        )
        return state


def build_alpha_scheduler(
    mode: str,
    alpha: float = 0.0,
    low: float = -1.0,
    high: float = 1.0,
    max_abs: float = 1.0,
    warmup_epochs: int = 80,
    min_abs: float = 0.0,
) -> AlphaScheduler:
    if mode == "none":
        return NoAlphaScheduler()
    if mode == "fixed":
        return FixedAlphaScheduler(alpha)
    if mode == "uniform":
        return UniformAlphaScheduler(low, high)
    if mode == "curriculum":
        return CurriculumAlphaScheduler(max_abs=max_abs, warmup_epochs=warmup_epochs, min_abs=min_abs)
    raise ValueError(f"Unsupported alpha scheduler mode: {mode}")
