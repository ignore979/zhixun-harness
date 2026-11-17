"""Deterministic feedback policy. These limits are local settings, not contest rules."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from zhixun.drift import DriftSignal


@dataclass(frozen=True)
class Budget:
    max_training_seconds: float = 900.0
    max_seconds_per_round: float = 300.0
    max_updates: int = 3
    max_llm_calls: int = 0

    def __post_init__(self) -> None:
        values = (self.max_training_seconds, self.max_seconds_per_round)
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values)
                or not all(math.isfinite(v) for v in
                    (self.max_training_seconds, self.max_seconds_per_round))
                or not 0 < self.max_training_seconds <= 900
                or not 0 < self.max_seconds_per_round <= 300
                or type(self.max_updates) is not int or not 0 <= self.max_updates <= 3
                or type(self.max_llm_calls) is not int or self.max_llm_calls != 0):
            raise ValueError("Invalid local budget; this policy makes no LLM calls")


@dataclass(frozen=True)
class UpdatePlan:
    action: str
    epochs: int = 0
    learning_rate: float = 0.0
    rationale: str = ""
    severity: str = "none"

    def __post_init__(self) -> None:
        if self.action not in {"hold", "retrain", "rollback"}:
            raise ValueError("Unknown policy action")
        if self.action == "retrain" and (
            type(self.epochs) is not int or not 1 <= self.epochs <= 20
            or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.001
        ):
            raise ValueError("Policy training proposal exceeds local limits")
        if self.action != "retrain" and (self.epochs != 0 or self.learning_rate != 0):
            raise ValueError("Nontraining actions must not request optimizer work")


class Policy(Protocol):
    def decide(self, signal: DriftSignal, diagnosis: dict, budget: Budget,
               history: list[dict]) -> UpdatePlan: ...


class RulePolicy:
    def decide(self, signal: DriftSignal, diagnosis: dict, budget: Budget,
               history: list[dict]) -> UpdatePlan:
        if not signal.detected:
            return UpdatePlan("hold", rationale="no_drift_skip")
        severity = "severe" if signal.mean_psi >= 0.25 else "mild"
        epochs, rate = (5, 0.001) if severity == "severe" else (2, 0.0005)
        rationale = "input_only_drift_trigger; confidence_alone_does_not_prove_rollback"
        previous = [r for r in history if r.get("plan", {}).get("severity") == severity
                    and r.get("plan", {}).get("action") == "retrain"]
        if previous and not previous[-1].get("accepted", False):
            epochs = 5
            rate = previous[-1]["plan"]["learning_rate"] / 2
            rationale += "; previous_same_severity_rejected: reduce_lr_and_use_5_epochs"
        return UpdatePlan("retrain", epochs, rate, rationale, severity)
