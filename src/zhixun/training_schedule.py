"""A bounded training amount, after an update has already been requested by the policy."""
from __future__ import annotations

import math


def schedule_epochs(mean_psi: float, performance_drop: float, pool_rows: int,
                    *, base_epochs=5, max_epochs=20, drop_threshold=0.05) -> int:
    if (not all(math.isfinite(v) for v in (mean_psi, performance_drop, drop_threshold))
            or mean_psi < 0 or pool_rows < 1 or drop_threshold <= 0
            or type(base_epochs) is not int or not 1 <= base_epochs <= max_epochs <= 20):
        raise ValueError("Invalid epoch schedule inputs")
    # A true performance decline must not be vetoed by low PSI.
    severity = min(2.0, max(0.4, mean_psi / 0.25, performance_drop / drop_threshold))
    factor = min(1.5, 1 + pool_rows / 20000)
    return min(max_epochs, max(2, int(base_epochs * severity * factor)))


def lr_at(epoch: int, total_epochs: int, base_lr: float, *, decay=0.5) -> float:
    if (type(epoch) is not int or not 0 <= epoch < total_epochs
            or not 0 < base_lr <= 0.01 or not 0 < decay <= 1):
        raise ValueError("Invalid learning-rate schedule")
    return base_lr if epoch < total_epochs * 0.6 else base_lr * decay
