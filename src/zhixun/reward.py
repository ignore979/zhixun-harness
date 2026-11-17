"""Offline accuracy-floor ranking. Future labels are never runtime observations."""
from __future__ import annotations

import math


def agent_reward(accuracy, cost, *, flat_baseline, floor_ratio=0.99, lam=0.001):
    if (not all(math.isfinite(v) for v in [accuracy, cost, flat_baseline, floor_ratio, lam])
            or not 0 <= accuracy <= 1 or not 0 <= flat_baseline <= 1 or cost < 0
            or not 0 < floor_ratio <= 1 or lam < 0):
        raise ValueError("Invalid reward inputs")
    floor = flat_baseline * floor_ratio
    return -(floor - accuracy) * 100 if accuracy < floor else accuracy - lam * cost


def rank_guarded(outcomes, reference_action, *, floor_ratio=0.99, lam=0.001):
    references = [r for r in outcomes if r["action"] == reference_action]
    if len(references) != 1:
        raise ValueError("Each state requires its actual same-state flat reference outcome")
    reference = references[0]["next_metrics"]["accuracy"]
    ranked = []
    for row in outcomes:
        accuracy = row["next_metrics"]["accuracy"]
        ranked.append({**row, "flat_reference_accuracy": reference,
                       "floor": reference * floor_ratio,
                       "floor_feasible": accuracy >= reference * floor_ratio,
                       "reward": agent_reward(accuracy, row["epochs"], flat_baseline=reference,
                                               floor_ratio=floor_ratio, lam=lam)})
    # Feasibility first is an actual hard floor, independent of penalty magnitude/cost.
    return sorted(ranked, key=lambda r: (not r["floor_feasible"], -r["reward"], r["epochs"]))
