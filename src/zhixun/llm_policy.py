"""Untrusted LLM output is data: accept only bounded structured proposals."""
from __future__ import annotations

import json
import math


def parse_decision(raw: str, *, rollback_versions: list[int] | None = None) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate policy JSON key")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != {
        "action", "epochs", "learning_rate", "target_version"
    }:
        raise ValueError("Policy must return exactly the declared JSON fields")
    action, epochs, rate, target = (value[k] for k in
                                    ["action", "epochs", "learning_rate", "target_version"])
    if action not in {"hold", "retrain", "rollback"} or type(epochs) is not int:
        raise ValueError("Invalid policy action or epochs")
    if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate):
        raise ValueError("Invalid policy learning rate")
    if action == "retrain":
        if not 1 <= epochs <= 20 or not 0 < rate <= 0.001 or target is not None:
            raise ValueError("Training proposal is outside the allowed action space")
    elif epochs != 0 or rate != 0:
        raise ValueError("Nontraining proposal requested optimizer work")
    if action == "rollback":
        if type(target) is not int or target not in (rollback_versions or []):
            raise ValueError("Rollback target was not previously adopted")
    elif target is not None:
        raise ValueError("Only rollback may specify a target version")
    return value
