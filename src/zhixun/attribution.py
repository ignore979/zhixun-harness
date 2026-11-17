"""Statistical attribution; feature ranking is not a causal explanation."""
from __future__ import annotations

from zhixun.drift import DriftSignal


def attribute_drift(signal: DriftSignal, feature_names: list[str], *, top_k: int = 10) -> dict:
    if len(feature_names) != len(signal.per_feature_psi) or top_k < 1:
        raise ValueError("Feature names/top_k do not match drift schema")
    order = sorted(range(len(feature_names)), key=lambda i: (-signal.per_feature_psi[i], i))
    return {
        "interpretation": "distribution_association_not_causation",
        "top_features": [{"path": feature_names[i], "psi": float(signal.per_feature_psi[i])}
                         for i in order[:top_k]],
        "summary": ", ".join(signal.reasons) or "no_drift_detected",
    }


def attribute_performance(before: dict, after: dict) -> dict:
    labels = [label for label in before if label not in {"confusion_matrix", "metric_conventions"}]
    return {
        "per_class_f1_delta": {label: after[label]["f1"] - before[label]["f1"]
                               for label in labels},
        "zero_recall_classes": [label for label in labels
                                if after[label]["support"] > 0 and after[label]["recall"] == 0],
    }
