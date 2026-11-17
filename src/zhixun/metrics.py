from __future__ import annotations

import numpy as np

from zhixun.data import Dataset
from zhixun.vendor.model import Model


def score(model: Model, data: Dataset) -> tuple[dict[str, float], dict, np.ndarray]:
    predicted = model.predict(data.x)
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Prediction contains nonfinite values")
    if model.meta["task"] == "regression":
        error = predicted - data.y
        return {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
        }, {}, predicted
    metrics, details = classification_metrics(predicted, data.y, model.meta["classes"])
    return metrics, details, predicted


def classification_metrics(predicted, y, classes):
    """Fixed-class F1; G-mean/balanced accuracy use classes present in this batch."""
    predicted, y = np.asarray(predicted), np.asarray(y)
    if (predicted.shape != (len(y), len(classes)) or not len(y) or y.ndim != 1
            or y.dtype.kind not in "iu" or y.min() < 0 or y.max() >= len(classes)
            or not np.isfinite(predicted).all() or np.any(predicted < 0)
            or not np.allclose(predicted.sum(axis=1), 1, rtol=1e-7, atol=1e-9)):
        raise ValueError("Invalid classification probabilities or labels")
    labels = predicted.argmax(axis=1)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(matrix, (y, labels), 1)
    details = {}
    f1s = []
    for index, label in enumerate(classes):
        true_positive = int(matrix[index, index])
        actual = int(matrix[index].sum())
        proposed = int(matrix[:, index].sum())
        precision = true_positive / proposed if proposed else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * true_positive / (actual + proposed) if actual + proposed else 0.0
        f1s.append(f1)
        details[label] = {"support": actual, "precision": precision, "recall": recall, "f1": f1}
    details["confusion_matrix"] = matrix.tolist()
    recalls = np.array([details[c]["recall"] for c in classes])
    present = matrix.sum(axis=1) > 0
    actual, proposed = matrix.sum(axis=1), matrix.sum(axis=0)
    accuracy = float(np.trace(matrix) / len(y))
    expected = float(actual @ proposed / float(len(y))**2)
    # Degenerate one-class agreement has no chance-corrected information; record zero.
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else 0.0
    g_mean = float(np.exp(np.log(recalls[present]).mean())) if np.all(
        recalls[present] > 0) else 0.0
    details["metric_conventions"] = {
        "macro_f1": "all_declared_classes_absent_class_zero",
        "g_mean_and_balanced_accuracy": "only_classes_with_positive_true_support",
        "kappa_expected_agreement_one": "zero_no_chance_corrected_information",
        "present_classes": [c for c, active in zip(classes, present, strict=True) if active],
    }
    return {
        "accuracy": accuracy, "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(recalls[present].mean()), "g_mean": g_mean,
        "cohen_kappa": float(kappa),
    }, details


def promotion(before: dict[str, float], after: dict[str, float], profile: str) -> tuple[bool, str]:
    if profile == "A":
        passed = after["macro_f1"] > before["macro_f1"] and after["accuracy"] >= before["accuracy"]
        return passed, "macro_f1 must increase and accuracy must not decrease"
    passed = after["rmse"] < before["rmse"] and after["mae"] <= before["mae"]
    return passed, "rmse must decrease and mae must not increase"
