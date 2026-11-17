"""Version ensembles with explicit temporal eligibility and prequential reliability."""
from __future__ import annotations

import copy

import numpy as np

from zhixun.vendor.model import Model


def combine_predictions(predictions, weights, task):
    values, weights = np.asarray(predictions), np.asarray(weights, dtype=float)
    if (not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or len(values) != len(weights)):
        raise ValueError("Invalid ensemble predictions or weights")
    if task == "classification":
        if (values.ndim != 3 or weights.shape not in {(len(values),),
                                                     (len(values), values.shape[2])}
                or np.any(values < 0) or not np.allclose(values.sum(axis=2), 1)):
            raise ValueError("Classification ensemble requires probability matrices")
        if weights.ndim == 1:
            weights = np.broadcast_to(weights[:, None], (len(values), values.shape[2])).copy()
        if np.any(weights.sum(axis=0) <= 0):
            raise ValueError("Every class needs positive ensemble weight")
        weights = weights / weights.sum(axis=0)
        out = (values * weights[:, None, :]).sum(axis=0)
        return out / out.sum(axis=1, keepdims=True)
    if task != "regression" or values.ndim != 2 or weights.shape != (len(values),):
        raise ValueError("Invalid regression ensemble shape or task")
    if weights.sum() <= 0:
        raise ValueError("Ensemble needs positive total weight")
    return np.maximum(0, np.average(values, axis=0, weights=weights))


class EnsemblePredictor:
    """Only adopted, eligible versions. Current-label feedback is a separate method.

    Reliability uses only predictions actually emitted BEFORE those labels were
    released, never retrospective scores on the expert's own training data.
    """
    def __init__(self, registry, *, k=5, decay=.9, mode="recency", allowed_versions=None,
                 current_version=None, window=2, shrinkage=20.0, history=None):
        if (k is not None and (type(k) is not int or not 1 <= k <= 20)):
            raise ValueError("Ensemble k must be 1..20 or None for all")
        if (not np.isfinite(decay) or not 0 < decay <= 1
                or mode not in {"recency", "class_reliability"}
                or type(window) is not int or window < 1
                or not np.isfinite(shrinkage) or shrinkage <= 0):
            raise ValueError("Invalid ensemble settings")
        self.registry, self.k, self.decay, self.mode = registry, k, decay, mode
        self.allowed_versions = None if allowed_versions is None else set(allowed_versions)
        self.current = registry.state["current"] if current_version is None else current_version
        self.window, self.shrinkage = window, shrinkage
        self.history = copy.deepcopy(history or {})
        self.pending = None

    def _adopted(self):
        rows = [r for r in self.registry.state["versions"] if r["adopted"] and (
            self.allowed_versions is None or r["version"] in self.allowed_versions)]
        current = next((r for r in rows if r["version"] == self.current), None)
        if current is None:
            raise ValueError("Current version must be previously adopted and temporally eligible")
        # Incumbent is always included; k=1 remains the exact control after rollbacks.
        previous = [r for r in rows if r["version"] != self.current]
        return (previous if self.k is None else previous[-(self.k - 1):]
                if self.k > 1 else []) + [current]

    def weights(self, rows, classes):
        recency = self.decay ** np.arange(len(rows) - 1, -1, -1, dtype=float)
        if self.mode == "recency":
            return recency
        scores = []
        for row in rows:
            records = self.history.get(str(row["version"]), [])[-self.window:]
            tp = np.zeros(classes)
            denominators = np.zeros(classes)
            for record in records:
                matrix = np.asarray(record["confusion_matrix"], dtype=float)
                if matrix.shape != (classes, classes) or np.any(matrix < 0):
                    raise ValueError("Invalid expert reliability state")
                tp += 2 * np.diag(matrix)
                denominators += matrix.sum(axis=0) + matrix.sum(axis=1)
            scores.append((tp + .5 * self.shrinkage) / (denominators + self.shrinkage))
        return np.asarray(scores) * recency[:, None]

    def predict(self, x, task=None, *, transform=None, batch=None):
        if self.pending is not None:
            raise ValueError("Release or discard the previous prediction before another batch")
        rows, predictions, models = self._adopted(), [], []
        for row in rows:
            model = Model.load(self.registry.checkpoint(row["version"]))
            if task is not None and model.meta["task"] != task:
                raise ValueError("Ensemble task does not match checkpoint")
            if models and (model.meta["task"] != models[0].meta["task"]
                           or model.meta["paths"] != models[0].meta["paths"]
                           or model.meta.get("classes") != models[0].meta.get("classes")
                           or not np.array_equal(model.mean, models[0].mean)
                           or not np.array_equal(model.scale, models[0].scale)):
                raise ValueError("Ensemble schema or frozen normalization differs")
            if transform is None:
                predicted = model.predict(x)
            else:
                if model.meta["task"] != "classification" or transform.shape != x.shape:
                    raise ValueError("Feature adaptation requires a matching classification matrix")
                _, logits = model.forward(transform)
                logits -= logits.max(axis=1, keepdims=True)
                predicted = np.exp(logits)
                predicted /= predicted.sum(axis=1, keepdims=True)
            predictions.append(predicted)
            models.append(model)
        task = models[0].meta["task"]
        if task != "classification" and self.mode != "recency":
            raise ValueError("Per-class reliability requires classification")
        weights = self.weights(rows, len(models[0].meta.get("classes", [])))
        output = combine_predictions(predictions, weights, task)
        self.pending = {"batch": batch, "versions": [r["version"] for r in rows],
                        "predictions": predictions, "weights": weights.tolist(), "task": task}
        self.last_audit = {"versions": [r["version"] for r in rows],
                           "checkpoint_sha256": [r["sha256"] for r in rows],
                           "weights": weights.tolist(), "forward_passes": len(rows),
                           "current_labels_used": False, "reliability_window": self.window,
                           "parameter_bytes": sum(a.nbytes for m in models
                                                  for a in m.params.values())}
        return output

    def release(self, y, *, batch):
        pending = self.pending
        if pending is None or pending["batch"] != batch:
            raise ValueError("Labels must match a previously emitted batch")
        if pending["task"] == "classification":
            y = np.asarray(y)
            classes = pending["predictions"][0].shape[1]
            if (y.shape != (len(pending["predictions"][0]),) or y.dtype.kind not in "iu"
                    or y.min() < 0 or y.max() >= classes):
                raise ValueError("Invalid released labels")
            for version, prediction in zip(pending["versions"], pending["predictions"],
                                            strict=True):
                history = self.history.setdefault(str(version), [])
                if history and batch <= history[-1]["batch"]:
                    raise ValueError("Expert feedback cannot repeat or go backward")
                matrix = np.zeros((classes, classes), dtype=int)
                np.add.at(matrix, (y, prediction.argmax(axis=1)), 1)
                history.append({"batch": batch, "confusion_matrix": matrix.tolist()})
                self.history[str(version)] = history[-self.window:]
        self.pending = None


def temporal_versions(events, index):
    """A final registry is unsafe by itself: future adopted versions are excluded."""
    if not 0 <= index < len(events):
        raise ValueError("Invalid temporal cursor")
    if [e["batch_index"] for e in events[:index + 1]] != list(range(index + 1)):
        raise ValueError("Source trajectory order changed")
    versions = {0}
    versions.update(e["selected_version"] for e in events[:index])
    if events[index]["parent_version"] not in versions:
        raise ValueError("Source parent was not committed before prediction")
    return versions
