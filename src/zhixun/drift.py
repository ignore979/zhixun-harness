"""Input-only PSI diagnostics against the immutable original training set."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zhixun.vendor.model import Model


def matrix(x: np.ndarray, features: int | None = None) -> None:
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Drift input must be a nonempty finite feature matrix")
    if features is not None and x.shape[1] != features:
        raise ValueError("Drift feature schema changed")


@dataclass(frozen=True)
class DriftReference:
    # Interior edges only; histogram tails extend to infinity at evaluation time.
    bin_edges: tuple[np.ndarray, ...]
    expected_counts: tuple[np.ndarray, ...]
    reference_confidence: float | None
    fitted_on: str = "original_public_train_only"

    def summary(self) -> dict:
        return {"fitted_on": self.fitted_on,
                "bin_edges": [v.tolist() for v in self.bin_edges],
                "expected_counts": [v.tolist() for v in self.expected_counts],
                "reference_confidence": self.reference_confidence}


@dataclass(frozen=True)
class DriftSignal:
    mean_psi: float
    per_feature_psi: np.ndarray
    n_features_over: int
    confidence_drop: float | None
    detected: bool
    reasons: tuple[str, ...]

    def summary(self) -> dict:
        return {**self.__dict__, "per_feature_psi": self.per_feature_psi.tolist()}


def counts(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)


def fit_reference(x: np.ndarray, model: Model, *, n_bins: int = 10) -> DriftReference:
    matrix(x)
    if type(n_bins) is not int or not 2 <= n_bins <= 100:
        raise ValueError("n_bins must be an integer in 2..100")
    edges, frequencies = [], []
    for column in x.T:
        if np.min(column) == np.max(column):
            # A separate central bin detects changes on either side of a constant.
            value = column[0]
            boundary = np.array([np.nextafter(value, -np.inf), np.nextafter(value, np.inf)])
            boundary = boundary[np.isfinite(boundary)]
        else:
            boundary = np.unique(np.quantile(column, np.arange(1, n_bins) / n_bins))
        edges.append(boundary)
        frequencies.append(counts(column, boundary))
    confidence = None
    if model.meta["task"] == "classification":
        confidence = float(model.predict(x).max(axis=1).mean())
    return DriftReference(tuple(edges), tuple(frequencies), confidence)


def psi_from_reference(ref: DriftReference, x: np.ndarray) -> np.ndarray:
    matrix(x, len(ref.bin_edges))
    values = []
    for column, edges, expected in zip(x.T, ref.bin_edges, ref.expected_counts, strict=True):
        actual = counts(column, edges)
        # Smooth observed counts, then renormalize. Do not assume uniform quantile bins.
        p = np.maximum(expected / expected.sum(), 1e-6)
        q = np.maximum(actual / actual.sum(), 1e-6)
        p, q = p / p.sum(), q / q.sum()
        values.append(float(np.sum((q - p) * np.log(q / p))))
    return np.array(values)


def detect_drift(
    model: Model, ref: DriftReference, x: np.ndarray, *, psi_threshold: float = 0.25,
    conf_drop_threshold: float = 0.05,
) -> DriftSignal:
    if not np.isfinite([psi_threshold, conf_drop_threshold]).all() or not (
        0 < psi_threshold <= 10 and 0 < conf_drop_threshold <= 1
    ):
        raise ValueError("Invalid drift thresholds")
    psi = psi_from_reference(ref, x)
    over = int(np.sum(psi > psi_threshold))
    drop = None
    reasons = []
    if psi.mean() > psi_threshold or over / len(psi) > 0.1:
        reasons.append("input_distribution_shift")
    if ref.reference_confidence is not None:
        drop = ref.reference_confidence - float(model.predict(x).max(axis=1).mean())
        if drop > conf_drop_threshold:
            reasons.append("confidence_drop_requires_validation")
    return DriftSignal(float(psi.mean()), psi, over, drop, bool(reasons), tuple(reasons))
