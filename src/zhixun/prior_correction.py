"""Prediction-only adaptation. No current/future labels or weight mutation in predict."""
from __future__ import annotations

import numpy as np


def symmetric_power(matrix, exponent):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, 1e-8)**exponent) @ vectors.T


def class_prior(y, classes, smoothing=1.0):
    y = np.asarray(y)
    if (y.ndim != 1 or not len(y) or y.dtype.kind not in "iu"
            or y.min() < 0 or y.max() >= classes or smoothing <= 0):
        raise ValueError("Invalid prior labels or smoothing")
    counts = np.bincount(y, minlength=classes).astype(float) + smoothing
    return counts / counts.sum()


def reweight(probabilities, target, source, strength=1.0):
    probabilities, target, source = map(np.asarray, (probabilities, target, source))
    if (probabilities.ndim != 2 or source.shape != target.shape
            or source.shape != (probabilities.shape[1],) or not 0 <= strength <= 1
            or not all(np.isfinite(a).all() for a in [probabilities, target, source])
            or np.any(probabilities < 0) or np.any(target <= 0) or np.any(source <= 0)
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Invalid prior correction inputs")
    # Log-space calculation avoids overflow/underflow for highly confident classifiers.
    logits = np.log(np.maximum(probabilities, 1e-300)) + strength * np.log(target / source)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


class PriorCorrectedModel:
    """Historical-label prior heuristic, deliberately not called BBSE."""
    def __init__(self, model, n_classes=None, *, window=3, strength=1.0, smoothing=1e-6):
        self.model, self.meta = model, model.meta
        self.n_classes = n_classes or len(model.meta["classes"])
        if type(window) is not int or window < 1 or not 0 <= strength <= 1 or smoothing <= 0:
            raise ValueError("Invalid prior settings")
        self.window, self.strength, self.smoothing = window, strength, smoothing
        self.recent, self.p_train, self.released_batches = [], None, []

    def fit_prior(self, y_initial):
        if self.p_train is not None:
            raise ValueError("Initial prior is immutable")
        self.p_train = class_prior(y_initial, self.n_classes, self.smoothing)

    def release(self, y_batch, *, batch):
        if type(batch) is not int or batch < 2 or (
                self.released_batches and batch != self.released_batches[-1] + 1):
            raise ValueError("Released prior batches must be sequential")
        if not self.released_batches and batch != 2:
            raise ValueError("First released batch must be batch2")
        # Validate and copy counts, never retain a caller-owned label array.
        class_prior(y_batch, self.n_classes, self.smoothing)
        self.recent.append(np.bincount(y_batch, minlength=self.n_classes).astype(float))
        self.recent = self.recent[-self.window:]
        self.released_batches.append(batch)

    def p_test(self):
        if self.p_train is None:
            raise ValueError("Initial prior has not been fitted")
        if not self.recent:
            return self.p_train.copy()
        counts = np.sum(self.recent, axis=0) + self.smoothing
        return counts / counts.sum()

    def predict(self, x):
        return reweight(self.model.predict(x), self.p_test(), self.p_train, self.strength)


def project_simplex(value):
    """Euclidean projection onto a probability simplex (Duchi et al. algorithm)."""
    ordered = np.sort(value)[::-1]
    offsets = (np.cumsum(ordered) - 1) / np.arange(1, len(value) + 1)
    index = np.flatnonzero(ordered > offsets)[-1]
    return np.maximum(value - offsets[index], 0)


def bbse_prior(source_probabilities, source_y, target_probabilities, *, ridge=0.01):
    """Regularized soft BBSE: solve C q = mean(target predictions) on the simplex.

    C columns are conditional source predictions. Missing source classes are rejected;
    ill-conditioning and residual are exposed rather than claimed to certify label shift.
    """
    p, t = np.asarray(source_probabilities), np.asarray(target_probabilities)
    source_y = np.asarray(source_y)
    if p.ndim != 2 or t.ndim != 2:
        raise ValueError("BBSE requires probability matrices")
    classes = p.shape[1]
    source = class_prior(source_y, classes, 1.0)
    if (p.ndim != 2 or t.ndim != 2 or p.shape[0] != len(source_y) or t.shape[1] != classes
            or not len(t) or not np.isfinite(p).all() or not np.isfinite(t).all()
            or np.any(p < 0) or np.any(t < 0) or ridge <= 0
            or not np.allclose(p.sum(axis=1), 1) or not np.allclose(t.sum(axis=1), 1)):
        raise ValueError("Invalid BBSE source or unlabeled target probabilities")
    if np.any(np.bincount(source_y, minlength=classes) == 0):
        raise ValueError("BBSE requires each source class")
    conditional = np.column_stack([p[source_y == c].mean(axis=0) for c in range(classes)])
    target_mean = t.mean(axis=0)
    hessian = conditional.T @ conditional + ridge * np.eye(classes)
    gradient_offset = conditional.T @ target_mean + ridge * source
    step = 1 / np.linalg.eigvalsh(hessian).max()
    q = source.copy()
    for _ in range(500):
        updated = project_simplex(q - step * (hessian @ q - gradient_offset))
        if np.max(np.abs(updated - q)) < 1e-10:
            q = updated
            break
        q = updated
    q = np.maximum(q, 1e-6)
    q /= q.sum()
    return q, {"conditional_matrix": conditional.tolist(),
               "condition_number": float(min(np.linalg.cond(conditional), 1e16)),
               "mixture_residual_l2": float(np.linalg.norm(conditional @ q - target_mean)),
               "ridge": ridge, "target_labels_used": False}


def coral_probabilities(model, source_x, target_x, *, strength=0.5, ridge=1.0):
    """Target-to-source CORAL variant after frozen normalization, including centering.

    Transductive: statistics use the whole current UNLABELED feature batch. This does
    not refit model.mean/scale and is not proof of formal competition compatibility.
    """
    if not 0 <= strength <= 1 or ridge <= 0 or min(len(source_x), len(target_x)) < 2:
        raise ValueError("Invalid CORAL settings or batch size")
    source, target = model.normalized(source_x), model.normalized(target_x)

    eye = np.eye(source.shape[1])
    cs = np.cov(source, rowvar=False) + ridge * eye
    ct = np.cov(target, rowvar=False) + ridge * eye
    aligned = ((target - target.mean(axis=0)) @ symmetric_power(ct, -0.5)
               @ symmetric_power(cs, 0.5)
               + source.mean(axis=0))
    transformed = (1 - strength) * target + strength * aligned
    _, logits = model.forward(np.clip(transformed, -10, 10))
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


class CoralAdapter:
    """Portable source-only geometry; current batch statistics are computed without labels."""
    def __init__(self, source_mean, source_root, normalization_mean, normalization_scale,
                 *, strength=.25, ridge=1., source_sha256=""):
        arrays = [source_mean, source_root, normalization_mean, normalization_scale]
        self.source_mean, self.source_root, self.mean, self.scale = [
            np.array(a, dtype=float, copy=True) for a in arrays]
        dim = len(self.mean)
        if (self.mean.shape != (dim,) or self.scale.shape != (dim,)
                or self.source_mean.shape != (dim,) or self.source_root.shape != (dim, dim)
                or not all(np.isfinite(a).all() for a in arrays)
                or np.any(self.scale <= 0) or not 0 <= strength <= 1 or ridge <= 0):
            raise ValueError("Invalid CORAL source artifact")
        self.strength, self.ridge, self.source_sha256 = strength, ridge, source_sha256

    @classmethod
    def from_source(cls, model, x, *, strength=.25, ridge=1., source_sha256=""):
        if model.meta["task"] != "classification" or len(x) < 2:
            raise ValueError("CORAL requires a classification source with at least two rows")
        z = model.normalized(x)
        covariance = np.cov(z, rowvar=False) + ridge * np.eye(z.shape[1])
        return cls(z.mean(axis=0), symmetric_power(covariance, .5), model.mean, model.scale,
                   strength=strength, ridge=ridge, source_sha256=source_sha256)

    def transform(self, model, x):
        if (model.meta["task"] != "classification" or len(x) < 2
                or not np.array_equal(model.mean, self.mean)
                or not np.array_equal(model.scale, self.scale)):
            raise ValueError("Adapter requires matching frozen normalization and a whole batch")
        z = model.normalized(x)
        covariance = np.cov(z, rowvar=False) + self.ridge * np.eye(z.shape[1])
        aligned = ((z - z.mean(axis=0)) @ symmetric_power(covariance, -.5) @ self.source_root
                   + self.source_mean)
        transformed = (1 - self.strength) * z + self.strength * aligned
        return np.clip(transformed, -10, 10)

    def predict(self, model, x):
        _, logits = model.forward(self.transform(model, x))
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def save(self, path):
        with path.open("xb") as stream:
            np.savez_compressed(stream, source_mean=self.source_mean, source_root=self.source_root,
                                normalization_mean=self.mean, normalization_scale=self.scale,
                                strength=self.strength, ridge=self.ridge,
                                source_sha256=np.array(self.source_sha256))

    @classmethod
    def load(cls, path):
        if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Invalid adapter file")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"source_mean", "source_root", "normalization_mean",
                                     "normalization_scale", "strength", "ridge", "source_sha256"}:
                raise ValueError("Invalid adapter fields")
            return cls(arrays['source_mean'], arrays['source_root'], arrays['normalization_mean'],
                       arrays['normalization_scale'], strength=float(arrays['strength']),
                       ridge=float(arrays['ridge']), source_sha256=str(arrays['source_sha256']))
