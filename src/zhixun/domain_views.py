"""Follow-up domain views; no target labels in transformations, no vendor mutation."""
from __future__ import annotations

import numpy as np

from zhixun.prior_correction import symmetric_power


def domain_transform(model, source, target_x, spec, *, past=(), current_batch=None):
    source_z, z = model.normalized(source.x), model.normalized(target_x)
    mode, alpha = spec["kind"], spec.get("strength", .25)
    if not 0 <= alpha <= 1 or len(z) < 2:
        raise ValueError("Invalid domain view strength or batch size")
    ridge = spec.get("ridge", 1.)
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Domain ridge must be finite and positive")
    if mode == "identity":
        return z
    source_mean, target_mean = source_z.mean(axis=0), z.mean(axis=0)
    if mode == "past_class_shift":
        if current_batch is None or any(batch >= current_batch for batch, _ in past):
            raise ValueError("Class shift must use strictly earlier released batches")
        if not past:
            return z
        selected = past[-spec.get("window", 2):]
        past_z = np.concatenate([model.normalized(d.x) for _, d in selected])
        labels = np.concatenate([d.y for _, d in selected])
        offsets = []
        for c in range(len(model.meta["classes"])):
            active = labels == c
            if active.any():
                offset = past_z[active].mean(axis=0) - source_z[source.y == c].mean(axis=0)
                offsets.append(offset * active.sum() / (active.sum() + 20))
        shift = np.mean(offsets, axis=0)
        return np.clip(z - alpha * np.clip(shift, -3, 3), -10, 10)
    if mode == "diagonal":
        ratio = np.sqrt((source_z.var(axis=0, ddof=1) + ridge) /
                        (z.var(axis=0, ddof=1) + ridge))
        aligned = (z - target_mean) * ratio + source_mean
        return np.clip((1 - alpha) * z + alpha * aligned, -10, 10)
    if mode == "block_coral":
        size = spec.get("block_size", 8)
        if type(size) is not int or size < 2 or source_z.shape[1] % size:
            raise ValueError("Block size must divide the existing feature dimension")
        aligned = np.empty_like(z)
        # Index blocks are an engineering partition, not an invented official sensor schema.
        for start in range(0, z.shape[1], size):
            sl = slice(start, start + size)
            cs = np.cov(source_z[:, sl], rowvar=False) + ridge * np.eye(size)
            ct = np.cov(z[:, sl], rowvar=False) + ridge * np.eye(size)
            aligned[:, sl] = ((z[:, sl] - target_mean[sl]) @ symmetric_power(ct, -.5)
                              @ symmetric_power(cs, .5) + source_mean[sl])
        return np.clip((1 - alpha) * z + alpha * aligned, -10, 10)
    if mode in {"split_coral", "coral"}:
        cs = np.cov(source_z, rowvar=False) + ridge * np.eye(z.shape[1])
        ct = np.cov(z, rowvar=False) + ridge * np.eye(z.shape[1])
        centered = z - target_mean
        covariance_delta = centered @ symmetric_power(ct, -.5) @ symmetric_power(cs, .5) - centered
        mean_strength = spec.get("mean_strength", alpha)
        if not 0 <= mean_strength <= 1:
            raise ValueError("Invalid mean-alignment strength")
        return np.clip(z + alpha * covariance_delta + mean_strength * (
            source_mean - target_mean), -10, 10)
    raise ValueError("Unknown domain view")


def view_probabilities(model, source, target_x, spec, *, past=(), current_batch=None):
    if spec["kind"] == "multi_view":
        base = model.predict(target_x)
        corrected = view_probabilities(model, source, target_x,
                                        {"kind": "coral", "strength": .25, "ridge": 1.})
        return .5 * (base + corrected)
    transformed = domain_transform(model, source, target_x, spec, past=past,
                                   current_batch=current_batch)
    _, logits = model.forward(transformed)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


class BlockCoralAdapter:
    """Portable index-block source geometry, fitted once on original batch1 features."""
    def __init__(self, source_mean, roots, mean, scale, *, strength=.25, ridge=1.,
                 block_size=8, source_sha256=''):
        self.source_mean, self.roots, self.mean, self.scale = [np.array(v, copy=True)
            for v in [source_mean, roots, mean, scale]]
        dimension = len(self.mean)
        if (type(block_size) is not int or block_size < 2 or dimension % block_size
                or self.scale.shape != (dimension,) or self.source_mean.shape != (dimension,)
                or self.roots.shape != (dimension // block_size, block_size, block_size)
                or not all(np.isfinite(v).all() for v in [self.source_mean, self.roots,
                                                        self.mean, self.scale])
                or np.any(self.scale <= 0) or not 0 <= strength <= 1
                or not np.isfinite(ridge) or ridge <= 0):
            raise ValueError('Invalid block CORAL artifact')
        self.strength, self.ridge, self.block_size = strength, ridge, block_size
        self.source_sha256 = source_sha256

    @classmethod
    def from_source(cls, model, x, *, strength=.25, ridge=1., block_size=8, source_sha256=''):
        z = model.normalized(x)
        if model.meta['task'] != 'classification' or len(z) < 2 or z.shape[1] % block_size:
            raise ValueError('Invalid source block geometry')
        roots = [symmetric_power(np.cov(z[:, j:j + block_size], rowvar=False)
                                 + ridge * np.eye(block_size), .5)
                 for j in range(0, z.shape[1], block_size)]
        return cls(z.mean(axis=0), roots, model.mean, model.scale, strength=strength, ridge=ridge,
                   block_size=block_size, source_sha256=source_sha256)

    def transform(self, model, x):
        if (model.meta['task'] != 'classification' or len(x) < 2
                or not np.array_equal(model.mean, self.mean)
                or not np.array_equal(model.scale, self.scale)):
            raise ValueError('Block adapter needs matching frozen normalization and a whole batch')
        z = model.normalized(x)
        aligned = np.empty_like(z)
        for block, start in enumerate(range(0, z.shape[1], self.block_size)):
            sl = slice(start, start + self.block_size)
            mean = z[:, sl].mean(axis=0)
            covariance = np.cov(z[:, sl], rowvar=False) + self.ridge * np.eye(self.block_size)
            aligned[:, sl] = ((z[:, sl] - mean) @ symmetric_power(covariance, -.5)
                              @ self.roots[block] + self.source_mean[sl])
        return np.clip((1 - self.strength) * z + self.strength * aligned, -10, 10)

    def predict(self, model, x):
        _, logits = model.forward(self.transform(model, x))
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def save(self, path):
        with path.open('xb') as stream:
            np.savez_compressed(stream, adapter_kind=np.array('block_coral_v1'),
                                source_mean=self.source_mean, roots=self.roots, mean=self.mean,
                                scale=self.scale, strength=self.strength, ridge=self.ridge,
                                block_size=self.block_size, source_sha256=self.source_sha256)

    @classmethod
    def load(cls, path):
        if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError('Invalid block adapter file')
        with np.load(path, allow_pickle=False) as arrays:
            expected = {'adapter_kind', 'source_mean', 'roots', 'mean', 'scale', 'strength',
                        'ridge', 'block_size', 'source_sha256'}
            if set(arrays.files) != expected or str(arrays['adapter_kind']) != 'block_coral_v1':
                raise ValueError('Invalid block adapter fields')
            return cls(arrays['source_mean'], arrays['roots'], arrays['mean'], arrays['scale'],
                       strength=float(arrays['strength']), ridge=float(arrays['ridge']),
                       block_size=int(arrays['block_size']),
                       source_sha256=str(arrays['source_sha256']))


def load_domain_adapter(path):
    from zhixun.prior_correction import CoralAdapter

    if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('Invalid domain adapter file')
    with np.load(path, allow_pickle=False) as arrays:
        kind = str(arrays['adapter_kind']) if 'adapter_kind' in arrays.files else 'legacy_coral'
    if kind == 'block_coral_v1':
        return BlockCoralAdapter.load(path)
    if kind == 'legacy_coral':
        return CoralAdapter.load(path)
    raise ValueError('Unknown domain adapter kind')
