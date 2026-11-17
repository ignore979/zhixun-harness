"""Label-released training history. No normalization is fitted here."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from zhixun.data import Dataset, verify_split


@dataclass
class TrainingPool:
    chunks: list[Dataset] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)

    @classmethod
    def from_dataset(cls, data: Dataset) -> TrainingPool:
        return cls([data], [{"batch_id": "initial-train", "samples": len(data.ids),
                             "label_release": "initial_public_training_labels"}])

    def dataset(self) -> Dataset:
        x, y = self.as_train()
        return Dataset(tuple(i for d in self.chunks for i in d.ids), x, y,
                       frozenset(g for d in self.chunks for g in d.groups))

    def verify_future(self, data: Dataset) -> None:
        verify_split(self.dataset(), data)

    def append(self, data: Dataset, *, batch_id: str, label_frac: float = 1.0) -> None:
        if label_frac != 1.0:
            raise ValueError("Partial-label selection is not implemented; do not invent labels")
        if not batch_id or batch_id in {r["batch_id"] for r in self.provenance}:
            raise ValueError("Repeated or empty training-pool batch identity")
        if not len(data.ids) or len(data.ids) != len(data.x) or len(data.ids) != len(data.y):
            raise ValueError("Empty or inconsistent training-pool batch")
        self.verify_future(data)
        self.chunks.append(data)
        self.provenance.append({"batch_id": batch_id, "samples": len(data.ids),
                                "label_release": "after_current_batch_decision"})

    def as_train(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.chunks:
            raise ValueError("Training pool must start with supplied training data")
        return (np.concatenate([d.x for d in self.chunks]),
                np.concatenate([d.y for d in self.chunks]))

    def summary(self) -> dict:
        data = self.dataset()
        checksum = hashlib.sha256()
        checksum.update(data.x.tobytes())
        checksum.update(data.y.tobytes())
        return {"rows": len(data.ids), "groups": len(data.groups),
                "content_sha256": checksum.hexdigest(), "provenance": list(self.provenance)}

    def sample(self, *, cap_rows: int | None = None, seed: int = 42) -> tuple[Dataset, dict]:
        data = self.dataset()
        total = len(data.ids)
        if cap_rows is not None and (type(cap_rows) is not int or cap_rows < len(self.chunks)):
            raise ValueError("Sampling cap must retain at least one row per historical chunk")
        selected = np.arange(total)
        quotas = [len(c.ids) for c in self.chunks]
        if cap_rows is not None and total > cap_rows:
            quotas = [0] * len(self.chunks)
            remaining = cap_rows
            while remaining:
                active = [i for i, c in enumerate(self.chunks) if quotas[i] < len(c.ids)]
                allocation = max(1, remaining // len(active))
                for i in active:
                    addition = min(allocation, len(self.chunks[i].ids) - quotas[i], remaining)
                    quotas[i] += addition
                    remaining -= addition
            indices, offset = [], 0
            for i, (chunk, quota) in enumerate(zip(self.chunks, quotas, strict=True)):
                rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
                indices.extend(offset + np.sort(rng.choice(len(chunk.ids), quota, replace=False)))
                offset += len(chunk.ids)
            selected = np.asarray(indices, dtype=int)
        sampled = Dataset(tuple(data.ids[i] for i in selected), data.x[selected], data.y[selected],
                          data.groups)
        return sampled, {"pool_rows": total, "sampled_rows": len(selected), "cap_rows": cap_rows,
                         "seed": seed, "per_chunk_rows": quotas,
                         "sample_ids_sha256": hashlib.sha256(
                             "\n".join(sampled.ids).encode("utf-8")).hexdigest()}
