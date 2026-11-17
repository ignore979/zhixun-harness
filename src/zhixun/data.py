from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zhixun.vendor.data import read_dataset
from zhixun.vendor.model import Model

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


@dataclass(frozen=True)
class Dataset:
    ids: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    groups: frozenset[str]


def verify_inputs(profile: str, root: Path = ROOT) -> dict:
    if profile not in {"A", "B", "C"}:
        raise ValueError("Task must be A, B or C")
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    declared = manifest["profiles"][profile]
    if set(declared["files"]) != {
        "train.jsonl", "validation.jsonl", "checkpoint.npz", "profile.json"
    }:
        raise ValueError("Input manifest must name all four required source files")
    for name, expected in declared["files"].items():
        if name not in {"train.jsonl", "validation.jsonl", "checkpoint.npz", "profile.json"}:
            raise ValueError("Unexpected input manifest entry")
        path = root / "data" / profile / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected["sha256"]:
            raise ValueError(f"Input differs from the archived download: {profile}/{name}")
    return declared


def load_dataset(path: Path, model: Model, profile: str) -> Dataset:
    data = read_dataset(path, model.meta["task"], model.meta["classes"], model.meta["paths"])
    groups = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            meta = json.loads(line)["meta"]
            if profile == "A":
                group: Any = meta["source_batch"]
            elif profile == "B":
                group = (Path(meta["source"]).parent.as_posix(), meta["bearing"])
            else:
                group = (meta["source_subset"], meta["unit"])
            groups.add(json.dumps(group))
    return Dataset(tuple(data["ids"]), data["x"], data["y"], frozenset(groups))


def verify_split(train: Dataset, validation: Dataset) -> None:
    if set(train.ids) & set(validation.ids):
        raise ValueError("Training and validation IDs overlap")
    if train.groups & validation.groups:
        raise ValueError("Training and validation equipment/batch groups overlap")
    fingerprints = {hashlib.sha256(x.tobytes()).digest() for x in train.x}
    if any(hashlib.sha256(x.tobytes()).digest() in fingerprints for x in validation.x):
        raise ValueError("Exact model inputs overlap across train and validation")


def verify_parent(model: Model, profile: str, train: Dataset) -> None:
    if model.meta.get("profile") != profile:
        raise ValueError("Parent checkpoint belongs to another task")
    if (model.meta["task"] == "classification") != (profile == "A"):
        raise ValueError("Checkpoint task conflicts with the supplied task definition")
    scale = train.x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    if not np.allclose(model.mean, train.x.mean(axis=0), atol=1e-12, rtol=1e-12):
        raise ValueError("Parent mean is not fitted on the named training set")
    if not np.allclose(model.scale, scale, atol=1e-12, rtol=1e-12):
        raise ValueError("Parent scale is not fitted on the named training set")
    if profile != "A" and (
        not np.isclose(model.meta["target_mean"], train.y.mean())
        or not np.isclose(model.meta["target_scale"], max(float(train.y.std()), 1.0))
    ):
        raise ValueError("Parent target scaling does not match training")
