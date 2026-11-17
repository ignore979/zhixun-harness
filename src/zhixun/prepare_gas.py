"""Deterministic UCI source conversion, gated by exact reproduction of received batches 1/2."""
from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

from zhixun.data import ROOT, digest, write_json

SOURCE_SHA256 = "91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset"
LABELS = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia", 4: "Acetaldehyde",
          5: "Acetone", 6: "Toluene"}


def convert(raw: str, batch: int) -> list[dict]:
    rows = []
    for index, line in enumerate(raw.splitlines(), 1):
        fields = line.split()
        if len(fields) != 129 or int(fields[0]) not in LABELS:
            raise ValueError("UCI line must have one valid label and 128 explicit features")
        features = [0.0] * 128
        seen = set()
        for pair in fields[1:]:
            position, value = pair.split(":")
            position, value = int(position), float(value)
            if position in seen or not 1 <= position <= 128 or not math.isfinite(value):
                raise ValueError("Duplicate, out-of-range or nonfinite UCI feature")
            seen.add(position)
            features[position - 1] = value
        rows.append({"id": f"gas_b{batch:02d}_{index:06d}",
                     "inputs": {"features": features}, "answer": LABELS[int(fields[0])],
                     "meta": {"source_batch": batch, "source_index": index}})
    if not rows:
        raise ValueError("UCI batch is empty")
    return rows


def prepare_gas(archive: Path, *, root: Path = ROOT) -> dict:
    if archive.is_symlink() or digest(archive) != SOURCE_SHA256:
        raise ValueError("UCI archive differs from the audited local source")
    with zipfile.ZipFile(archive) as source:
        # Read named members, never extract untrusted paths.
        batches = {i: convert(source.read(f"Dataset/batch{i}.dat").decode("utf-8"), i)
                   for i in range(1, 11)}
    reproduced = {}
    for batch, name in [(1, "train.jsonl"), (2, "validation.jsonl")]:
        path = root / "data/A" / name
        supplied = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if supplied != batches[batch]:
            raise ValueError(f"Converted batch {batch} does not exactly reproduce {name}")
        reproduced[name] = {"sha256": digest(path), "rows": len(supplied),
                            "comparison": "id_inputs_answer_meta_full_equality"}
    destination = root / "data/replay-gas"
    destination.mkdir(parents=True, exist_ok=True)
    local_source = destination / "uci-gas-source.zip"
    if local_source.exists():
        if digest(local_source) != SOURCE_SHA256:
            raise ValueError("Refusing to overwrite a different replay source archive")
    else:
        shutil.copy2(archive, local_source)
    manifest = {"format": "zhixun.uci-local-replay.v1", "source_url": SOURCE_URL,
                "source_sha256": SOURCE_SHA256, "source_file": local_source.name,
                "scope": "UCI_original_labels; project_conversion; not_official_contest_stream",
                "reproduced_received_files": reproduced, "batches": []}
    for batch in range(3, 11):
        content = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                          for row in batches[batch])
        path = destination / f"batch{batch}.jsonl"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Refusing to overwrite a different derived replay batch")
        path.write_text(content, encoding="utf-8", newline="\n")
        manifest["batches"].append({"batch": batch, "file": path.name,
                                     "rows": len(batches[batch]), "sha256": digest(path),
                                     "source_member": f"Dataset/batch{batch}.dat"})
    write_json(destination / "manifest.json", manifest)
    return manifest


def verify_replay(root: Path = ROOT) -> dict:
    folder = root / "data/replay-gas"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["source_sha256"] != SOURCE_SHA256
            or manifest["source_file"] != "uci-gas-source.zip"
            or digest(folder / manifest["source_file"]) != SOURCE_SHA256):
        raise ValueError("Replay source integrity check failed")
    if [r["batch"] for r in manifest["batches"]] != list(range(3, 11)):
        raise ValueError("Replay batch order changed")
    for row in manifest["batches"]:
        if row["file"] != f"batch{row['batch']}.jsonl":
            raise ValueError("Unexpected replay path")
        path = folder / row["file"]
        if path.is_symlink() or digest(path) != row["sha256"]:
            raise ValueError("Derived replay batch integrity check failed")
    for name, row in manifest["reproduced_received_files"].items():
        if name not in {"train.jsonl", "validation.jsonl"} or digest(
            root / "data/A" / name
        ) != row["sha256"]:
            raise ValueError("Received public split changed after conversion")
    return manifest
