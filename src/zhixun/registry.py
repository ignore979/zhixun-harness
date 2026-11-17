"""Atomic version/cursor/budget commits and a process lock released by the OS."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from zhixun.data import digest, write_json


def atomic_json(path: Path, value: dict) -> None:
    staged = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    write_json(staged, value)
    os.replace(staged, path)


@contextmanager
def process_lock(folder: Path):
    path = folder / ".stream.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            def acquire():
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise RuntimeError("Another process holds this stream session") from exc
        try:
            yield
        finally:
            stream.seek(0)
            release()


class ModelRegistry:
    def __init__(self, folder: Path, signature: dict, initial: Path | None = None):
        self.folder = folder.resolve()
        self.path = self.folder / "stream-state.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state["signature"] != signature:
                raise ValueError("Resume data, order, configuration or source code changed")
        else:
            if initial is None:
                raise ValueError("A new registry requires the original checkpoint")
            self.state = {
                "format": "zhixun.stream-state.v1", "signature": signature,
                "cursor": 0, "current": 0, "versions": [], "events": [],
                "training_seconds_charged": 0.0, "updates_charged": 0, "inflight": None,
            }
            import shutil

            shutil.copy2(initial, self.folder / "version-0.npz")
            self.register(0, self.folder / "version-0.npz", parent=None, adopted=True)
            self.save()
        for row in self.state["versions"]:
            self.checkpoint(row["version"])

    def save(self) -> None:
        atomic_json(self.path, self.state)

    def checkpoint(self, version: int) -> Path:
        row = next((r for r in self.state["versions"] if r["version"] == version), None)
        if row is None:
            raise ValueError("Unknown model version")
        stored = self.folder / row["path"]
        path = stored.resolve()
        if (not path.is_relative_to(self.folder) or stored.is_symlink()
                or digest(path) != row["sha256"]):
            raise ValueError("Registered checkpoint integrity check failed")
        return path

    def register(self, version: int, path: Path, *, parent: int | None, adopted: bool) -> None:
        if type(version) is not int or version < 0 or type(adopted) is not bool:
            raise ValueError("Invalid version metadata")
        if parent is not None and not any(r["version"] == parent for r in self.state["versions"]):
            raise ValueError("Parent version must already be registered")
        if any(r["version"] == version for r in self.state["versions"]):
            raise ValueError("Model versions are immutable and unique")
        if not path.resolve().is_relative_to(self.folder):
            raise ValueError("Checkpoint must be stored inside the stream session")
        self.state["versions"].append({
            "version": version, "path": path.relative_to(self.folder).as_posix(),
            "sha256": digest(path), "parent_version": parent, "adopted": adopted,
        })

    def rollback(self, version: int) -> None:
        row = next((r for r in self.state["versions"] if r["version"] == version), None)
        if row is None or not row["adopted"]:
            raise ValueError("Rollback requires a previously adopted version")
        self.checkpoint(version)
        self.state["current"] = version

    def recover_interruption(self) -> None:
        if self.state["inflight"] is not None:
            self.state["events"].append({
                "status": "interrupted_attempt", "reservation_retained": True,
                "attempt": self.state["inflight"],
            })
            self.state["inflight"] = None
            self.save()
