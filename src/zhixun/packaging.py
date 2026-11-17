from __future__ import annotations

import hashlib
import os
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from zhixun.data import ROOT

EXCLUDED = {"指导", ".git", "__pycache__", ".ruff_cache", ".pytest_cache"}
FORBIDDEN_NAMES = {".env", ".swanlab", ".netrc", ".pytrio", "credentials.json"}


def submission_files(root: Path) -> list[Path]:
    files = []
    for current, directories, names in os.walk(root, followlinks=False):
        folder = Path(current)
        if any(name in FORBIDDEN_NAMES for name in directories):
            raise ValueError(f"Credential directory cannot be submitted: {folder}")
        directories[:] = [name for name in directories if name not in EXCLUDED]
        if any((folder / name).is_symlink() for name in directories):
            raise ValueError("Linked directories cannot be included in a submission")
        for name in names:
            path = folder / name
            if name.endswith(".pyc"):
                continue
            if path.is_symlink() or name in FORBIDDEN_NAMES or name.startswith(".env."):
                raise ValueError(f"Unapproved link or credential file: {path.relative_to(root)}")
            if name.endswith((".key", ".token")):
                raise ValueError("Key/token files must not be included in a submission")
            if path.suffix in {".json", ".jsonl", ".md", ".py", ".ps1", ".txt",
                               ".log", ".yaml", ".yml"}:
                with path.open(encoding="utf-8-sig", errors="replace") as stream:
                    for line in stream:
                        if re.search(r"\b(?:sk-[A-Za-z0-9]{24,}|swlab_[A-Za-z0-9]{20,})\b", line):
                            raise ValueError(f"Potential secret found in {path.relative_to(root)}")
            files.append(path)
    return sorted(files)


def build_package(root: Path = ROOT) -> dict:
    files = submission_files(root)
    destination = root / "指导/交付"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"zhixun-school-{datetime.now(UTC):%Y%m%dT%H%M%S}.zip"
    with path.open("xb") as stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            archive.write(source, source.relative_to(root).as_posix())
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Submission ZIP CRC check failed")
        if any("指导" in Path(name).parts for name in archive.namelist()):
            raise ValueError("Guidance leaked into the submission")
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"archive": str(path), "sha256": checksum, "files": len(files),
            "guidance_excluded": True, "formal_evaluation_ready": False}
