"""Stage the official SDK image without datasets, historical weights or sampling clients."""

import hashlib
import json
import shutil
from pathlib import Path

from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "指导/临时/official-image-context-20261117-v1"
SDK_SHA = "c676c8beeef69cece9cbec5a633d05dbb9832eb22b6f761913f2315bd1f85133"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare():
    source = ROOT / "指导/原始资料/评测平台_20261117/contest-harness-sdk-1.0.2.tar.gz"
    if sha(source) != SDK_SHA:
        raise ValueError("Official SDK archive digest mismatch")
    if CONTEXT.exists():
        raise ValueError("Preserve the existing context; create a new version")
    CONTEXT.mkdir(parents=True)
    names = [
        "__init__.py",
        "data.py",
        "metrics.py",
        "model_extension.py",
        "tracking.py",
        "contest_runtime.py",
        "vendor/__init__.py",
        "vendor/data.py",
        "vendor/model.py",
    ]
    for name in names:
        dest = CONTEXT / "src/zhixun" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src/zhixun" / name, dest)
    for name, relative in [
        ("contest_submission.py", "submission/contest_submission.py"),
        ("harness.py", "submission/official_sdk_1_0_2/template/harness.py"),
        ("Dockerfile", "submission/Dockerfile.official"),
        ("NOTICE.md", "NOTICE.md"),
    ]:
        shutil.copy2(ROOT / relative, CONTEXT / name)
    locks = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line:
            name, version = line.strip().split("==")
            locks.append((canonicalize_name(name), version))
    wheel_dir = CONTEXT / "wheels"
    wheel_dir.mkdir()
    available = {}
    for p in (ROOT / "指导/临时/trial-linux-cp312-wheels").glob("*.whl"):
        name, version, _, _ = parse_wheel_filename(p.name)
        available[(canonicalize_name(name), str(version))] = p
    for pair in locks:
        shutil.copy2(available[pair], wheel_dir / available[pair].name)
    sdk_wheel = ROOT / "指导/临时/official-sdk-wheels/contest_harness-1.0.2-py3-none-any.whl"
    shutil.copy2(sdk_wheel, wheel_dir / sdk_wheel.name)
    (wheel_dir / "requirements.txt").write_text(
        (ROOT / "requirements.txt").read_text(encoding="utf-8") + "\ncontest-harness==1.0.2\n",
        encoding="utf-8",
    )
    metadata = {
        "sdk_version": "1.0.2",
        "sdk_source": "https://rsi-index.115.zone/downloads/contest-harness-sdk-1.0.2.tar.gz",
        "sdk_archive_sha256": SDK_SHA,
        "protocol": "contest.harness.workspace.v2",
        "model_access": "host_gateway_rpc",
        "local_data_or_weights_bundled": False,
        "external_pytrio_client_bundled": False,
    }
    (CONTEXT / "SDK_SOURCE.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    files = {p.relative_to(CONTEXT).as_posix(): sha(p) for p in CONTEXT.rglob("*") if p.is_file()}
    (CONTEXT / "CONTEXT_SHA256.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"context": str(CONTEXT), "files": len(files), "core_pins": len(locks)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    prepare()
