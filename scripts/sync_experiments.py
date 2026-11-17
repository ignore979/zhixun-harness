"""Upload an explicit, frozen experiment queue without resubmitting training."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=[0, 1], default=0)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    names = json.loads(args.queue.read_text(encoding="utf-8"))
    if len(names) != len(set(names)) or args.worker >= args.workers:
        raise ValueError("Queue contains duplicates or invalid worker index")
    destination = ROOT / "evidence/cloud_sync"
    destination.mkdir(exist_ok=True)
    for name in names[args.worker::args.workers]:
        if not re.fullmatch(r"(?:run-|pytrio-)[A-Za-z0-9_-]{1,90}", name):
            raise ValueError("Invalid experiment name")
        saved = destination / f"{name}.json"
        if saved.exists():
            continue
        folder = ROOT / "artifacts" / name
        report = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        if report["status"] != "completed" or report["tracking"]["mode"] != "offline":
            raise ValueError("Queue requires completed offline runs only")
        runs = list((folder / "swanlog").glob("run-*"))
        if len(runs) != 1:
            raise ValueError("Expected exactly one immutable SwanLab source log")
        log = ROOT / "指导/临时" / f"sync-{name}.log"
        with log.open("wb") as output:
            result = subprocess.run([
                sys.executable, "-B", str(ROOT / "run.py"), "swanlab", "sync", str(runs[0]),
                "--workspace", args.workspace, "--project", args.project,
            ], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"SwanLab upload failed; inspect {log}")
        pattern = (r"https://swanlab.cn/@" + re.escape(args.workspace) + "/"
                   + re.escape(args.project) + r"/runs/[a-zA-Z0-9]+")
        urls = set(re.findall(pattern, log.read_text(encoding="utf-8")))
        if len(urls) != 1:
            raise ValueError("Missing or ambiguous SwanLab destination URL")
        record = {"experiment_id": name, "cloud_url": urls.pop(),
                  "verification_status": "uploaded_not_yet_scalar_verified"}
        saved.write_text(json.dumps(record) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
