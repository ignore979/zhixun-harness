from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from zhixun.data import ROOT, write_json


def record_cloud(experiment: str, url: str, reviewer: str, root: Path = ROOT) -> dict:
    if (
        Path(experiment).name != experiment
        or not (root / "artifacts" / experiment / "run.json").is_file()
    ):
        raise ValueError("Name an existing completed local experiment")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https" or parsed.hostname != "swanlab.cn"
        or not parsed.path.startswith("/@") or "/runs/" not in parsed.path
        or parsed.username or parsed.password or not reviewer.strip()
    ):
        raise ValueError("Provide a real SwanLab experiment page and the reviewer name")
    folder = root / "evidence"
    folder.mkdir(exist_ok=True)
    path = folder / "cloud_records.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    record = {
        "cloud_url": urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")),
        "reviewed_by": reviewer.strip(), "reviewed_at": datetime.now(UTC).isoformat(),
        "verification_method": "manual_browser_review_by_submitter",
        "independently_verified_by_program": False,
    }
    records[experiment] = record
    write_json(path, records)
    return {"experiment_id": experiment, **record}
