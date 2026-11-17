from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from zhixun.data import ROOT, write_json


def parse_cloud_url(url: str) -> tuple[str, str]:
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "https" or parsed.hostname != "swanlab.cn"
        or parsed.username or parsed.password or parsed.port not in (None, 443)
        or len(parts) < 4 or not parts[0].startswith("@") or parts[2] != "runs"
        or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", v) for v in
                   [parts[0][1:], parts[1], parts[3]])
        or any(v in {".", ".."} for v in [parts[0][1:], parts[1], parts[3]])
    ):
        raise ValueError("Expected a SwanLab HTTPS experiment URL")
    path = "/".join([parts[0][1:], parts[1], parts[3]])
    canonical = urlunsplit(("https", "swanlab.cn", "/" + "/".join(parts[:4]), "", ""))
    return path, canonical


def normalize_config(config: dict) -> dict:
    """Unwrap the observed SwanLab value/desc/sort config envelope."""
    return {
        key: value["value"] if isinstance(value, dict) and "value" in value
        and ("desc" in value or "sort" in value) else value
        for key, value in config.items()
    }


def expected_points(history: list[dict]) -> dict[str, dict[int, float]]:
    result: dict[str, dict[int, float]] = {}
    for row in history:
        for key, value in row["metrics"].items():
            points = result.setdefault(key, {})
            if row["step"] in points or not math.isfinite(value):
                raise ValueError("Local metric history contains duplicate or nonfinite points")
            points[row["step"]] = value
    return result


def compare_config(expected: dict, actual: dict) -> list[str]:
    normalized_null_fields = []

    def compare(value, remote, path):
        # SDK null compatibility is explicit and one-way, including nested upgrade settings.
        if value is None and remote == {}:
            normalized_null_fields.append(path)
        elif isinstance(value, dict):
            if not isinstance(remote, dict) or set(value) != set(remote):
                raise ValueError(f"Cloud configuration differs: {path}")
            for key in value:
                compare(value[key], remote[key], f"{path}.{key}")
        elif isinstance(value, list):
            if not isinstance(remote, list) or len(value) != len(remote):
                raise ValueError(f"Cloud configuration differs: {path}")
            for index, (left, right) in enumerate(zip(value, remote, strict=True)):
                compare(left, right, f"{path}[{index}]")
        elif remote != value:
            raise ValueError(f"Cloud configuration differs: {path}")

    for key in expected:
        if key not in actual:
            raise ValueError(f"Cloud configuration differs: missing {key}")
        compare(expected[key], actual[key], key)
    return normalized_null_fields


def compare_points(expected: dict[str, dict[int, float]], response: dict) -> int:
    actual = {}
    for metric in response["list"]:
        if metric["key"] in actual:
            raise ValueError("Cloud returned a duplicate metric key")
        points = {}
        for point in metric["metrics"]:
            step, value = point["step"], point["value"]
            if type(step) is not int or step in points or not math.isfinite(value):
                raise ValueError("Cloud returned duplicate or invalid scalar points")
            points[step] = value
        actual[metric["key"]] = points
    checked = 0
    for key, points in expected.items():
        if key not in actual or set(points) != set(actual[key]):
            raise ValueError(f"Cloud steps differ from local history: {key}")
        for step, value in points.items():
            if not math.isclose(value, actual[key][step], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Cloud scalar differs from local history: {key}, step {step}")
            checked += 1
    return checked


def verify_cloud_runs(
    urls: list[str] | None = None, *, root: Path = ROOT, api=None,
) -> dict:
    evidence = root / "evidence"
    path = evidence / "cloud_records.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not urls:
        urls = [record["cloud_url"] for record in existing.values()]
    if not urls:
        raise ValueError("No saved cloud links; supply --url for each uploaded experiment")
    parsed = [parse_cloud_url(url) for url in urls]
    if len({item[0] for item in parsed}) != len(parsed):
        raise ValueError("Cloud verification URLs must be distinct")
    if api is None:
        import swanlab

        api = swanlab.Api()
    verified = {}
    for remote_path, canonical in parsed:
        run = api.run(remote_path)
        name = run.name
        if not re.fullmatch(r"(?:run-|pytrio-)[A-Za-z0-9_-]{1,90}", name):
            raise ValueError("Cloud experiment name is not a local experiment identifier")
        folder = root / "artifacts" / name
        if folder.is_symlink():
            raise ValueError("Local experiment folder must not be a link")
        report = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        registration = json.loads((folder / "registration.json").read_text(encoding="utf-8"))
        if report.get("status") != "completed":
            raise ValueError("Only completed local experiments can be verified")
        raw_config = run.profile["config"]
        if isinstance(raw_config, str):
            raw_config = json.loads(raw_config)
        config = normalize_config(raw_config)
        fields = list(registration)
        normalized_null_fields = compare_config(registration, config)
        if config.get("experiment_id", name) != name:
            raise ValueError("Cloud name and registered experiment ID disagree")
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()]
        expected = expected_points(history)
        response = run.metrics(keys=list(expected), all=True)
        checked = compare_points(expected, response)
        verified[name] = {
            "cloud_url": canonical,
            "verified_at": datetime.now(UTC).isoformat(),
            "verification_method": "swanlab_official_sdk_config_and_scalar_comparison",
            "independently_verified_by_program": True,
            "configuration_fields_checked": fields,
            "sdk_null_to_empty_mapping_fields": normalized_null_fields,
            "scalar_points_checked": checked,
            "metric_keys_checked": list(expected),
        }
    existing.update(verified)
    evidence.mkdir(parents=True, exist_ok=True)
    staged = evidence / "cloud_records.tmp"
    write_json(staged, existing)
    staged.replace(path)
    completed = {p.parent.name for p in (root / "artifacts").glob("*/run.json")
                 if json.loads(p.read_text(encoding="utf-8")).get("status") == "completed"}
    status_path = evidence / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    checked_names = {name for name, record in existing.items()
                     if record.get("independently_verified_by_program") is True}
    status.update({
        "cloud_upload_verified": bool(completed) and completed <= checked_names,
        "cloud_verification_as_of": datetime.now(UTC).isoformat(),
        "cloud_verified_local_experiments": sorted(completed & checked_names),
    })
    write_json(status_path, status)
    return {"verified_experiments": verified,
            "all_current_completed_runs_verified": status["cloud_upload_verified"],
            "remote_project_mutated": False}
