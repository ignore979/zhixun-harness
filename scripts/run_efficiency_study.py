"""Execute the fixed round-five ablation plan and preserve every real result."""
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zhixun.data import digest, write_json
from zhixun.efficiency import run_capacity
from zhixun.replay import run_replay

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["offline", "online"], default="offline")
args = parser.parse_args()
folder = ROOT / "artifacts" / f"efficiency-study-{datetime.now(UTC):%Y%m%dT%H%M%S}"
folder.mkdir(parents=True, exist_ok=False)
configs = ["replay.json", "replay_weighted.json", "replay_scheduled.json", "replay_memory.json",
           "efficiency_study.json"]
manifest = {"created_at": datetime.now(UTC).isoformat(), "status": "running",
            "evaluation_scope": "repeated_public_development_replay_not_independent_hidden_test",
            "config_sha256": {name: digest(ROOT / "configs" / name) for name in configs},
            "A": [], "B": []}
write_json(folder / "study.json", manifest)


def guard():
    if any(digest(ROOT / "configs" / name) != sha
           for name, sha in manifest["config_sha256"].items()):
        raise ValueError("Study configuration changed after registration")


try:
    for variant in ["control", "weighted", "scheduled", "memory"]:
        config = ROOT / "configs" / ("replay.json" if variant == "control"
                                      else f"replay_{variant}.json")
        for strategy in (["static", "always", "performance"] if variant == "control"
                         else ["always", "performance"]):
            guard()
            report = run_replay(strategy, config_path=config, mode=args.mode)
            row = {"variant": variant, "strategy": strategy, "experiment_id": report[
                "experiment_id"], "summary": report["summary"]}
            manifest["A"].append(row)
            write_json(folder / "study.json", manifest)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    for hidden, epochs, decay in [(24, 5, 1.0), (96, 5, 1.0), (96, 20, 0.5)]:
        guard()
        report = run_capacity(hidden=hidden, epochs=epochs, decay=decay, mode=args.mode)
        row = {"hidden": hidden, "epochs": epochs, "decay": decay,
               "experiment_id": report["experiment_id"], "accepted": report["accepted"],
               "before": report["validation_before"], "after": report["validation_after"],
               "selected": report["selected_metrics"],
               "migration_max_difference": report["migration_max_prediction_difference"]}
        manifest["B"].append(row)
        write_json(folder / "study.json", manifest)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    manifest["status"] = "completed"
    write_json(folder / "study.json", manifest)
    print(json.dumps({"study": str(folder.resolve()), "status": "completed"}), flush=True)
except BaseException as exc:
    manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    write_json(folder / "study.json", manifest)
    raise
