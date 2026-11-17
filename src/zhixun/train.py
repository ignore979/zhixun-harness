from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import monotonic, perf_counter
from uuid import uuid4

import numpy as np

from zhixun.data import (
    ROOT,
    digest,
    load_dataset,
    verify_inputs,
    verify_parent,
    verify_split,
    write_json,
)
from zhixun.metrics import promotion, score
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model


@dataclass(frozen=True)
class RoundPlan:
    profile: str
    epochs: int = 5
    learning_rate: float = 0.001
    max_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.profile not in {"A", "B", "C"}:
            raise ValueError("Task must be A, B or C")
        if type(self.epochs) is not int or not 1 <= self.epochs <= 50:
            raise ValueError("Epochs must be an integer in 1..50")
        if not 0 < self.learning_rate <= 0.01 or not 0 < self.max_seconds <= 300:
            raise ValueError("Learning rate or wall-clock budget exceeds local safety limits")


def run_round(
    plan: RoundPlan, *, mode: str = "offline", project: str = "zhixun-school",
    workspace: str | None = None, parent: Path | None = None,
    output: Path | None = None, root: Path = ROOT,
) -> dict:
    started = monotonic()
    declared = verify_inputs(plan.profile, root)
    folder = (output or root / "artifacts") / (
        f"run-{plan.profile}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    inputs = root / "data" / plan.profile
    parent_path = (parent or inputs / "checkpoint.npz").resolve()
    source_hashes = {
        p.relative_to(root).as_posix(): digest(p) for p in sorted((root / "src").rglob("*.py"))
    }
    registration = {
        "format": "zhixun.experiment.v1", "experiment_id": folder.name,
        "created_at": datetime.now(UTC).isoformat(), "profile": plan.profile,
        "protocol": "public-cpu-continuation-round1-v1",
        "plan_file_sha256": digest(root / "configs/round1.json"),
        "epochs": plan.epochs, "learning_rate": plan.learning_rate,
        "max_seconds": plan.max_seconds,
        "parent_checkpoint_sha256": digest(parent_path),
        "dataset": declared, "source_sha256": source_hashes,
        "numpy_version": np.__version__, "swanlab_version": version("swanlab"),
        "evaluation_scope": "public_development_validation; not official hidden evaluation",
        "ai_assisted_development": True,
    }
    write_json(folder / "registration.json", registration)
    recorder = None
    try:
        incumbent = Model.load(parent_path)
        candidate = Model.load(parent_path)
        train = load_dataset(inputs / "train.jsonl", incumbent, plan.profile)
        validation = load_dataset(inputs / "validation.jsonl", incumbent, plan.profile)
        verify_parent(incumbent, plan.profile, train)
        verify_split(train, validation)
        recorder = Recorder(folder, registration, mode=mode, project=project, workspace=workspace)
        before, before_details, _ = score(incumbent, validation)
        recorder.log(0, {f"validation/{k}": v for k, v in before.items()})
        candidate.meta["learning_rate"] = plan.learning_rate
        initial_steps, initial_epochs = candidate.meta["steps"], candidate.meta["epochs"]
        training_seconds = 0.0
        epoch_rows = []
        for epoch in range(1, plan.epochs + 1):
            if monotonic() - started >= plan.max_seconds:
                raise TimeoutError("Round wall-clock budget reached before training")
            training_started = perf_counter()
            loss = candidate.fit(train.x, train.y, 1)[0]
            training_seconds += perf_counter() - training_started
            metrics, _, _ = score(candidate, validation)
            logged = {
                "train/loss": float(loss), "train/seconds": training_seconds,
                "train/optimizer_steps": float(candidate.meta["steps"]),
                **{f"validation/{k}": value for k, value in metrics.items()},
            }
            recorder.log(epoch, logged)
            epoch_rows.append({"epoch": epoch, **logged})
        if monotonic() - started >= plan.max_seconds:
            raise TimeoutError("Round exceeded its cooperative wall-clock budget")
        verify_inputs(plan.profile, root)
        if digest(parent_path) != registration["parent_checkpoint_sha256"]:
            raise ValueError("Parent checkpoint changed during training")
        if not np.array_equal(candidate.mean, incumbent.mean) or not np.array_equal(
            candidate.scale, incumbent.scale
        ):
            raise ValueError("Continuation changed the inherited normalization")
        expected_steps = initial_steps + plan.epochs * (
            (len(train.ids) + candidate.meta["batch_size"] - 1) // candidate.meta["batch_size"]
        )
        if candidate.meta["steps"] != expected_steps:
            raise ValueError("Continuation did not preserve the optimizer step counter")
        after, after_details, predicted = score(candidate, validation)
        accepted, gate = promotion(before, after, plan.profile)
        candidate.save(folder / "candidate.npz")
        selected_path = folder / "checkpoint.npz"
        shutil.copy2(folder / "candidate.npz" if accepted else parent_path, selected_path)
        recorder.log(plan.epochs + 1, {"decision/accepted": float(accepted)})
        recorder.finish()
        report = {
            **registration, "status": "completed", "train_rows": len(train.ids),
            "validation_rows": len(validation.ids), "train_groups": len(train.groups),
            "validation_groups": len(validation.groups), "validation_before": before,
            "validation_after": after, "accepted": accepted, "gate": gate,
            "selected": "candidate" if accepted else "parent",
            "selected_metrics": after if accepted else before,
            "per_class_before": before_details, "per_class_after": after_details,
            "initial_steps": initial_steps, "final_steps": candidate.meta["steps"],
            "initial_epochs": initial_epochs, "final_epochs": candidate.meta["epochs"],
            "selected_checkpoint_sha256": digest(selected_path),
            "candidate_checkpoint_sha256": digest(folder / "candidate.npz"),
            "resources": {"training_seconds": training_seconds,
                          "elapsed_seconds": monotonic() - started,
                          "candidate_models": 1, "llm_calls": 0,
                          "cpu_threads_requested": 1, "target_environment_measured": False},
            "tracking": recorder.reference(),
            "artifact_directory": folder.relative_to(root).as_posix()
            if folder.is_relative_to(root) else str(folder.resolve()),
        }
        with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["epoch", *sorted(set(epoch_rows[0]) - {"epoch"})]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(epoch_rows)
        with (folder / "candidate_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for index, identity in enumerate(validation.ids):
                stream.write(json.dumps({"id": identity, "prediction": predicted[index].tolist()},
                                        allow_nan=False) + "\n")
        write_json(folder / "run.json", report)
        write_json((output or root / "artifacts") / f"latest-{plan.profile}.json", {
            "experiment_id": folder.name,
            "directory": folder.relative_to(output or root / "artifacts").as_posix(),
        })
        return report
    except BaseException as exc:
        write_json(folder / "failure.json", {
            "status": "failed", "error_type": type(exc).__name__, "message": str(exc),
            "registration": registration,
        })
        if recorder is not None:
            try:
                recorder.finish(failed=True)
            except Exception:
                pass
        raise
