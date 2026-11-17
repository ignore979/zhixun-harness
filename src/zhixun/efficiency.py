"""Controlled public B continuation with explicit, function-preserving width migration."""
from __future__ import annotations

import csv
import math
import shutil
from datetime import UTC, datetime
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
from zhixun.model_extension import expand_hidden
from zhixun.tracking import Recorder
from zhixun.training_schedule import lr_at
from zhixun.vendor.model import Model


def run_capacity(*, hidden: int = 96, epochs: int = 5, learning_rate: float = 0.001,
                 decay: float = 1.0, mode="offline", project="zhixun-school", workspace=None,
                 root: Path = ROOT, output: Path | None = None) -> dict:
    if (hidden not in {24, 96} or type(epochs) is not int or not 1 <= epochs <= 20
            or not 0 < learning_rate <= 0.001 or not 0 < decay <= 1):
        raise ValueError("Capacity experiment exceeds the registered settings")
    started = monotonic()
    declared = verify_inputs("B", root)
    parent = root / "data/B/checkpoint.npz"
    base = Model.load(parent)
    train = load_dataset(root / "data/B/train.jsonl", base, "B")
    validation = load_dataset(root / "data/B/validation.jsonl", base, "B")
    verify_parent(base, "B", train)
    verify_split(train, validation)
    folder = (output or root / "artifacts") / (
        f"run-capacity-B-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    registration = {
        "experiment_id": folder.name, "task": "B", "format": "zhixun.capacity-training.v1",
        "created_at": datetime.now(UTC).isoformat(), "hidden": hidden, "epochs": epochs,
        "learning_rate": learning_rate, "lr_decay": decay, "dataset": declared,
        "parent_checkpoint_sha256": digest(parent), "numpy_version": np.__version__,
        "source_sha256": {p.relative_to(root).as_posix(): digest(p)
                           for p in sorted((root / "src").rglob("*.py"))},
        "study_sha256": digest(root / "configs/efficiency_study.json"),
        "evaluation_scope": "unchanged_public_B_validation; fixed_final_epoch",
        "width_migration": "exact_function_preserving; old_Adam_prefix_new_zero_moments",
        "migration_seed": 42, "outgoing_zero_sum_noise": 0.001,
    }
    write_json(folder / "registration.json", registration)
    recorder = None
    try:
        candidate = (expand_hidden(base, hidden) if hidden > base.meta["hidden"]
                     else Model.load(parent))
        migrated = candidate.predict(validation.x)
        maximum_difference = float(np.max(np.abs(migrated - base.predict(validation.x))))
        if not np.allclose(migrated, base.predict(validation.x), rtol=1e-12, atol=1e-10):
            raise ValueError("Width migration changed the inherited prediction function")
        candidate.save(folder / "starting-checkpoint.npz")
        before, _, _ = score(base, validation)
        recorder = Recorder(folder, registration, mode=mode, project=project, workspace=workspace)
        recorder.log(0, {**{f"validation/{k}": v for k, v in before.items()},
                         "migration/max_prediction_difference": maximum_difference,
                         "model/parameter_count": candidate.meta["parameter_count"]})
        steps, initial_epochs = candidate.meta["steps"], candidate.meta["epochs"]
        seconds, history = 0.0, []
        for epoch in range(epochs):
            if monotonic() - started >= 300:
                raise TimeoutError("Capacity training cooperative deadline reached")
            candidate.meta["learning_rate"] = lr_at(epoch, epochs, learning_rate, decay=decay)
            began = perf_counter()
            loss = candidate.fit(train.x, train.y, 1)[0]
            seconds += perf_counter() - began
            metrics, _, _ = score(candidate, validation)
            row = {"train/loss": float(loss),
                   "train/learning_rate": candidate.meta["learning_rate"],
                   "train/optimizer_steps": candidate.meta["steps"], "train/seconds": seconds,
                   **{f"validation/{k}": v for k, v in metrics.items()}}
            recorder.log(epoch + 1, row)
            history.append({"epoch": epoch + 1, **row})
        verify_inputs("B", root)
        if digest(parent) != registration["parent_checkpoint_sha256"]:
            raise ValueError("Original parent changed during capacity training")
        if (not np.array_equal(candidate.mean, base.mean)
                or not np.array_equal(candidate.scale, base.scale)
                or candidate.meta["steps"] != steps + epochs * math.ceil(
                    len(train.ids) / candidate.meta["batch_size"])
                or candidate.meta["epochs"] != initial_epochs + epochs):
            raise ValueError("Capacity continuation normalization or counters changed")
        after, _, _ = score(candidate, validation)
        accepted, gate = promotion(before, after, "B")
        candidate.save(folder / "candidate.npz")
        shutil.copy2(folder / "candidate.npz" if accepted else parent,
                     folder / "selected-checkpoint.npz")
        recorder.log(epochs + 1, {"decision/accepted": float(accepted)})
        with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        recorder.finish()
        report = {**registration, "status": "completed", "validation_before": before,
                  "validation_after": after, "selected_metrics": after if accepted else before,
                  "accepted": accepted, "gate": gate, "training_seconds": seconds,
                  "migration_max_prediction_difference": maximum_difference,
                  "parameter_count": candidate.meta["parameter_count"],
                  "initial_steps": steps, "final_steps": candidate.meta["steps"],
                  "train_rows": len(train.ids), "validation_rows": len(validation.ids),
                  "selected_checkpoint_sha256": digest(folder / "selected-checkpoint.npz"),
                  "tracking": recorder.reference(), "llm_calls": 0}
        write_json(folder / "run.json", report)
        return report
    except BaseException as exc:
        write_json(folder / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                                             "message": str(exc)})
        if recorder:
            try:
                recorder.finish(failed=True)
            except Exception:
                pass
        raise
