"""Resumable development loop. The CLI replays ONE unchanged public validation file.

EvaluationBatch is an internal contract, not a claimed organizer streaming API.
Only the original public training set is ever passed to Model.fit.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, perf_counter
from uuid import uuid4

import numpy as np

from zhixun.attribution import attribute_drift, attribute_performance
from zhixun.data import (
    ROOT,
    Dataset,
    digest,
    load_dataset,
    verify_inputs,
    verify_parent,
    verify_split,
    write_json,
)
from zhixun.drift import detect_drift, fit_reference
from zhixun.metrics import promotion, score
from zhixun.policy import Budget, RulePolicy, UpdatePlan
from zhixun.registry import ModelRegistry, atomic_json, process_lock
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model


@dataclass(frozen=True)
class EvaluationBatch:
    index: int
    identity: str
    data: Dataset
    source_sha256: str


def verify_sequence(batches: list[EvaluationBatch], train: Dataset) -> None:
    seen_ids, seen_groups, seen_batches = set(), set(), set()
    if not batches:
        raise ValueError("At least one explicit evaluation batch is required")
    for index, batch in enumerate(batches):
        if type(batch.index) is not int or batch.index != index or not batch.identity:
            raise ValueError("Batch order must be contiguous, starting at zero")
        if batch.identity in seen_batches or seen_ids.intersection(batch.data.ids):
            raise ValueError("Repeated evaluation batch or sample ID")
        if seen_groups.intersection(batch.data.groups):
            raise ValueError("Equipment/batch groups repeat across evaluation batches")
        if not len(batch.data.ids) or len(batch.data.ids) != len(batch.data.x):
            raise ValueError("Empty or inconsistent evaluation batch")
        verify_split(train, batch.data)
        seen_batches.add(batch.identity)
        seen_ids.update(batch.data.ids)
        seen_groups.update(batch.data.groups)


def load_config(path: Path) -> tuple[dict, Budget]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"protocol", "psi_threshold", "conf_drop_threshold", "policy", "budget",
                "max_rollback_evaluations", "validation_role", "training_role"}
    if set(config) != required or config["policy"] != "rule":
        raise ValueError("Unexpected stream configuration; only the rule policy is implemented")
    if (config["protocol"] != "public-validation-drift-loop-v1"
            or config["training_role"] != "original_public_train_only"
            or config["validation_role"] != "unchanged_public_development_validation"):
        raise ValueError("Unsupported data-role or protocol declaration")
    for key, limit in [("psi_threshold", 10), ("conf_drop_threshold", 1)]:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (
            math.isfinite(value) and 0 < value <= limit
        ):
            raise ValueError("Invalid drift threshold in stream configuration")
    limit = config["max_rollback_evaluations"]
    if type(limit) is not int or not 0 <= limit <= 3:
        raise ValueError("Rollback evaluations must be an integer in 0..3")
    if not isinstance(config["budget"], dict) or set(config["budget"]) != {
        "max_training_seconds", "max_seconds_per_round", "max_updates", "max_llm_calls"
    }:
        raise ValueError("Unexpected resource budget configuration")
    return config, Budget(**config["budget"])


def execute_batches(
    registry: ModelRegistry, train: Dataset, batches: list[EvaluationBatch],
    initial: Path, task: str, config: dict, budget: Budget, recorder,
    input_guard: Callable[[], None] | None = None,
) -> list[dict]:
    """Caller holds the session lock. Commit cursor, version and charge together."""
    verify_sequence(batches, train)
    if not 0 <= registry.state["cursor"] <= len(batches):
        raise ValueError("Saved batch cursor is out of range")
    registry.recover_interruption()
    base = Model.load(initial)
    verify_parent(base, task, train)
    ref = fit_reference(train.x, base)
    atomic_json(registry.folder / "drift-reference.json", ref.summary())
    policy = RulePolicy()
    step = 0
    recorder.log(step, {"loop/start": 1.0, "loop/resumed_cursor": registry.state["cursor"]})
    for batch in batches[registry.state["cursor"]:]:
        round_started = monotonic()
        current = registry.state["current"]
        current_path = registry.checkpoint(current)
        incumbent = Model.load(current_path)
        # The detector/decision receives x only. Labels are reserved for the gate.
        signal = detect_drift(incumbent, ref, batch.data.x,
                              psi_threshold=config["psi_threshold"],
                              conf_drop_threshold=config["conf_drop_threshold"])
        diagnosis = attribute_drift(signal, incumbent.meta["paths"])
        plan = policy.decide(signal, diagnosis, budget, registry.state["events"])
        before, before_details, _ = score(incumbent, batch.data)
        step += 1
        logged = {"drift/mean_psi": signal.mean_psi,
                  "drift/features_over": signal.n_features_over,
                  "drift/detected": float(signal.detected),
                  **{f"before/{k}": v for k, v in before.items()}}
        if signal.confidence_drop is not None:
            logged["drift/confidence_drop"] = signal.confidence_drop
        recorder.log(step, logged)
        event = {"batch_index": batch.index, "batch_id": batch.identity,
                 "validation_source_sha256": batch.source_sha256,
                 "signal": signal.summary(), "diagnosis": diagnosis,
                 "before": before, "parent_version": current,
                 "candidate_version": None, "accepted": False,
                 "training_seconds": 0.0, "rollback_comparisons": []}
        after, after_details = before, before_details
        selected = current
        # Confidence loss requests evidence, never an unconditional rollback.
        if signal.confidence_drop is not None and signal.confidence_drop > config[
            "conf_drop_threshold"
        ]:
            versions = [r for r in registry.state["versions"]
                        if r["adopted"] and r["version"] != current]
            best = before
            for row in list(reversed(versions))[:config["max_rollback_evaluations"]]:
                old_metrics, old_details, _ = score(
                    Model.load(registry.checkpoint(row["version"])), batch.data
                )
                event["rollback_comparisons"].append(
                    {"version": row["version"], "same_batch_metrics": old_metrics}
                )
                if promotion(best, old_metrics, task)[0]:
                    selected, best = row["version"], old_metrics
                    after, after_details = old_metrics, old_details
            if selected != current:
                plan = UpdatePlan("rollback", rationale="historical_version_passed_same_batch_gate")
        remaining = budget.max_training_seconds - registry.state["training_seconds_charged"]
        if plan.action == "retrain" and (
            remaining <= 0 or registry.state["updates_charged"] >= budget.max_updates
        ):
            plan = UpdatePlan("hold", rationale="training_budget_exhausted")
        if plan.action == "retrain":
            allowance = min(remaining, budget.max_seconds_per_round)
            attempt = registry.state["updates_charged"] + 1
            version = max(r["version"] for r in registry.state["versions"]) + attempt
            candidate_path = registry.folder / f"attempt-{attempt}-version-{version}.npz"
            reservation = {"batch_index": batch.index, "attempt": attempt,
                           "reserved_seconds": allowance, "plan": asdict(plan)}
            # Crash before commit keeps the full reservation charged. No free retries.
            registry.state["inflight"] = reservation
            registry.state["updates_charged"] += 1
            registry.state["training_seconds_charged"] += allowance
            registry.save()
            candidate = Model.load(current_path)
            candidate.meta["learning_rate"] = plan.learning_rate
            initial_steps = candidate.meta["steps"]
            initial_epochs = candidate.meta["epochs"]
            training_seconds = 0.0
            for epoch in range(1, plan.epochs + 1):
                if monotonic() - round_started >= allowance:
                    raise TimeoutError("Stream cooperative round budget reached before training")
                started = perf_counter()
                loss = candidate.fit(train.x, train.y, 1)[0]
                training_seconds += perf_counter() - started
                step += 1
                recorder.log(step, {"train/loss": float(loss), "train/epoch": epoch,
                                    "train/seconds": training_seconds,
                                    "train/optimizer_steps": candidate.meta["steps"]})
            if monotonic() - round_started >= allowance:
                raise TimeoutError("Stream cooperative round budget exceeded; candidate rejected")
            if not (np.array_equal(candidate.mean, incumbent.mean)
                    and np.array_equal(candidate.scale, incumbent.scale)):
                raise ValueError("Continuation changed inherited normalization")
            expected_steps = initial_steps + plan.epochs * math.ceil(
                len(train.ids) / candidate.meta["batch_size"]
            )
            if (candidate.meta["steps"] != expected_steps
                    or candidate.meta["epochs"] != initial_epochs + plan.epochs):
                raise ValueError("Continuation optimizer/epoch counters do not match")
            after, after_details, _ = score(candidate, batch.data)
            accepted, gate = promotion(before, after, task)
            registry.checkpoint(current)
            if input_guard is not None:
                input_guard()
            candidate.save(candidate_path)
            registry.register(version, candidate_path, parent=current, adopted=accepted)
            selected = version if accepted else current
            registry.state["training_seconds_charged"] += training_seconds - allowance
            registry.state["inflight"] = None
            event.update({"accepted": accepted, "gate": gate, "candidate_version": version,
                          "training_seconds": training_seconds, "initial_steps": initial_steps,
                          "final_steps": candidate.meta["steps"]})
        elif plan.action == "rollback":
            registry.rollback(selected)
            event["accepted"] = True
        if input_guard is not None:
            input_guard()
        registry.state["current"] = selected
        event.update({"status": "completed", "plan": asdict(plan), "candidate_metrics": after,
                      "train_rows": len(train.ids), "validation_rows": len(batch.data.ids),
                      "selected_version": selected,
                      "selected_metrics": after if event["accepted"] else before,
                      "performance_diagnosis": attribute_performance(before_details, after_details),
                      "round_elapsed_seconds": monotonic() - round_started})
        step += 1
        recorder.log(step, {
            "decision/accepted": float(event["accepted"]),
            "decision/trained": float(plan.action == "retrain"),
            "decision/rollback": float(plan.action == "rollback"),
            "budget/training_seconds_charged": registry.state["training_seconds_charged"],
            **{f"selected/{k}": v for k, v in event["selected_metrics"].items()},
        })
        recorder.text(step, "diagnosis/report", {
            "batch_id": batch.identity, "plan": asdict(plan), "diagnosis": diagnosis,
            "performance": event["performance_diagnosis"],
        })
        atomic_json(registry.folder / f"batch-{batch.index}-attempt-result.json", event)
        registry.state["events"].append(event)
        registry.state["cursor"] = batch.index + 1
        registry.save()
    return registry.state["events"]


def run_stream(
    task: str, *, mode: str = "offline", project: str = "zhixun-school",
    workspace: str | None = None, resume: Path | None = None,
    config_path: Path | None = None, root: Path = ROOT, output: Path | None = None,
) -> dict:
    config, budget = load_config(config_path or root / "configs/stream.json")
    declared = verify_inputs(task, root)
    inputs = root / "data" / task
    initial = inputs / "checkpoint.npz"
    model = Model.load(initial)
    train = load_dataset(inputs / "train.jsonl", model, task)
    validation = load_dataset(inputs / "validation.jsonl", model, task)
    verify_parent(model, task, train)
    batches = [EvaluationBatch(0, "supplied-public-validation", validation,
                               digest(inputs / "validation.jsonl"))]
    verify_sequence(batches, train)
    source = {p.relative_to(root).as_posix(): digest(p)
              for p in sorted((root / "src").rglob("*.py"))}
    signature = {"task": task, "config": config, "dataset": declared, "source_sha256": source,
                 "batches": [{"index": b.index, "id": b.identity, "sha256": b.source_sha256}
                             for b in batches]}
    folder = (output or root / "artifacts") / (
        f"run-stream-{task}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    session = resume.resolve() if resume else folder
    if resume and (not session.is_dir() or session.is_symlink() or not (
        session / "stream-state.json"
    ).is_file()):
        raise ValueError("Resume requires a real existing stream experiment directory")
    registration = {
        "format": "zhixun.stream-experiment.v1", "experiment_id": folder.name,
        "created_at": datetime.now(UTC).isoformat(), **signature,
        "evaluation_scope": "one_unchanged_public_validation_file; not_official_hidden_stream",
        "ai_assisted_development": True,
        "session_directory": session.relative_to(root).as_posix()
        if session.is_relative_to(root) else str(session),
        "resumed": resume is not None,
    }
    write_json(folder / "registration.json", registration)
    recorder = None
    try:
        with process_lock(session):
            registry = ModelRegistry(session, signature, initial)
            recorder = Recorder(folder, registration, mode=mode, project=project,
                                workspace=workspace)
            events = execute_batches(
                registry, train, batches, initial, task, config, budget, recorder,
                input_guard=lambda: verify_inputs(task, root),
            )
            verify_inputs(task, root)
            selected_path = registry.checkpoint(registry.state["current"])
            shutil.copy2(selected_path, folder / "selected-checkpoint.npz")
            history = [json.loads(line) for line in (folder / "history.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()]
            with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["step", "metric", "value"])
                for row in history:
                    writer.writerows((row["step"], key, value)
                                     for key, value in row["metrics"].items())
            recorder.finish()
            report = {**registration, "status": "completed", "events": events,
                      "completed_batches": registry.state["cursor"],
                      "selected_version": registry.state["current"],
                      "selected_checkpoint_sha256": digest(selected_path),
                      "resources": {"training_seconds_charged": registry.state[
                          "training_seconds_charged"], "updates_charged": registry.state[
                              "updates_charged"], "llm_calls": 0,
                          "target_environment_measured": False},
                      "tracking": recorder.reference()}
            write_json(folder / "run.json", report)
            return report
    except BaseException as exc:
        write_json(folder / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                                             "message": str(exc), "registration": registration})
        if recorder is not None:
            try:
                recorder.finish(failed=True)
            except Exception:
                pass
        raise
