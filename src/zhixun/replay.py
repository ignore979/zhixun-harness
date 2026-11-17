"""Local UCI prequential experiments with explicit label-release and selection boundaries."""
from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, perf_counter
from uuid import uuid4

import numpy as np

from zhixun.attribution import attribute_drift
from zhixun.data import ROOT, digest, load_dataset, verify_inputs, verify_parent, write_json
from zhixun.drift import detect_drift, fit_reference
from zhixun.loop import EvaluationBatch, verify_sequence
from zhixun.metrics import promotion, score
from zhixun.model_extension import ClassWeightedModel, class_weights
from zhixun.pool import TrainingPool
from zhixun.prepare_gas import verify_replay
from zhixun.registry import ModelRegistry, atomic_json, process_lock
from zhixun.tracking import Recorder
from zhixun.training_schedule import lr_at, schedule_epochs
from zhixun.vendor.model import Model

STRATEGIES = {"static", "always", "performance"}


def load_replay_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {"protocol", "strategies", "performance_metric", "performance_drop_threshold",
                "psi_threshold", "conf_drop_threshold", "epochs", "learning_rate", "max_updates",
                "max_training_seconds", "max_seconds_per_round", "max_rollback_evaluations",
                "labels_available", "pool_release", "normalization"}
    if set(config) not in (expected, expected | {"training_upgrade"}) or set(
        config["strategies"]
    ) != STRATEGIES:
        raise ValueError("Unexpected replay configuration")
    if (config["protocol"] != "uci-gas-prequential-local-replay-v1"
            or config["performance_metric"] != "macro_f1"
            or config["labels_available"] != "after_prediction_before_decision"
            or config["pool_release"] != "after_decision"
            or config["normalization"] != "original_batch1_unchanged"):
        raise ValueError("Replay protocol or label-release declaration changed")
    limits = {"performance_drop_threshold": (0, 1), "psi_threshold": (0, 10),
              "conf_drop_threshold": (0, 1), "learning_rate": (0, 0.01),
              "max_training_seconds": (0, 3600), "max_seconds_per_round": (0, 300)}
    for key, (lower, upper) in limits.items():
        value = config[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not lower < value <= upper):
            raise ValueError(f"Invalid replay setting: {key}")
    for key, upper in [("epochs", 50), ("max_updates", 9), ("max_rollback_evaluations", 3)]:
        if type(config[key]) is not int or not 0 <= config[key] <= upper:
            raise ValueError(f"Invalid replay setting: {key}")
    if not config["epochs"]:
        raise ValueError("Replay training epochs must be positive")
    if "training_upgrade" in config:
        upgrade = config["training_upgrade"]
        if not isinstance(upgrade, dict) or set(upgrade) != {
            "cw_power", "adaptive_epochs", "max_epochs", "lr_decay", "pool_cap_rows"
        }:
            raise ValueError("Unexpected training upgrade configuration")
        power = upgrade["cw_power"]
        if power is not None and (isinstance(power, bool) or not isinstance(power, (int, float))
                                  or not math.isfinite(power) or not 0 <= power <= 2):
            raise ValueError("Invalid class-weight power")
        if (type(upgrade["adaptive_epochs"]) is not bool
                or type(upgrade["max_epochs"]) is not int
                or not config["epochs"] <= upgrade["max_epochs"] <= 20):
            raise ValueError("Invalid adaptive epoch limit")
        decay = upgrade["lr_decay"]
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 < decay <= 1:
            raise ValueError("Invalid learning-rate decay")
        cap = upgrade["pool_cap_rows"]
        if cap is not None and (type(cap) is not int or not 10 <= cap <= 20000):
            raise ValueError("Invalid historical-pool sample cap")
    return config


def decide(strategy: str, before: dict, reference: float, signal, config: dict,
           history: list[dict]) -> dict:
    drop = reference - before["macro_f1"]
    if strategy == "static":
        action, reason = "hold", "static_baseline"
    elif strategy == "always":
        action, reason = "retrain", "always_train_control"
    elif drop > config["performance_drop_threshold"]:
        action, reason = "retrain", "observed_macro_f1_drop"
    else:
        action, reason = "hold", "performance_stable_despite_possible_input_shift"
    rate = config["learning_rate"] if action == "retrain" else 0.0
    if strategy == "performance" and action == "retrain":
        past = [r for r in history if r.get("plan", {}).get("action") == "retrain"]
        if past and not past[-1]["accepted"]:
            rate = past[-1]["plan"]["learning_rate"] / 2
            reason += "; previous_rejected_reduce_lr"
    return {"action": action, "epochs": config["epochs"] if action == "retrain" else 0,
            "learning_rate": rate, "rationale": reason}


def execute_replay(registry, train, batches, initial, strategy, config, recorder,
                   input_guard=None) -> list[dict]:
    verify_sequence(batches, train)
    registry.recover_interruption()
    cursor = registry.state["cursor"]
    if not 0 <= cursor <= len(batches):
        raise ValueError("Replay cursor is out of range")
    pool = TrainingPool.from_dataset(train)
    for batch in batches[:cursor]:
        pool.append(batch.data, batch_id=batch.identity)
    if "pool" in registry.state and registry.state["pool"] != pool.summary():
        raise ValueError("Recovered training pool differs from committed provenance")
    base = Model.load(initial)
    verify_parent(base, "A", train)  # Always check against batch1, never the growing pool.
    reference = fit_reference(train.x, base)
    atomic_json(registry.folder / "drift-reference.json", reference.summary())
    for batch in batches[cursor:]:
        round_started = monotonic()
        pool.verify_future(batch.data)
        parent_version = registry.state["current"]
        parent = registry.checkpoint(parent_version)
        incumbent = Model.load(parent)
        # Persist predictions from the incoming incumbent before any label-based action/selection.
        before, before_details, predicted = score(incumbent, batch.data)
        prediction_path = registry.folder / f"prequential-batch-{batch.index}.jsonl"
        with prediction_path.open("w", encoding="utf-8") as stream:
            for identity, probabilities in zip(batch.data.ids, predicted, strict=True):
                stream.write(json.dumps({"id": identity, "probabilities": probabilities.tolist()})
                             + "\n")
        if "reference_macro_f1" not in registry.state:
            registry.state["reference_macro_f1"] = before["macro_f1"]
            registry.save()
        reference_metric = registry.state["reference_macro_f1"]
        signal = detect_drift(incumbent, reference, batch.data.x,
                              psi_threshold=config["psi_threshold"],
                              conf_drop_threshold=config["conf_drop_threshold"])
        plan = decide(strategy, before, reference_metric, signal, config, registry.state["events"])
        upgrade = config.get("training_upgrade", {})
        if plan["action"] == "retrain" and upgrade.get("adaptive_epochs"):
            plan["epochs"] = schedule_epochs(
                signal.mean_psi, reference_metric - before["macro_f1"],
                pool.summary()["rows"], base_epochs=config["epochs"],
                max_epochs=upgrade["max_epochs"],
                drop_threshold=config["performance_drop_threshold"],
            )
        event = {"status": "completed", "batch_index": batch.index, "batch_id": batch.identity,
                 "strategy": strategy, "source_sha256": batch.source_sha256,
                 "rows": len(batch.data.ids), "prequential": before,
                 "reference_macro_f1": reference_metric,
                 "performance_drop": reference_metric - before["macro_f1"],
                 "performance_drift": reference_metric - before["macro_f1"] > config[
                     "performance_drop_threshold"],
                 "signal": signal.summary(),
                 "diagnosis": attribute_drift(signal, incumbent.meta["paths"]),
                 "pool_before": pool.summary(), "parent_version": parent_version,
                 "accepted": False, "training_seconds": 0.0, "rollback_comparisons": [],
                 "budget_before": {"updates_remaining": config["max_updates"] - registry.state[
                     "updates_charged"], "training_seconds_remaining": config[
                         "max_training_seconds"] - registry.state["training_seconds_charged"]}}
        event["per_class_before"] = before_details
        selected, selected_metrics = parent_version, before
        selected_details = before_details
        if (strategy == "performance" and event["performance_drift"]
                and signal.confidence_drop is not None
                and signal.confidence_drop > config["conf_drop_threshold"]):
            versions = [v for v in registry.state["versions"]
                        if v["adopted"] and v["version"] != parent_version]
            for row in list(reversed(versions))[:config["max_rollback_evaluations"]]:
                metrics, details, _ = score(
                    Model.load(registry.checkpoint(row["version"])), batch.data)
                event["rollback_comparisons"].append(
                    {"version": row["version"], "metrics": metrics}
                )
                if promotion(selected_metrics, metrics, "A")[0]:
                    selected, selected_metrics = row["version"], metrics
                    selected_details = details
            if selected != parent_version:
                plan = {"action": "rollback", "epochs": 0, "learning_rate": 0.0,
                        "rationale": "historical_version_passed_current_batch_gate"}
                event["accepted"] = True
        remaining = config["max_training_seconds"] - registry.state["training_seconds_charged"]
        if plan["action"] == "retrain" and (remaining <= 0 or registry.state[
            "updates_charged"
        ] >= config["max_updates"]):
            plan = {"action": "hold", "epochs": 0, "learning_rate": 0.0,
                    "rationale": "training_budget_exhausted"}
        if plan["action"] == "retrain":
            allowance = min(remaining, config["max_seconds_per_round"])
            attempt = registry.state["updates_charged"] + 1
            version = max(v["version"] for v in registry.state["versions"]) + attempt
            registry.state["inflight"] = {"batch_index": batch.index, "attempt": attempt,
                                          "reserved_seconds": allowance}
            registry.state["updates_charged"] += 1
            registry.state["training_seconds_charged"] += allowance
            registry.save()
            power = upgrade.get("cw_power")
            candidate = (ClassWeightedModel if power is not None else Model).load(parent)
            candidate.meta["learning_rate"] = plan["learning_rate"]
            steps, epochs = candidate.meta["steps"], candidate.meta["epochs"]
            expected_steps = steps
            event["epoch_training"] = []
            for epoch in range(plan["epochs"]):
                if monotonic() - round_started >= allowance:
                    raise TimeoutError("Replay cooperative round budget reached")
                sample, sampling = pool.sample(
                    cap_rows=upgrade.get("pool_cap_rows"), seed=candidate.meta["epochs"] + 42)
                x, y = sample.x, sample.y
                rate = lr_at(epoch, plan["epochs"], plan["learning_rate"],
                             decay=upgrade.get("lr_decay", 1.0))
                candidate.meta["learning_rate"] = rate
                counts = np.bincount(y, minlength=len(candidate.meta["classes"]))
                weights = class_weights(y, len(counts), power) if power is not None else None
                started = perf_counter()
                loss = (candidate.fit(x, y, 1, cw_power=power)[0] if power is not None
                        else candidate.fit(x, y, 1)[0])
                event["training_seconds"] += perf_counter() - started
                expected_steps += math.ceil(len(y) / candidate.meta["batch_size"])
                event["epoch_training"].append({"epoch": epoch + 1, "learning_rate": rate,
                                                "sampling": sampling,
                                                "class_counts": counts.tolist(),
                                                "class_weights": weights.tolist()
                                                if weights is not None else None})
                recorder.log(batch.index * 100 + epoch + 1, {
                    "train/loss": float(loss), "train/pool_rows": len(y),
                    "train/optimizer_steps": candidate.meta["steps"],
                    "train/seconds": event["training_seconds"],
                    "train/learning_rate": rate, "train/planned_epochs": plan["epochs"],
                })
            if monotonic() - round_started >= allowance:
                raise TimeoutError("Replay cooperative round budget exceeded; candidate rejected")
            if not (np.array_equal(candidate.mean, base.mean)
                    and np.array_equal(candidate.scale, base.scale)):
                raise ValueError("Growing pool changed original normalization")
            if candidate.meta["steps"] != expected_steps or candidate.meta["epochs"] != (
                epochs + plan["epochs"]
            ):
                raise ValueError("Growing-pool optimizer continuation counters are inconsistent")
            candidate_metrics, candidate_details, _ = score(candidate, batch.data)
            accepted, gate = promotion(before, candidate_metrics, "A")
            path = registry.folder / f"attempt-{attempt}-version-{version}.npz"
            if input_guard:
                input_guard()
            registry.checkpoint(parent_version)
            candidate.save(path)
            registry.register(version, path, parent=parent_version, adopted=accepted)
            if accepted:
                selected, selected_metrics = version, candidate_metrics
                selected_details = candidate_details
            registry.state["training_seconds_charged"] += event["training_seconds"] - allowance
            registry.state["inflight"] = None
            event.update({"accepted": accepted, "candidate_version": version,
                          "candidate_metrics": candidate_metrics, "gate": gate,
                          "initial_steps": steps, "final_steps": candidate.meta["steps"]})
            event["per_class_candidate"] = candidate_details
        elif plan["action"] == "rollback":
            registry.rollback(selected)
        if input_guard:
            input_guard()
        event.update({"plan": plan, "selected_version": selected,
                      "selected_same_batch": selected_metrics,
                      "per_class_selected": selected_details})
        # Labels enter the pool only after predictions, decision, training and selection.
        pool.append(batch.data, batch_id=batch.identity)
        registry.state["pool"] = pool.summary()
        registry.state["current"] = selected
        registry.state["cursor"] = batch.index + 1
        event["pool_after"] = registry.state["pool"]
        event["round_elapsed_seconds"] = monotonic() - round_started
        recorder.log(batch.index * 100 + 90, {
            "batch/number": batch.index + 2, "drift/mean_psi": signal.mean_psi,
            "drift/performance_drop": event["performance_drop"],
            "decision/trained": float(plan["action"] == "retrain"),
            "decision/accepted": float(event["accepted"]),
            "decision/rollback": float(plan["action"] == "rollback"),
            "pool/rows_after": registry.state["pool"]["rows"],
            "budget/updates_charged": registry.state["updates_charged"],
            **{f"prequential/{k}": v for k, v in before.items()},
            **{f"selected_same_batch/{k}": v for k, v in selected_metrics.items()},
            **{f"class_before/{name}/f1": before_details[name]["f1"]
               for name in incumbent.meta["classes"]},
        })
        recorder.text(batch.index * 100 + 90, "diagnosis/report", {
            "batch": batch.identity, "diagnosis": event["diagnosis"], "plan": plan,
            "pool_before": event["pool_before"]["provenance"],
        })
        registry.state["events"].append(event)
        registry.save()  # Cursor, version, budget and pool release commit atomically.
    return registry.state["events"]


def run_replay(strategy: str, *, mode="offline", project="zhixun-school", workspace=None,
               config_path: Path | None = None, resume: Path | None = None,
               root: Path = ROOT, output: Path | None = None) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError("Unknown replay strategy")
    started = monotonic()
    config = load_replay_config(config_path or root / "configs/replay.json")
    declared, replay = verify_inputs("A", root), verify_replay(root)
    initial = root / "data/A/checkpoint.npz"
    model = Model.load(initial)
    train = load_dataset(root / "data/A/train.jsonl", model, "A")
    paths = [root / "data/A/validation.jsonl"] + [root / "data/replay-gas" / r["file"]
                                                       for r in replay["batches"]]
    batches = [EvaluationBatch(i, f"gas-b{i + 2:02d}", load_dataset(path, model, "A"), digest(path))
               for i, path in enumerate(paths)]
    source = {p.relative_to(root).as_posix(): digest(p)
              for p in [root / "run.py", *sorted((root / "src").rglob("*.py"))]}
    signature = {"strategy": strategy, "config": config, "dataset": declared,
                 "replay_manifest_sha256": digest(root / "data/replay-gas/manifest.json"),
                 "source_sha256": source,
                 "batches": [{"id": b.identity, "sha256": b.source_sha256} for b in batches]}
    folder = (output or root / "artifacts") / (
        f"run-replay-{strategy}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    session = resume.resolve() if resume else folder
    if resume and not (session / "stream-state.json").is_file():
        raise ValueError("Resume requires an existing replay session")
    registration = {"format": "zhixun.replay-experiment.v1", "experiment_id": folder.name,
                    "created_at": datetime.now(UTC).isoformat(), **signature,
                    "evaluation_scope": "local_UCI_labeled_replay_not_contest_hidden_score",
                    "primary_report": "prediction_before_current_label_based_decision",
                    "ai_assisted_development": True, "resumed": resume is not None,
                    "numpy_version": np.__version__}
    write_json(folder / "registration.json", registration)
    recorder = None

    def guard():
        verify_inputs("A", root)
        if digest(root / "data/replay-gas/manifest.json") != signature["replay_manifest_sha256"]:
            raise ValueError("Replay manifest changed during experiment")
        for path, batch in zip(paths, batches, strict=True):
            if digest(path) != batch.source_sha256:
                raise ValueError("Replay data changed during experiment")

    try:
        with process_lock(session):
            registry = ModelRegistry(session, signature, initial)
            recorder = Recorder(folder, registration, mode=mode, project=project,
                                workspace=workspace)
            events = execute_replay(registry, train, batches, initial, strategy,
                                    config, recorder, guard)
            completed = [e for e in events if e["status"] == "completed"]
            total = sum(e["rows"] for e in completed)
            summary = {
                "prequential_sample_weighted_accuracy": sum(e["prequential"]["accuracy"] * e[
                    "rows"] for e in completed) / total,
                "prequential_mean_batch_macro_f1": float(np.mean([
                    e["prequential"]["macro_f1"] for e in completed])),
                "selected_same_batch_sample_weighted_accuracy": sum(e["selected_same_batch"][
                    "accuracy"] * e["rows"] for e in completed) / total,
                "selected_same_batch_mean_batch_macro_f1": float(np.mean([
                    e["selected_same_batch"]["macro_f1"] for e in completed])),
                "updates_charged": registry.state["updates_charged"],
                "training_seconds_charged": registry.state["training_seconds_charged"],
            }
            recorder.log(1000, {f"summary/{k}": v for k, v in summary.items()})
            shutil.copy2(registry.checkpoint(registry.state["current"]),
                         folder / "selected-checkpoint.npz")
            with (folder / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["batch", "accuracy_before", "macro_f1_before", "action",
                                 "pool_rows_before", "accuracy_selected_same_batch"])
                for e in completed:
                    writer.writerow([e["batch_id"], e["prequential"]["accuracy"],
                                     e["prequential"]["macro_f1"], e["plan"]["action"],
                                     e["pool_before"]["rows"],
                                     e["selected_same_batch"]["accuracy"]])
            recorder.finish()
            report = {**registration, "status": "completed", "events": events,
                      "summary": summary, "completed_batches": registry.state["cursor"],
                      "selected_version": registry.state["current"],
                      "elapsed_seconds": monotonic() - started,
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
