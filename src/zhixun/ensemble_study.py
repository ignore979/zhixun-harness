"""Registered two-stage temporal ensemble study; audit metrics never select a case."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

import numpy as np

from zhixun.adaptation_study import summarize
from zhixun.data import ROOT, digest, write_json
from zhixun.ensemble import EnsemblePredictor, temporal_versions
from zhixun.fair_core import load_fair_data
from zhixun.metrics import classification_metrics
from zhixun.prior_correction import CoralAdapter
from zhixun.registry import ModelRegistry
from zhixun.tracking import Recorder


def safe_candidate(candidate, baseline, *, strict_accuracy=True):
    accuracy = candidate["accuracy"] > baseline["accuracy"] if strict_accuracy else (
        candidate["accuracy"] >= baseline["accuracy"])
    return bool(accuracy and all(candidate[k] >= baseline[k] for k in [
        "mean_macro_f1", "mean_g_mean"]) and candidate["pooled_per_class_recall"][
            "Acetaldehyde"] >= baseline["pooled_per_class_recall"]["Acetaldehyde"])


def select_development(rows, baseline):
    eligible = [r for r in rows if safe_candidate(r["summary"]["development"],
                                                 baseline["summary"]["development"])]
    return max(eligible, key=lambda r: (r["summary"]["development"]["accuracy"],
                                        r["summary"]["development"]["mean_g_mean"],
                                        -r["spec"]["strength"]), default=baseline)


def run_ensemble_study(*, root=ROOT, mode="offline", project="zhixun-school"):
    config_path = root / "configs/ensemble_study.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _, base, train, batches = load_fair_data(root)
    classes = base.meta["classes"]
    study = root / "artifacts" / f"study-ensemble-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    study.mkdir(parents=True, exist_ok=False)
    write_json(study / "registration.json", {"config": config, "sha256": digest(config_path),
                                             "created_at": datetime.now(UTC).isoformat()})
    outcomes, transforms = [], {}

    def transformed(batch_index, strength, ridge):
        if strength == 0:
            return None
        key = (batch_index, strength, ridge)
        if key not in transforms:
            adapter = CoralAdapter.from_source(base, train.x, strength=strength, ridge=ridge)
            transforms[key] = adapter.transform(base, batches[batch_index].data.x)
        return transforms[key]

    for source in config["source_runs"]:
        source_folder = root / "artifacts" / source
        source_report = json.loads((source_folder / "run.json").read_text(encoding="utf-8"))
        full_state = json.loads((source_folder / "stream-state.json").read_text(encoding="utf-8"))
        if source_report["status"] != "completed":
            raise ValueError("Source trajectory must be complete")
        registry = ModelRegistry(source_folder, full_state["signature"])
        source_events = [e for e in source_report["events"] if e["status"] == "completed"]
        if len(source_events) != 9:
            raise ValueError("Source must have nine real time batches")
        baseline, first_choice, stages = None, None, []
        for stage in [1, 2]:
            if stage == 1:
                specs = [{**v, "strength": strength, "ridge": 1.}
                         for v in config["ensemble_variants"]
                         for strength in config["first_stage_strengths"]]
            else:
                specs = [{**first_choice["spec"], "strength": a, "ridge": ridge}
                         for a in config["strength_grid"] for ridge in config["ridge_grid"]]
            folder = root / "artifacts" / (f"run-ensemble-s{stage}-"
                                           f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}")
            folder.mkdir(exist_ok=False)
            registration = {
                "experiment_id": folder.name, "created_at": datetime.now(UTC).isoformat(),
                "task": "temporal_ensemble_prediction_only", "stage": stage, "config": config,
                "specifications": specs, "source_run": source,
                "source_report_sha256": digest(source_folder / "run.json"),
                "source_registry_sha256": digest(source_folder / "stream-state.json"),
                "source_sha256": {p.relative_to(root).as_posix(): digest(p)
                                   for p in sorted((root / "src").rglob("*.py"))},
                "future_checkpoint_policy": "only_initial_and_past_committed_selected_versions",
                "current_target_labels_used_in_prediction": False,
                "additional_training_epochs": 0,
                "reliability": "past_predictions_only_no_retrospective_training_set_scoring",
                "timing_scope": "shared_geometry_cache_screening_not_deployment_latency_benchmark",
            }
            write_json(folder / "registration.json", registration)
            recorder, rows = None, []
            try:
                recorder = Recorder(folder, registration, mode=mode, project=project,
                                    workspace=None)
                for case_index, spec in enumerate(specs):
                    case_folder = folder / f"case-{case_index:03d}"
                    case_folder.mkdir()
                    events, history = [], {}
                    for index, batch in enumerate(batches):
                        started = perf_counter()
                        allowed = temporal_versions(source_events, index)
                        ensemble = EnsemblePredictor(
                            registry, k=spec["k"], decay=spec["decay"], mode=spec["mode"],
                            allowed_versions=allowed,
                            current_version=source_events[index]["parent_version"],
                            window=config["reliability_window"],
                            shrinkage=config["reliability_shrinkage"], history=history)
                        transform = transformed(index, spec["strength"], spec["ridge"])
                        prediction = ensemble.predict(batch.data.x, "classification",
                                                      transform=transform, batch=index + 2)
                        seconds = perf_counter() - started
                        path = case_folder / f"batch-{index + 2:02d}.npz"
                        np.savez_compressed(path, probabilities=prediction,
                                            ids=np.array(batch.data.ids))
                        # No label is used by inference; only the durable predictions are scored.
                        metrics, details = classification_metrics(prediction, batch.data.y, classes)
                        audit = ensemble.last_audit
                        if not set(audit["versions"]) <= allowed:
                            raise ValueError("Temporal model leakage")
                        row = {"batch": index + 2, "rows": len(batch.data.ids), "metrics": metrics,
                               "details": details, "ensemble": audit, "prediction_seconds": seconds,
                               "prediction_sha256": digest(path),
                               "data_sha256": batch.source_sha256}
                        events.append(row)
                        recorder.log(case_index * 100 + index, {
                            **{f"prequential/{k}": v for k, v in metrics.items()},
                            **{f"recall/{c}": details[c]["recall"] for c in classes},
                            "batch/number": index + 2, "case/number": case_index,
                            "ensemble/members": len(audit["versions"]),
                            "prediction/seconds": seconds})
                        ensemble.release(batch.data.y, batch=index + 2)
                        history = ensemble.history
                    summaries = {"full": summarize(events, classes, list(range(2, 11))),
                                 "development": summarize(events, classes,
                                                          config["development_batches"]),
                                 "audit": summarize(events, classes, config["audit_batches"])}
                    row = {"case": case_index, "spec": spec, "summary": summaries,
                           "events": events, "experiment_id": folder.name,
                           "case_directory": str(case_folder.relative_to(root))}
                    rows.append(row)
                    write_json(case_folder / "result.json", row)
                    recorder.log(case_index * 100 + 90, {
                        f"summary/{scope}/{k}": v for scope, summary in summaries.items()
                        for k, v in summary.items() if isinstance(v, (int, float))})
                if stage == 1:
                    baseline = next(r for r in rows if r["spec"]["name"] == "k1" and r[
                        "spec"]["strength"] == .25)
                    first_choice = select_development(rows, baseline)
                    selected = first_choice
                else:
                    selected = select_development([first_choice, *rows], baseline)
                # Selection function sees only development fields; audit is evaluated afterward.
                accepted = safe_candidate(selected["summary"]["audit"],
                                          baseline["summary"]["audit"])
                recorder.log(10000, {"selection/accepted_audit_gate": float(accepted),
                                     "selection/strength": selected["spec"]["strength"]})
                for index, (selected_event, base_event) in enumerate(zip(
                        selected["events"], baseline["events"], strict=True)):
                    recorder.log(11000 + index, {
                        **{f"selected/{k}": v for k, v in selected_event["metrics"].items()},
                        **{f"reference/{k}": v for k, v in base_event["metrics"].items()},
                        "selected/acetaldehyde_recall": selected_event["details"][
                            "Acetaldehyde"]["recall"],
                        "reference/acetaldehyde_recall": base_event["details"][
                            "Acetaldehyde"]["recall"], "selected/batch": index + 2})
                recorder.finish()
                report = {**registration, "status": "completed", "results": rows,
                          "selected": selected, "audit_accepted": accepted,
                          "tracking": recorder.reference()}
                write_json(folder / "run.json", report)
                stages.append({"stage": stage, "experiment_id": folder.name,
                               "selected": selected, "audit_accepted": accepted})
            except BaseException as exc:
                write_json(folder / "failure.json", {"status": "failed", "message": str(exc)})
                if recorder:
                    recorder.finish(failed=True)
                raise
        outcomes.append({"source": source, "baseline": baseline, "stages": stages})
    result = {"status": "completed", "config": config, "sources": outcomes,
              "study_directory": str(study.relative_to(root)),
              "selection_uses": "batch2_to7_only; audit_reused_public_development"}
    write_json(study / "study.json", result)
    return result
