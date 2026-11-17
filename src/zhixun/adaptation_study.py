"""Prequential prediction-only ablations on immutable real historical model trajectories."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

import numpy as np

from zhixun.data import ROOT, digest, write_json
from zhixun.fair_core import load_fair_data
from zhixun.metrics import classification_metrics
from zhixun.prior_correction import (
    PriorCorrectedModel,
    bbse_prior,
    class_prior,
    coral_probabilities,
    reweight,
)
from zhixun.tracking import Recorder
from zhixun.vendor.model import Model


def summarize(events, classes, batches):
    selected = [e for e in events if e["batch"] in batches]
    total = sum(e["rows"] for e in selected)
    matrix = sum((np.asarray(e["details"]["confusion_matrix"]) for e in selected),
                 np.zeros((len(classes), len(classes)), dtype=int))
    recall = {c: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else 0.0
              for i, c in enumerate(classes)}
    return {"accuracy": sum(e["metrics"]["accuracy"] * e["rows"] for e in selected) / total,
            "mean_macro_f1": float(np.mean([e["metrics"]["macro_f1"] for e in selected])),
            "mean_g_mean": float(np.mean([e["metrics"]["g_mean"] for e in selected])),
            "mean_kappa": float(np.mean([e["metrics"]["cohen_kappa"] for e in selected])),
            "pooled_per_class_recall": recall, "confusion_matrix": matrix.tolist(), "rows": total}


def run_adaptation_study(*, root=ROOT, mode="offline", project="zhixun-school"):
    config_path = root / "configs/targeted_study.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _, base, train, batches = load_fair_data(root)
    classes = base.meta["classes"]
    study = root / "artifacts" / f"study-targeted-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    study.mkdir(parents=True, exist_ok=False)
    write_json(study / "registration.json", {"config": config, "sha256": digest(config_path)})
    results = []
    for source in config["source_runs"]:
        source_folder = root / "artifacts" / source
        report = json.loads((source_folder / "run.json").read_text(encoding="utf-8"))
        registry = json.loads((source_folder / "stream-state.json").read_text(encoding="utf-8"))
        versions = {v["version"]: v for v in registry["versions"]}
        source_events = [e for e in report["events"] if e["status"] == "completed"]
        if len(source_events) != 9:
            raise ValueError("Adaptation requires a complete nine-batch source trajectory")
        for method in config["methods"]:
            folder = root / "artifacts" / (f"run-adapt-{method['name']}-"
                                           f"{datetime.now(UTC):%Y%m%dT%H%M%S}-"
                                           f"{uuid4().hex[:8]}")
            folder.mkdir(exist_ok=False)
            registration = {
                "experiment_id": folder.name, "created_at": datetime.now(UTC).isoformat(),
                "task": "prediction_only_temporal_ablation", "method": method,
                "source_run": source, "source_report_sha256": digest(source_folder / "run.json"),
                "study_config_sha256": digest(config_path), "study_config": config,
                "normalization": "unchanged_original_batch1",
                "current_labels_used_by_prediction": False, "additional_training_epochs": 0,
                "source_sha256": {p.relative_to(root).as_posix(): digest(p)
                                   for p in sorted((root / "src").rglob("*.py"))},
            }
            write_json(folder / "registration.json", registration)
            recorder, events = None, []
            try:
                recorder = Recorder(folder, registration, mode=mode, project=project,
                                    workspace=None)
                pc = PriorCorrectedModel(base, window=3, strength=method.get("strength", 1),
                                         smoothing=method.get("smoothing", 1e-6))
                pc.fit_prior(train.y)
                for index, batch in enumerate(batches):
                    event = source_events[index]
                    if event["batch_index"] != index:
                        raise ValueError("Source order changed")
                    version = versions[event["parent_version"]]
                    path = source_folder / version["path"]
                    if digest(path) != version["sha256"]:
                        raise ValueError("Historical checkpoint changed")
                    parent = Model.load(path)
                    if (not np.array_equal(parent.mean, base.mean)
                            or not np.array_equal(parent.scale, base.scale)):
                        raise ValueError("Source normalization changed")
                    started = perf_counter()
                    raw = parent.predict(batch.data.x)
                    audit = {"released_label_batches": pc.released_batches.copy(),
                             "target_labels_used": False}
                    kind = method["kind"]
                    if kind == "identity":
                        prediction = raw
                    elif kind == "history":
                        pc.model = parent
                        prediction = pc.predict(batch.data.x)
                        audit.update(source_prior=pc.p_train.tolist(),
                                     estimated_target_prior=pc.p_test().tolist())
                    elif kind == "bbse":
                        prior, diagnostics = bbse_prior(parent.predict(train.x), train.y, raw,
                                                        ridge=method["ridge"])
                        prediction = reweight(raw, prior, class_prior(train.y, len(classes)),
                                              method["strength"])
                        audit.update(diagnostics, estimated_target_prior=prior.tolist())
                    elif kind == "coral":
                        prediction = coral_probabilities(parent, train.x, batch.data.x,
                                                         strength=method["strength"],
                                                         ridge=method["ridge"])
                    else:
                        raise ValueError("Unregistered adaptation")
                    seconds = perf_counter() - started
                    # The predictions are durable before accessing labels for scoring/release.
                    prediction_path = folder / f"prequential-batch-{index + 2}.npz"
                    np.savez_compressed(prediction_path, probabilities=prediction,
                                        ids=np.array(batch.data.ids))
                    metrics, details = classification_metrics(prediction, batch.data.y, classes)
                    if kind == "identity" and any(not np.isclose(metrics[k], event[
                            "prequential"][k], atol=1e-12) for k in ["accuracy", "macro_f1"]):
                        raise ValueError("Identity did not reproduce source prequential metrics")
                    row = {"batch": index + 2, "rows": len(batch.data.ids), "metrics": metrics,
                           "details": details, "adaptation": audit, "prediction_seconds": seconds,
                           "prediction_sha256": digest(prediction_path),
                           "parent_checkpoint_sha256": version["sha256"],
                           "batch_data_sha256": batch.source_sha256}
                    events.append(row)
                    recorder.log(index, {**{f"prequential/{k}": v for k, v in metrics.items()},
                        **{f"recall/{c}": details[c]["recall"] for c in classes},
                        "prediction/seconds": seconds, "batch/number": index + 2})
                    write_json(folder / "batch-results.json", events)
                    pc.release(batch.data.y, batch=index + 2)
                summaries = {"full": summarize(events, classes, list(range(2, 11))),
                             "development": summarize(events, classes,
                                                      config["development_batches"]),
                             "audit": summarize(events, classes, config["audit_batches"])}
                recorder.log(100, {f"summary/{scope}/{k}": v
                                  for scope, summary in summaries.items()
                                  for k, v in summary.items() if isinstance(v, (int, float))})
                recorder.finish()
                result = {**registration, "status": "completed", "events": events,
                          "summary": summaries, "tracking": recorder.reference(),
                          "parent_training_epochs": sum(e.get("action", e.get("plan"))[
                              "epochs"] for e in source_events),
                          "prediction_seconds": sum(e["prediction_seconds"] for e in events)}
                write_json(folder / "run.json", result)
                results.append({"experiment_id": folder.name, "source": source,
                                "method": method["name"], "summary": summaries})
                write_json(study / "study.json", {"status": "running", "config": config,
                                                   "results": results})
            except BaseException as exc:
                write_json(folder / "failure.json", {"status": "failed", "message": str(exc)})
                if recorder:
                    recorder.finish(failed=True)
                raise
    selections = {}
    for source in config["source_runs"]:
        rows = [r for r in results if r["source"] == source]
        base_row = next(r for r in rows if r["method"] == "identity")
        baseline = base_row["summary"]["development"]
        eligible = [r for r in rows if r["summary"]["development"]["accuracy"] > baseline[
            "accuracy"] and r["summary"]["development"]["mean_macro_f1"] >= baseline[
            "mean_macro_f1"]]
        selected = max(eligible, key=lambda r: r["summary"]["development"]["accuracy"],
                       default=base_row)
        selections[source] = {"method": selected["method"],
                              "selection_uses_only_batches": config["development_batches"],
                              "audit_summary": selected["summary"]["audit"],
                              "experiment_id": selected["experiment_id"]}
    final = {"status": "completed", "config": config, "results": results,
             "selections": selections, "study_directory": study.relative_to(root).as_posix()}
    write_json(study / "study.json", final)
    return final
